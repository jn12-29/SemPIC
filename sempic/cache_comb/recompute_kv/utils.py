from collections.abc import Callable
from dataclasses import dataclass
from typing import NotRequired, TypedDict

import torch
from transformers.models.qwen3.modeling_qwen3 import repeat_kv
from ...cache.abc import KeyValue
from ...model import SupportedModel

class RecomputeResult(TypedDict):
    recomputed_hidden_states: torch.Tensor
    kv_from_hs: KeyValue
    query_states: torch.Tensor | None
    attention_probs: NotRequired[torch.Tensor]


@dataclass(frozen=True, slots=True)
class RecomputeAttentionBasis:
    scaled_masked_logits: torch.Tensor
    attention_probabilities: torch.Tensor
    physical_values: torch.Tensor
    query_to_kv_head: torch.LongTensor
    keep_mask: torch.BoolTensor


AttentionBasisSink = Callable[[RecomputeAttentionBasis], None]


def eager_attention_with_basis(
    *,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    keep_mask: torch.Tensor,
    scaling: float,
    sink: AttentionBasisSink,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eager attention once and synchronously expose its ephemeral basis."""
    if keep_mask.dtype != torch.bool:
        raise TypeError("Attention basis requires a boolean keep mask.")
    repeated_key = repeat_kv(key, module.num_key_value_groups)
    repeated_value = repeat_kv(value, module.num_key_value_groups)
    logits = torch.matmul(query, repeated_key.transpose(2, 3)) * scaling
    additive_mask = adapt_recompute_mask(
        keep_mask,
        attn_implementation="eager",
        dtype=query.dtype,
    )
    logits = logits + additive_mask
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    query_to_kv_head = (
        torch.arange(query.size(1), device=query.device, dtype=torch.long)
        // module.num_key_value_groups
    )
    sink(RecomputeAttentionBasis(
        scaled_masked_logits=logits,
        attention_probabilities=probabilities,
        physical_values=value,
        query_to_kv_head=query_to_kv_head,
        keep_mask=keep_mask,
    ))
    output = torch.matmul(probabilities, repeated_value)
    return output.transpose(1, 2).contiguous(), probabilities


def update_recomputed_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    key_states_from_hs: torch.Tensor,
    value_states_from_hs: torch.Tensor,
    token_idx: list[int],
    update_indices: list[int] | None,
    fuse_indices: list[int] | None,
    fuse_theta: float | None,
) -> None:
    if (fuse_indices is None) != (fuse_theta is None):
        raise ValueError("fuse_indices and fuse_theta must be provided together.")

    write_indices = token_idx if update_indices is None else update_indices
    source_index = {physical: local for local, physical in enumerate(token_idx)}
    if not set(write_indices).issubset(source_index):
        raise ValueError("update_indices must be a subset of token_idx.")
    fused = set() if fuse_indices is None else set(fuse_indices)
    if not fused.issubset(write_indices):
        raise ValueError("fuse_indices must be a subset of the updated indices.")

    overwrite_indices = [index for index in write_indices if index not in fused]
    if overwrite_indices:
        local_indices = [source_index[index] for index in overwrite_indices]
        key_states[:, :, overwrite_indices, :] = key_states_from_hs[:, :, local_indices, :]
        value_states[:, :, overwrite_indices, :] = value_states_from_hs[:, :, local_indices, :]

    if fuse_indices:
        local_indices = [source_index[index] for index in fuse_indices]
        key_states[:, :, fuse_indices, :] = (
            (1 - fuse_theta) * key_states[:, :, fuse_indices, :]
            + fuse_theta * key_states_from_hs[:, :, local_indices, :]
        )
        value_states[:, :, fuse_indices, :] = (
            (1 - fuse_theta) * value_states[:, :, fuse_indices, :]
            + fuse_theta * value_states_from_hs[:, :, local_indices, :]
        )


def adapt_recompute_mask(
    mask: torch.Tensor,
    *,
    attn_implementation: str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert the canonical boolean keep-mask for the selected backend."""
    if mask.dtype != torch.bool:
        raise TypeError("Recompute mask must be a boolean keep-mask.")

    if attn_implementation == "sdpa":
        return mask

    if attn_implementation == "eager":
        if not dtype.is_floating_point:
            raise TypeError("Eager attention requires a floating-point query dtype.")
        additive_mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        return additive_mask.masked_fill(~mask, torch.finfo(dtype).min)

    raise NotImplementedError(
        "KV recomputation supports only 'eager' and 'sdpa' attention, "
        f"not {attn_implementation!r}."
    )


def create_recompute_mask(
    query_len: int, 
    key_len: int,
    token_idx: list[int] | torch.Tensor,
    device: torch.device,
    to_4d: bool = False, 
    sliding_window: int | None = None,
) -> torch.Tensor:
    """
    Creates a boolean mask for attention recomputation.
    True = Attend (Keep), False = Mask out.
    """
    # 1. Ensure token_idx is a tensor (setup cost)
    if isinstance(token_idx, list):
        token_idx = torch.tensor(token_idx, device=device, dtype=torch.long)
    
    assert token_idx.dim() == 1 and token_idx.size(0) == query_len, \
        "token_idx must be a 1D tensor of length query_len"

    # 2. Create column indices [0, 1, 2, ..., key_len-1] (The Key positions)
    # Shape: [1, key_len]
    col_indices = torch.arange(key_len, device=device).unsqueeze(0)
    
    # 3. Create row thresholds (The Query positions)
    # Shape: [query_len, 1]
    row_limits = token_idx.unsqueeze(1)
    
    # 4. Broadcast comparison: 
    # Condition A (Causal): Key index (col) <= Query index (row)
    # Shape: [query_len, key_len]
    causal_mask = col_indices <= row_limits

    # 5. Apply Sliding Window Constraint (if provided)
    if sliding_window is not None:
        # Condition B (Window): Key index (col) > Query index (row) - window_size
        # Example: Query at 10000, Window 4096. 
        # Allowed keys: 10000 - 4096 < k <= 10000.
        # k > 5904
        lower_bound = row_limits - sliding_window
        window_mask = col_indices > lower_bound
        
        # Combine with Logical AND
        attn_mask = causal_mask & window_mask
    else:
        attn_mask = causal_mask

    # 6. Reshape for 4D
    if to_4d:
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        
    return attn_mask


def prepare_pos_embed_and_mask(
    model: SupportedModel,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    recompute_indices: list[int],
) -> tuple[tuple[torch.Tensor, torch.Tensor] | torch.Tensor, torch.Tensor]:
    pos_embed = model.model.rotary_emb(
        hidden_states,
        pos_ids,
    )
    assert isinstance(pos_embed, tuple|torch.Tensor)
    if isinstance(pos_embed, tuple):
        cos, sin = pos_embed
        cos = cos[:, recompute_indices].to(hidden_states.device)
        sin = sin[:, recompute_indices].to(hidden_states.device)
        pos_embed = (cos, sin)
    else:
        pos_embed = pos_embed[:, recompute_indices].to(hidden_states.device)

    recompute_mask = create_recompute_mask(
        query_len=len(recompute_indices),
        key_len=pos_ids.size(1),
        token_idx=recompute_indices,
        to_4d=True,
        device=model.device,
        sliding_window=getattr(model.config, "sliding_window", None),
    )
    return pos_embed, recompute_mask
