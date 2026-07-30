"""Streaming layer bases for gold-answer query passes."""

from dataclasses import dataclass
from typing import Any, Callable, TypeAlias

import torch

from ..cache import KVCache, concate_kv_caches
from ..cache_comb.abc import PrefillResult, TokenizerType
from ..cache_comb.recompute_kv import prepare_pos_embed_and_mask, recompute_kv
from ..model import SupportedModel
from .basis import LayerAttentionBasis, PhysicalLayout


LayerAttentionBasisSink: TypeAlias = Callable[[LayerAttentionBasis], None]


@dataclass(frozen=True, slots=True)
class AnswerAttentionMeta:
    prefix_physical_len: int
    answer_len: int
    num_layers: int
    num_query_heads: int


def tokenize_gold_answer(tokenizer: TokenizerType, answer: str) -> torch.LongTensor:
    """Tokenize the gold answer alone, without special tokens, padding, or EOS."""
    encoded: Any = tokenizer(
        answer,
        add_special_tokens=False,
        padding=False,
    )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        answer_ids = encoded.to(dtype=torch.long)
    else:
        answer_ids = torch.tensor(encoded, dtype=torch.long)
    if answer_ids.ndim != 1 or answer_ids.numel() == 0:
        raise ValueError("Gold answer must tokenize to a non-empty one-dimensional sequence.")
    return answer_ids


def stream_answer_attention(
    *,
    model: SupportedModel,
    prefill: PrefillResult,
    answer_ids: torch.LongTensor,
    basis_sink: LayerAttentionBasisSink,
    layout: PhysicalLayout,
) -> AnswerAttentionMeta:
    """Synchronously stream one layer of literal-Y attention probabilities at a time.

    The basis sink must consume each layer synchronously and not retain GPU tensors.
    """
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("Answer attention requires eager attention.")
    if answer_ids.ndim != 1 or answer_ids.numel() == 0:
        raise ValueError("answer_ids must be a non-empty one-dimensional tensor.")
    if prefill.position_ids.ndim != 2 or prefill.position_ids.size(0) != 1:
        raise ValueError("Prefill position_ids must have shape [1, sequence].")

    answer_ids_2d = answer_ids.to(device=model.device, dtype=torch.long).unsqueeze(0)
    answer_len = answer_ids_2d.size(1)
    prefix_physical_len = prefill.past_key_values.get_seq_length()
    if prefix_physical_len <= 0:
        raise ValueError("Answer attention requires a non-empty prompt cache.")

    prefix_positions = torch.arange(
        prefix_physical_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prefix_cache = KVCache.from_hf_cache(
        prefill.past_key_values,
        position_ids=prefix_positions,
    )
    first_kv = prefix_cache[0]
    last_prompt_position = int(prefill.position_ids[0, -1].item())
    answer_positions = torch.arange(
        last_prompt_position + 1,
        last_prompt_position + 1 + answer_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    answer_slots = KVCache.create_dummy(
        num_layers=len(prefix_cache.layers),
        batch_size=1,
        num_heads=first_kv.key.size(1),
        seq_len=answer_len,
        key_head_dim=first_kv.key.size(3),
        value_head_dim=first_kv.value.size(3),
        position_ids=answer_positions,
        dtype=first_kv.key.dtype,
        device=model.device,
    )
    full_cache = concate_kv_caches([prefix_cache, answer_slots])
    answer_indices = list(range(
        prefix_physical_len,
        prefix_physical_len + answer_len,
    ))
    position_ids = torch.cat((prefix_positions, answer_positions), dim=1)
    hidden_states = model.model.embed_tokens(answer_ids_2d)
    pos_embed, attention_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=position_ids,
        recompute_indices=answer_indices,
    )

    num_query_heads: int | None = None
    pass_layout = PhysicalLayout(
        physical_length=prefix_physical_len + answer_len,
        chunks=layout.chunks,
    )
    for layer_index in range(model.config.num_hidden_layers):
        result = recompute_kv(
            model=model,
            kv_cache=full_cache,
            hidden_states=hidden_states,
            pos_ids=position_ids,
            token_idx=answer_indices,
            layer_idx=layer_index,
            update_cache=True,
            token_position_ids=answer_positions,
            pos_embed=pos_embed,
            recompute_mask=attention_mask,
            return_attention_probs=True,
            attention_basis_sink=lambda value, layer_index=layer_index: basis_sink(
                LayerAttentionBasis.from_recompute(
                    layer_index=layer_index,
                    value=value,
                    layout=pass_layout,
                )
            ),
        )
        hidden_states = result["recomputed_hidden_states"]
        probabilities = result.get("attention_probs")
        if probabilities is None:
            raise ValueError(f"Layer {layer_index} returned no attention probabilities.")
        expected_shape = (
            1,
            probabilities.size(1),
            answer_len,
            prefix_physical_len + answer_len,
        )
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_index} attention shape mismatch: "
                f"expected {expected_shape}, got {tuple(probabilities.shape)}."
            )
        if num_query_heads is None:
            num_query_heads = probabilities.size(1)
        elif probabilities.size(1) != num_query_heads:
            raise ValueError("Attention query-head count changed across layers.")

    assert num_query_heads is not None
    return AnswerAttentionMeta(
        prefix_physical_len=prefix_physical_len,
        answer_len=answer_len,
        num_layers=model.config.num_hidden_layers,
        num_query_heads=num_query_heads,
    )


def stream_shifted_prediction_attention(
    *,
    model: SupportedModel,
    prefill: PrefillResult,
    answer_ids: torch.LongTensor,
    basis_sink: LayerAttentionBasisSink,
    layout: PhysicalLayout,
) -> AnswerAttentionMeta:
    """Stream causal query rows aligned with the gold tokens they predict.

    For targets ``[y0, ..., yT-1]``, the query inputs are the final prompt token
    followed by ``[y0, ..., yT-2]``. The basis sink receives one complete
    layer event with all aligned prediction-query rows.
    """
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("Shifted prediction attention requires eager attention.")
    if answer_ids.ndim != 1 or answer_ids.numel() == 0:
        raise ValueError("answer_ids must be a non-empty one-dimensional tensor.")
    if prefill.position_ids.ndim != 2 or prefill.position_ids.size(0) != 1:
        raise ValueError("Prefill position_ids must have shape [1, sequence].")
    if (
        prefill.generation_input_ids.ndim != 2
        or prefill.generation_input_ids.size(0) != 1
        or prefill.generation_input_ids.size(1) == 0
    ):
        raise ValueError("Prefill generation_input_ids must have shape [1, sequence].")

    prefix_physical_len = prefill.past_key_values.get_seq_length()
    if prefix_physical_len <= 0:
        raise ValueError("Shifted prediction attention requires a non-empty prompt cache.")
    if prefill.generation_input_ids.size(1) != prefix_physical_len:
        raise ValueError(
            "Prefill generation_input_ids must align with the physical prompt cache."
        )

    answer_ids = answer_ids.to(device=model.device, dtype=torch.long)
    answer_len = answer_ids.numel()
    prompt_last_id = prefill.generation_input_ids[0, -1].to(
        device=model.device, dtype=torch.long
    )
    query_ids = torch.cat((prompt_last_id.view(1), answer_ids[:-1])).unsqueeze(0)

    prefix_positions = torch.arange(
        prefix_physical_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    last_prompt_position = int(prefill.position_ids[0, -1].item())
    prefix_positions[0, -1] = last_prompt_position
    prefix_cache = KVCache.from_hf_cache(
        prefill.past_key_values,
        position_ids=prefix_positions,
    )
    first_kv = prefix_cache[0]
    answer_slot_count = answer_len - 1
    answer_positions = torch.arange(
        last_prompt_position + 1,
        last_prompt_position + 1 + answer_slot_count,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    if answer_slot_count:
        answer_slots = KVCache.create_dummy(
            num_layers=len(prefix_cache.layers),
            batch_size=1,
            num_heads=first_kv.key.size(1),
            seq_len=answer_slot_count,
            key_head_dim=first_kv.key.size(3),
            value_head_dim=first_kv.value.size(3),
            position_ids=answer_positions,
            dtype=first_kv.key.dtype,
            device=model.device,
        )
        full_cache = concate_kv_caches([prefix_cache, answer_slots])
    else:
        full_cache = prefix_cache

    query_indices = list(range(
        prefix_physical_len - 1,
        prefix_physical_len + answer_slot_count,
    ))
    update_indices = query_indices[1:]
    query_positions = torch.arange(
        last_prompt_position,
        last_prompt_position + answer_len,
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    position_ids = torch.cat((prefix_positions, answer_positions), dim=1)
    hidden_states = model.model.embed_tokens(query_ids)
    pos_embed, attention_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=position_ids,
        recompute_indices=query_indices,
    )

    num_query_heads: int | None = None
    pass_layout = PhysicalLayout(
        physical_length=prefix_physical_len + answer_slot_count,
        chunks=layout.chunks,
    )
    for layer_index in range(model.config.num_hidden_layers):
        result = recompute_kv(
            model=model,
            kv_cache=full_cache,
            hidden_states=hidden_states,
            pos_ids=position_ids,
            token_idx=query_indices,
            layer_idx=layer_index,
            update_cache=True,
            token_position_ids=query_positions,
            pos_embed=pos_embed,
            recompute_mask=attention_mask,
            update_indices=update_indices,
            return_attention_probs=True,
            attention_basis_sink=lambda value, layer_index=layer_index: basis_sink(
                LayerAttentionBasis.from_recompute(
                    layer_index=layer_index,
                    value=value,
                    layout=pass_layout,
                )
            ),
        )
        hidden_states = result["recomputed_hidden_states"]
        probabilities = result.get("attention_probs")
        if probabilities is None:
            raise ValueError(f"Layer {layer_index} returned no attention probabilities.")
        expected_shape = (
            1,
            probabilities.size(1),
            answer_len,
            prefix_physical_len + answer_slot_count,
        )
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError(
                f"Layer {layer_index} attention shape mismatch: "
                f"expected {expected_shape}, got {tuple(probabilities.shape)}."
            )
        if num_query_heads is None:
            num_query_heads = probabilities.size(1)
        elif probabilities.size(1) != num_query_heads:
            raise ValueError("Attention query-head count changed across layers.")

    assert num_query_heads is not None
    return AnswerAttentionMeta(
        prefix_physical_len=prefix_physical_len,
        answer_len=answer_len,
        num_layers=model.config.num_hidden_layers,
        num_query_heads=num_query_heads,
    )


__all__ = [
    "AnswerAttentionMeta",
    "LayerAttentionBasisSink",
    "stream_answer_attention",
    "stream_shifted_prediction_attention",
    "tokenize_gold_answer",
]
