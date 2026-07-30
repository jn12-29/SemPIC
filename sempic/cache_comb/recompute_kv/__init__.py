""" Utils for recomputing KV cache """
import torch
from transformers import (
    LlamaForCausalLM,
    Qwen3ForCausalLM,
)
from .utils import AttentionBasisSink, prepare_pos_embed_and_mask, RecomputeResult
from ...model import SupportedModel
from ...cache import KVCache

from .llama import llama_recompute_kv
from .qwen3 import qwen3_recompute_kv

__all__ = [
    "recompute_kv",
    "prepare_pos_embed_and_mask",
]

def recompute_kv(
    model: SupportedModel,
    kv_cache: KVCache,
    hidden_states: torch.Tensor,
    pos_ids: torch.Tensor,
    token_idx: list[int],
    layer_idx: int,
    update_cache: bool=False,
    token_position_ids: torch.Tensor|None=None,
    pos_embed: tuple[torch.Tensor, torch.Tensor]|torch.Tensor|None=None,
    recompute_mask: torch.Tensor|None=None,
    update_indices: list[int]|None=None,
    fuse_indices: list[int]|None=None,
    fuse_theta: float|None=None,
    return_query_states: bool=False,
    return_attention_probs: bool=False,
    attention_basis_sink: AttentionBasisSink | None = None,
) -> RecomputeResult:
    """
    Recompute key and value states for a specific layer and token indices.
    
    Args:
        model (SupportedModel): The model or its base model.
        kv_cache (KVCache): The KVCache instance containing cached key-value pairs. The keys and values should 
            have shape [1, num_heads, seq_len, head_dim].
        hidden_states (torch.Tensor): The hidden states tensor of shape [1, len(token_idx), hidden_size].
            The positions of hidden states should be within the kv_cache.
        pos_ids (torch.Tensor): The position IDs tensor of shape [1, seq_len] corresponding to kv_cache.
        token_idx (list of int): List of token indices to recompute. The length should match hidden_states' seq_len.
            All token indices must be within the range of kv_cache sequence length.
        layer_idx (int): The layer index to recompute.
        update_cache (bool): Whether to update the KVCache with the recomputed states. Default is True.
        token_position_ids (torch.Tensor|None): Shape [1, len(token_idx)]
            positional_ids to store in the KV cache. If not given, pos_ids[:,token_idx] is used.
        pos_embed: pos_embedding to avoid repeat computation.
        recompute_mask: recompute mask for attention. Used to avoid repeat computation.
        update_indices: which indices of KV cache to update, must be a subset of token_idx. If update_cache is True
            and update_indices is not given, all indices in token_idx will be updated.
        fuse_indices: updated cache indices whose old and new KVs should be mixed.
        fuse_theta: the value to mix old and new KVs at fuse_indices. Both fusion arguments must be provided together.
        return_query_states (bool): Whether to return the recomputed query states. Default is False.
        return_attention_probs (bool): Whether to return eager post-softmax attention probabilities.
    Returns:
        - result_dict (dict): A dictionary containing the recomputed states and optional query states.
    """
    if isinstance(model, LlamaForCausalLM):
        recompute_func = llama_recompute_kv
    elif isinstance(model, Qwen3ForCausalLM):
        recompute_func = qwen3_recompute_kv
    else:
        raise ValueError(f"Unsupported model type for recompute_kv: {type(model)}")

    with torch.no_grad():
        result_dict = recompute_func(
            model=model, # type: ignore
            kv_cache=kv_cache,
            hidden_states=hidden_states,
            pos_ids=pos_ids,
            token_idx=token_idx,
            layer_idx=layer_idx,
            update_cache=update_cache,
            token_position_ids=token_position_ids,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
            update_indices=update_indices,
            fuse_indices=fuse_indices,
            fuse_theta=fuse_theta,
            return_query_states=return_query_states,
            return_attention_probs=return_attention_probs,
            attention_basis_sink=attention_basis_sink,
        )
    return result_dict
