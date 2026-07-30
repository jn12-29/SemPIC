import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    eager_attention_forward as qwen3_eager_attention_forward,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)
from ...cache import KVCache, KeyValue
from .utils import (
    AttentionBasisSink,
    adapt_recompute_mask,
    create_recompute_mask,
    eager_attention_with_basis,
    RecomputeResult,
    update_recomputed_kv,
)

__all__ = [
    "qwen3_recompute_kv",
]

def qwen3_recompute_kv(
    model: Qwen3ForCausalLM|Qwen3Model,
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
    if isinstance(model, Qwen3ForCausalLM):
        model = model.model  # Extract the base model if a full model is provided

    assert isinstance(model.layers, torch.nn.ModuleList), "Model does not have layers attribute"
    assert 0 <= layer_idx < len(model.layers), "Invalid layer index"

    assert hidden_states.dim() == 3, "Hidden states must be a 3D tensor"
    assert hidden_states.size(0) == 1, "Batch size must be 1 for recomputation"
    assert hidden_states.size(1) == len(token_idx), "Hidden states sequence length must match token_idx length"
    assert isinstance(pos_embed, tuple|None), "pos_embed must be a tuple of (cos, sin) or None"
    decoder_layer = model.layers[layer_idx]
    assert isinstance(decoder_layer, Qwen3DecoderLayer), "Layer is not a Qwen3DecoderLayer"

    if pos_embed is None:
        pos_embed = model.rotary_emb(
            hidden_states,
            pos_ids,
        )
        assert isinstance(pos_embed, tuple)
        cos, sin = pos_embed

        cos = cos[:, token_idx].to(hidden_states.device)
        sin = sin[:, token_idx].to(hidden_states.device)
    else:
        cos, sin = pos_embed

    residual = hidden_states
    hidden_states = decoder_layer.input_layernorm(hidden_states)

    attn: Qwen3Attention = decoder_layer.self_attn

    ## Attention
    input_shape = hidden_states.size()[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    # [1, num_heads, len(token_idx), head_dim]
    # Qwen3 Difference: Uses q_norm and k_norm after projection and view, before transpose
    query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states_from_hs = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states_from_hs = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states, key_states_from_hs = qwen3_apply_rotary_pos_emb(
        query_states, key_states_from_hs, cos, sin
    )

    if token_position_ids is None:
        token_position_ids = pos_ids[:, token_idx]
    kv_from_hs = KeyValue(key=key_states_from_hs, value=value_states_from_hs, position_ids=token_position_ids)
    key_states = kv_cache[layer_idx]["key"] # [1, num_heads, seq_len, head_dim]
    value_states = kv_cache[layer_idx]["value"]

    if not update_cache:
        key_states = key_states.clone()
        value_states = value_states.clone()
    
    update_recomputed_kv(
        key_states,
        value_states,
        key_states_from_hs,
        value_states_from_hs,
        token_idx,
        update_indices,
        fuse_indices,
        fuse_theta,
    )
    

    if recompute_mask is None:
        recompute_mask = create_recompute_mask(
            query_len=query_states.size(2),
            key_len=key_states.size(2),
            token_idx=token_idx,
            to_4d=True,
            device=hidden_states.device,
        )

    attn_implementation = attn.config._attn_implementation
    if attention_basis_sink is not None:
        if attn_implementation != "eager":
            raise ValueError("Attention basis extraction requires eager attention.")
        hidden_states, attention_probs = eager_attention_with_basis(
            module=attn,
            query=query_states,
            key=key_states,
            value=value_states,
            keep_mask=recompute_mask,
            scaling=attn.scaling,
            sink=attention_basis_sink,
        )
    else:
        recompute_mask = adapt_recompute_mask(
            recompute_mask,
            attn_implementation=attn_implementation,
            dtype=query_states.dtype,
        )
        if attn_implementation == "eager":
            attention_interface = qwen3_eager_attention_forward
        else:
            if return_attention_probs:
                raise ValueError("Attention probabilities require eager attention.")
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
        hidden_states, attention_probs = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask=recompute_mask,
            dropout=0.0,
            scaling=attn.scaling,
        )

    hidden_states = hidden_states.reshape(*input_shape, -1).contiguous()
    hidden_states = attn.o_proj(hidden_states)

    hidden_states = residual + hidden_states # [1, len(token_idx), hidden_size]

    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    result_dict = RecomputeResult(
        recomputed_hidden_states=hidden_states,
        kv_from_hs=kv_from_hs,
        query_states=query_states if return_query_states else None,
    )
    if return_attention_probs:
        if attention_probs is None:
            raise ValueError("Eager attention returned no attention probabilities.")
        result_dict["attention_probs"] = attention_probs
    return result_dict
