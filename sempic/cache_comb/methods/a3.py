import logging
import torch
from typing import Callable
from transformers import GenerationConfig
from ...model import SupportedModel
from ..abc import PrefillResult, TokenizerType
from ..utils.flops import AutoFlopsCalculator
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ...cache.rotate import rerotate_kv_p, rerotate_kv_flops
from ...cache import KVCache, concate_kv_caches
from ...prompt import TokenizedPrompt
from ..abc import PreparedKVMapping
from ._prompt_utils import (
    finish_recomputed_prefill,
    prepare_recompute_inputs,
    recompute_prefix_indices,
)

LOGGER = logging.getLogger(__name__)


def a3_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    kwargs: dict|None = None
) -> PrefillResult:
    if kwargs is None:
        recompute_ratio = None
    else:
        kwargs = kwargs.copy()
        recompute_ratio = kwargs.pop("recompute_ratio", None)

    if not isinstance(recompute_ratio, float) or not (0.0 < recompute_ratio <= 1.0):
        raise ValueError(
            "a3_eval requires a float 'recompute_ratio' kwarg greater than 0.0 "
            "and at most 1.0; use no_recompute when no tokens should be recomputed"
        )
    
    if kwargs is not None and kwargs != {}:
        LOGGER.warning("a3_eval got unexpected kwargs: %s", kwargs)
    inputs = prepare_recompute_inputs(
        "a3", model, prompt, prepared_kvs
    )
    candidate_len = len(inputs.candidate_indices)
    num_recompute = int(recompute_ratio * candidate_len)
    if num_recompute == 0:
        raise ValueError(
            "a3_eval selects no ContextBlock tokens; use no_recompute instead"
        )
    full_kv = inputs.prefix_cache
    total_data_len = full_kv[0].key.size(2)
    old_pos = full_kv[0].position_ids.to(model.device)
    new_pos = torch.arange(total_data_len, device=model.device).unsqueeze(0)

    q_ids = inputs.input_ids[:, total_data_len:]
    query_len = q_ids.size(1)

    num_head, key_head_dim, value_head_dim = full_kv[0].key.size(1), full_kv[0].key.size(3), full_kv[0].value.size(3)
    dummy_query_cache = KVCache.create_dummy(
        batch_size=1,
        num_layers=len(full_kv.layers),
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=full_kv[0].key.dtype,
    )

    num_flops = 0
    flops_calculator = AutoFlopsCalculator(model)

    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)

    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)
    full_kv = concate_kv_caches([full_kv, dummy_query_cache])

    active_indices = sorted(inputs.candidate_indices + inputs.inline_indices)
    candidate_indices = list(inputs.candidate_indices)
    query_indices = list(range(total_data_len, total_data_len + query_len))
    recompute_indices = active_indices + query_indices
    hidden_states = model.model.embed_tokens(inputs.input_ids[:, recompute_indices])
    pos_ids = torch.arange(0, total_data_len + query_len, dtype=torch.long, device=model.device).unsqueeze(0)  # [1, total_data_len]

    first_pass_len = len(recompute_indices)
    num_flops += flops_calculator.decoder_layer_flops(
        batch_size=1,
        seq_len=first_pass_len,
        cache_len=full_kv[0].key.size(2) - first_pass_len,
    )
    token_position_ids = pos_ids[:, recompute_indices]
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )
    ## zero-th layer recompute, the KV cache are the same
    recomputed_result = recompute_kv(
        model=model,
        kv_cache=full_kv,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        token_idx=recompute_indices,
        layer_idx=0,
        update_cache=True,
        token_position_ids=token_position_ids,
        pos_embed=pos_embed,
        recompute_mask=recompute_mask,
        return_query_states=True,
    )
    recomputed_hs = recomputed_result["recomputed_hidden_states"]
    query_states = recomputed_result["query_states"] # [1, num_heads, seq_len + query_len, head_dim]
    key_states = recomputed_result["kv_from_hs"]["key"] # [1, num_key_heads, seq_len + query_len, head_dim]

    assert isinstance(query_states, torch.Tensor)

    q_view = query_states[:, :, -query_len:, :]  # [1, num_heads, query_len, head_dim]
    active_offsets = {index: offset for offset, index in enumerate(active_indices)}
    candidate_offsets = [active_offsets[index] for index in candidate_indices]
    k_target = key_states[:, :, candidate_offsets, :]
    num_heads = q_view.size(1)
    num_key_heads = k_target.size(1)

    if num_heads != num_key_heads:
        num_rep = num_heads // num_key_heads
        # Expand: [1, n_kv, 1, ...] -> [1, n_kv, n_rep, ...] -> [1, n_heads, ...]
        k_target = k_target[:, :, None, :, :].expand(
            1, num_key_heads, num_rep, candidate_len, k_target.size(-1)
        )
        k_target = k_target.reshape(1, num_heads, candidate_len, k_target.size(-1))

    # Calculate token importance scores
    with torch.no_grad():
        attn_logits = torch.matmul(q_view, k_target.transpose(-1, -2))
        head_dim = q_view.size(-1)
        attn_logits = attn_logits / (head_dim ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_logits, dim=-1)
        diff_along_seq = attn_weights.sum(dim=(0,1,2))  # [seq_len]
    batch_size, num_query_heads, query_rows, head_dim = q_view.shape
    score_rows = batch_size * num_query_heads * query_rows
    num_flops += (
        2 * score_rows * candidate_len * head_dim
        + score_rows * candidate_len
        + score_rows * (4 * candidate_len - 1)
        + candidate_len * (score_rows - 1)
    )

    # topk_indices_t: range from 0 to seq_len-1
    _, topk_indices_t = torch.topk(diff_along_seq, k=num_recompute, largest=True, sorted=False)
    topk_indices_t = torch.sort(topk_indices_t).values  # sort indices

    selected_context_indices = [
        candidate_indices[index] for index in topk_indices_t.cpu().tolist()
    ]
    recomputed_prefix_indices = recompute_prefix_indices(inputs, selected_context_indices)
    hidden_offsets = {index: offset for offset, index in enumerate(recompute_indices)}
    hs_indices = [hidden_offsets[index] for index in recomputed_prefix_indices + query_indices]
    recomputed_hs = recomputed_hs[:, hs_indices, :]

    recompute_indices = recomputed_prefix_indices + query_indices
    token_position_ids = pos_ids[:, recompute_indices]

    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=recomputed_hs,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )

    assert model.config.num_hidden_layers is not None
    for layer in range(1, model.config.num_hidden_layers):
        recompute_len = len(recompute_indices)
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1,
            seq_len=recompute_len,
            cache_len=full_kv[0].key.size(2) - recompute_len,
        )
        recomputed_hs = recompute_kv(
            model=model,
            kv_cache=full_kv,
            hidden_states=recomputed_hs,
            pos_ids=pos_ids,
            token_idx=recompute_indices,
            layer_idx=layer,
            update_cache=True,
            token_position_ids=token_position_ids,
            pos_embed=pos_embed,
            recompute_mask=recompute_mask,
        )["recomputed_hidden_states"]

    return finish_recomputed_prefill(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        cache=full_kv,
        terminal_hidden_states=recomputed_hs[:, -query_len:, :],
        terminal_ids=q_ids,
        prefix_physical_len=total_data_len,
        flops=num_flops,
    )
