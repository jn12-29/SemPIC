import logging
import torch
from typing import Callable
from transformers import GenerationConfig
from ...model import SupportedModel
from ..abc import PrefillResult, TokenizerType
from ..utils.flops import AutoFlopsCalculator
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ...cache.rotate import rerotate_kv_p, rerotate_kv_flops
from ...cache import KVCache, concate_kv_caches, KVDim
from ...prompt import TokenizedPrompt
from ..abc import PreparedKVMapping
from ._prompt_utils import (
    finish_recomputed_prefill,
    prepare_recompute_inputs,
    recompute_prefix_indices,
)

LOGGER = logging.getLogger(__name__)


def cache_blend_eval(
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
            "cache_blend_eval requires a float 'recompute_ratio' kwarg greater than 0.0 "
            "and at most 1.0; use no_recompute when no tokens should be recomputed"
        )
    
    if kwargs is not None and kwargs != {}:
        LOGGER.warning("cache_blend_eval got unexpected kwargs: %s", kwargs)

    inputs = prepare_recompute_inputs(
        "cache_blend", model, prompt, prepared_kvs
    )
    candidate_len = len(inputs.candidate_indices)
    num_recompute = int(recompute_ratio * candidate_len)
    if num_recompute == 0:
        raise ValueError(
            "cache_blend_eval selects no ContextBlock tokens; use no_recompute instead"
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
    num_flops += 2 * flops_calculator.decoder_layer_flops(
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
    recompute_result = recompute_kv(
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
    )

    recomputed_hs = recompute_result["recomputed_hidden_states"]

    ## first layer recompute, the KV cache are different
    recompute_result = recompute_kv(
        model=model,
        kv_cache=full_kv,
        hidden_states=recomputed_hs,
        pos_ids=pos_ids,
        token_idx=recompute_indices,
        layer_idx=1,
        update_cache=False,
        token_position_ids=token_position_ids,
        pos_embed=pos_embed,
        recompute_mask=recompute_mask,
    )

    recomputed_hs = recompute_result["recomputed_hidden_states"]
    kv_from_hs = recompute_result["kv_from_hs"]

    new_key, new_value = kv_from_hs['key'], kv_from_hs['value']
    active_offsets = {index: offset for offset, index in enumerate(active_indices)}
    candidate_offsets = [active_offsets[index] for index in candidate_indices]
    old_key = full_kv[1]["key"][:, :, candidate_indices, :]
    old_value = full_kv[1]["value"][:, :, candidate_indices, :]
    new_key_to_compare = new_key[:, :, candidate_offsets, :]
    new_value_to_compare = new_value[:, :, candidate_offsets, :]

    concat_old_kv = torch.cat([old_key, old_value], dim=3) # [batch, head, seq_len, key_dim+value_dim]
    concat_new_kv = torch.cat([new_key_to_compare, new_value_to_compare], dim=3)

    all_dims = list(KVDim.values())
    all_dims.remove(KVDim['seq'])

    num_flops += 3 * concat_old_kv.numel() - candidate_len
    diff_along_seq = torch.sum(
        (concat_old_kv - concat_new_kv) ** 2,
        dim=all_dims
    )  # [seq_len,]

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

    full_kv[1]["key"][:, :, recompute_indices, :] = new_key
    full_kv[1]["value"][:, :, recompute_indices, :] = new_value
    recompute_indices = recomputed_prefix_indices + query_indices
    token_position_ids = pos_ids[:, recompute_indices]

    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=recomputed_hs,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )

    assert model.config.num_hidden_layers is not None
    for layer in range(2, model.config.num_hidden_layers):
        recompute_len = len(recompute_indices)
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1,
            seq_len=recompute_len,
            cache_len=full_kv[0].key.size(2) - recompute_len,
        )
        recompute_result = recompute_kv(
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
        )
        recomputed_hs = recompute_result["recomputed_hidden_states"]

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
