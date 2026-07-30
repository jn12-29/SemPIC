import logging
import torch
import random
from typing import Callable
from transformers import GenerationConfig
from ..abc import PrefillResult, TokenizerType
from ..recompute_kv import recompute_kv, prepare_pos_embed_and_mask
from ..utils.flops import AutoFlopsCalculator
from ...model import SupportedModel
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


def rand_recompute_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig|None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]]|None = None,
    kwargs: dict|None = None
) -> PrefillResult:
    assert model.config.num_hidden_layers is not None
    if kwargs is None:
        recompute_ratio = None
        seed = None
    else:
        kwargs = kwargs.copy()
        recompute_ratio = kwargs.pop("recompute_ratio", None)
        seed = kwargs.pop("seed", None)
    if not isinstance(recompute_ratio, float) or recompute_ratio < 0:
        raise ValueError("rand_recompute_eval requires a non-negative 'recompute_ratio' kwarg")

    if kwargs is not None and kwargs != {}:
        LOGGER.warning("rand_recompute_eval got unexpected kwargs: %s", kwargs)

    inputs = prepare_recompute_inputs(
        "rand_recompute", model, prompt, prepared_kvs
    )
    full_kv = inputs.prefix_cache
    total_data_len = full_kv[0].key.size(2)
    old_pos = full_kv[0].position_ids.to(model.device)
    new_pos = torch.arange(total_data_len, device=model.device).unsqueeze(0)

    q_ids = inputs.input_ids[:, total_data_len:]
    query_len = q_ids.size(1)

    num_head, key_head_dim, value_head_dim = full_kv[0].key.size(1), full_kv[0].key.size(3), full_kv[0].value.size(3)
    dummy_query_cache = KVCache.create_dummy(
        num_layers=len(full_kv.layers),
        batch_size=1,
        num_heads=num_head,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        seq_len=query_len,
        device=model.device,
        dtype=full_kv[0].key.dtype,
    )
    flops_calculator = AutoFlopsCalculator(model)
    num_flops = 0

    nope_dim = getattr(model.config, "qk_nope_head_dim", None)
    assert isinstance(nope_dim, int|None)
    full_kv = rerotate_kv_p(full_kv, model.model.rotary_emb, old_pos, new_pos, nope_dim=nope_dim)
    num_flops += rerotate_kv_flops(full_kv, nope_dim=nope_dim)
    full_kv = concate_kv_caches([full_kv, dummy_query_cache])

    random.seed(seed)
    candidate_indices = list(inputs.candidate_indices)
    recompute_len = int(len(candidate_indices) * recompute_ratio)
    selected_context_indices = random.sample(candidate_indices, recompute_len)
    recomputed_prefix_indices = recompute_prefix_indices(inputs, selected_context_indices)

    query_indices = list(range(total_data_len, total_data_len + query_len))
    recompute_indices = recomputed_prefix_indices + query_indices
    hidden_states = model.model.embed_tokens(inputs.input_ids[:, recompute_indices])
    pos_ids = torch.arange(
        0, total_data_len + query_len,
        dtype=torch.long, device=model.device
    ).unsqueeze(0)
    token_position_ids = pos_ids[:, recompute_indices]

    recomputed_hs = hidden_states
    pos_embed, recompute_mask = prepare_pos_embed_and_mask(
        model=model,
        hidden_states=hidden_states,
        pos_ids=pos_ids,
        recompute_indices=recompute_indices,
    )
    for layer in range(model.config.num_hidden_layers):
        recomputed_hs  = recompute_kv(
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
        recompute_len = len(recompute_indices)
        num_flops += flops_calculator.decoder_layer_flops(
            batch_size=1,
            seq_len=recompute_len,
            cache_len=full_kv[0].key.size(2) - recompute_len,
        )
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
