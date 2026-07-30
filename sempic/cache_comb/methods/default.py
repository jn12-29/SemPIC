import logging
from typing import Callable

import torch
from transformers import GenerationConfig

from ...model import SupportedModel
from ...prompt import TokenizedPrompt
from ..abc import PrefillResult, PreparedKVMapping, TokenizerType
from ..utils.flops import AutoFlopsCalculator
from ._prompt_utils import ids_2d, inline_only_ids, terminal_span


LOGGER = logging.getLogger(__name__)


def _ordinary_prompt_eval(
    *,
    model: SupportedModel,
    input_ids: torch.Tensor,
) -> PrefillResult:
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    if outputs.past_key_values is None:
        raise ValueError("Prompt prefill returned no KV cache.")
    return PrefillResult(
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        generation_input_ids=input_ids,
        position_ids=torch.tensor(
            [[input_ids.size(1) - 1]], dtype=torch.long, device=model.device
        ),
        attention_mask=attention_mask,
        flops=AutoFlopsCalculator(model).forward_flops(
            batch_size=input_ids.size(0),
            seq_len=input_ids.size(1),
            logits_rows=1,
        ),
    )


def no_cache_eval(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]] | None = None,
    kwargs: dict | None = None,
) -> PrefillResult:
    if kwargs:
        LOGGER.warning("no_cache_eval got unexpected kwargs: %s", kwargs)
    if prepared_kvs:
        raise ValueError("no_cache does not consume prepared KV artifacts.")
    return _ordinary_prompt_eval(model=model, input_ids=inline_only_ids(prompt))


def full_recompute(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]] | None = None,
    kwargs: dict | None = None,
) -> PrefillResult:
    if kwargs:
        LOGGER.warning("full_recompute got unexpected kwargs: %s", kwargs)
    if prepared_kvs:
        raise ValueError("full_recompute does not consume prepared KV artifacts.")
    return _ordinary_prompt_eval(model=model, input_ids=ids_2d(prompt))


def single_cache(
    model: SupportedModel,
    tokenizer: TokenizerType,
    generation_config: GenerationConfig | None,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    answer: str,
    answer_postprocess_func: Callable[[str, str], tuple[str, str]] | None = None,
    kwargs: dict | None = None,
) -> PrefillResult:
    if kwargs:
        LOGGER.warning("single_cache got unexpected kwargs: %s", kwargs)
    terminal = terminal_span(prompt)
    key = len(prompt.parts) - 1
    if set(prepared_kvs) != {key}:
        raise ValueError(f"single_cache requires exactly the whole-prefix KV artifact {key}.")
    cache = prepared_kvs[key]
    source_len = cache[0].position_ids.size(1)
    if source_len != terminal.start:
        raise ValueError(
            f"single_cache prefix position length mismatch: expected {terminal.start}, got {source_len}."
        )
    physical_len = cache[0].key.size(2)
    terminal_ids = ids_2d(prompt, terminal.start, terminal.end).to(model.device)
    terminal_positions = torch.arange(
        terminal.start,
        terminal.start + terminal_ids.size(1),
        device=model.device,
    ).unsqueeze(0)
    attention_mask = torch.ones(
        (1, physical_len + terminal_ids.size(1)), dtype=torch.long, device=model.device
    )
    with torch.no_grad():
        outputs = model(
            input_ids=terminal_ids,
            past_key_values=cache.to_hf_cache(config=model.config),
            position_ids=terminal_positions,
            cache_position=torch.arange(
                physical_len, physical_len + terminal_ids.size(1), device=model.device
            ),
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    if outputs.past_key_values is None:
        raise ValueError("single_cache terminal prefill returned no KV cache.")

    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_ids = torch.full(
        (1, physical_len), dummy_id, dtype=torch.long, device=model.device
    )
    generation_input = torch.cat([dummy_ids, terminal_ids], dim=1)
    return PrefillResult(
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        generation_input_ids=generation_input,
        position_ids=terminal_positions[:, -1:],
        attention_mask=attention_mask,
        flops=AutoFlopsCalculator(model).forward_flops(
            batch_size=1,
            seq_len=terminal_ids.size(1),
            cache_len=physical_len,
            logits_rows=1,
        ),
    )
