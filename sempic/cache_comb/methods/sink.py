import logging
from typing import Callable
from warnings import warn

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DynamicCache, GenerationConfig

from ...cache import get_kv_caches
from ...cache.hf_cache import concat_hf_caches, select_hf_cache
from ...model import SupportedModel
from ...prompt import TokenizedPrompt
from ..abc import PrefillResult, PreparedKVMapping, TokenizerType
from ..utils.flops import AutoFlopsCalculator
from ._prompt_utils import ids_2d, terminal_span
from .default import _ordinary_prompt_eval


LOGGER = logging.getLogger(__name__)


def sink_eval(
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
        LOGGER.warning("sink_eval got unexpected kwargs: %s", kwargs)
    if prepared_kvs:
        raise ValueError(
            "sink prepares ContextBlocks internally and does not consume prepared KV artifacts."
        )
    terminal = terminal_span(prompt)
    context_parts = [
        (part_index, span)
        for part_index, span in enumerate(prompt.parts[:-1])
        if span.kind == "context"
    ]
    if not context_parts:
        return _ordinary_prompt_eval(model=model, input_ids=ids_2d(prompt))

    flops_counter = AutoFlopsCalculator(model)
    num_flops = 0
    leading = prompt.parts[0] if prompt.parts[0].kind == "inline" else None
    if leading is not None:
        leading_ids = ids_2d(prompt, leading.start, leading.end).to(model.device)
        context_len = leading_ids.size(1)
        context_cache = get_kv_caches(model, input_ids=leading_ids)[0].to_hf_cache(
            config=model.config
        )
        num_flops += flops_counter.forward_flops(
            batch_size=1,
            seq_len=context_len,
            logits_rows=1,
        )
    else:
        context_cache = None
        context_len = 0
        warn(
            "sink_eval requires a leading Inline to create the sink cache, but none was provided."
        )

    doc_ids = [
        prompt.input_ids[span.start:span.end].to(model.device)
        for _, span in context_parts
    ]
    doc_lens = [ids.size(0) for ids in doc_ids]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    assert isinstance(pad_id, int)
    batched_doc_ids = pad_sequence(doc_ids, batch_first=True, padding_value=pad_id)
    batch_size, max_seq_len = batched_doc_ids.shape
    batch_pos_ids = torch.arange(
        max_seq_len, dtype=torch.long, device=model.device
    ).unsqueeze(0).repeat(batch_size, 1)
    offsets = torch.tensor(
        [span.start for _, span in context_parts],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(1)
    batch_pos_ids += offsets
    lengths = torch.tensor(doc_lens, device=model.device).unsqueeze(1)
    attention_mask = (
        torch.arange(max_seq_len, device=model.device).unsqueeze(0) < lengths
    ).long()
    if context_cache is not None:
        context_cache.batch_repeat_interleave(batch_size)
        attention_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, context_len), dtype=torch.long, device=model.device
                ),
                attention_mask,
            ],
            dim=1,
        )

    with torch.no_grad():
        doc_outputs = model.model(
            input_ids=batched_doc_ids,
            past_key_values=context_cache,
            position_ids=batch_pos_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    if doc_outputs.past_key_values is None:
        raise ValueError("sink batched ContextBlock prefill returned no KV cache.")
    new_kvs = doc_outputs.past_key_values
    num_flops += flops_counter.body_flops(
        batch_size=batch_size,
        seq_len=max_seq_len,
        cache_len=context_len,
    )

    leading_cache = (
        select_hf_cache(
            cache=new_kvs,
            batch_indices=torch.tensor([0], dtype=torch.long, device=model.device),
            seq_indices=torch.arange(context_len, dtype=torch.long, device=model.device),
        )
        if leading is not None
        else None
    )
    prepared_contexts: dict[int, DynamicCache] = {}
    for row, ((part_index, _), doc_len) in enumerate(
        zip(context_parts, doc_lens, strict=True)
    ):
        prepared_contexts[part_index] = select_hf_cache(
            cache=new_kvs,
            batch_indices=torch.tensor([row], dtype=torch.long, device=model.device),
            seq_indices=torch.arange(
                context_len,
                context_len + doc_len,
                dtype=torch.long,
                device=model.device,
            ),
        )

    full_kv: DynamicCache | None = None
    physical_len = 0
    for part_index, span in enumerate(prompt.parts[:-1]):
        if span.kind == "context":
            block = prepared_contexts[part_index]
            full_kv = block if full_kv is None else concat_hf_caches([full_kv, block])
            physical_len += span.end - span.start
            continue
        if leading is not None and part_index == 0:
            assert leading_cache is not None
            full_kv = leading_cache
            physical_len = context_len
            continue
        inline_ids = ids_2d(prompt, span.start, span.end).to(model.device)
        if inline_ids.size(1) == 0:
            continue
        with torch.no_grad():
            inline_outputs = model.model(
                input_ids=inline_ids,
                past_key_values=full_kv,
                position_ids=torch.arange(
                    span.start, span.end, dtype=torch.long, device=model.device
                ).unsqueeze(0),
                cache_position=torch.arange(
                    physical_len,
                    physical_len + inline_ids.size(1),
                    device=model.device,
                ),
                attention_mask=torch.ones(
                    (1, physical_len + inline_ids.size(1)),
                    dtype=torch.long,
                    device=model.device,
                ),
                use_cache=True,
            )
        if inline_outputs.past_key_values is None:
            raise ValueError("sink Inline prefill returned no KV cache.")
        full_kv = inline_outputs.past_key_values
        num_flops += flops_counter.body_flops(
            batch_size=1,
            seq_len=inline_ids.size(1),
            cache_len=physical_len,
        )
        physical_len += inline_ids.size(1)

    if full_kv is None:
        raise ValueError("sink produced no prefix KV cache.")
    if physical_len != terminal.start:
        raise ValueError(
            f"sink prefix length mismatch: expected {terminal.start}, got {physical_len}."
        )

    terminal_ids = ids_2d(prompt, terminal.start, terminal.end).to(model.device)
    terminal_positions = torch.arange(
        terminal.start, terminal.end, dtype=torch.long, device=model.device
    ).unsqueeze(0)
    full_attention_mask = torch.ones(
        (1, physical_len + terminal_ids.size(1)),
        dtype=torch.long,
        device=model.device,
    )
    with torch.no_grad():
        outputs = model(
            input_ids=terminal_ids,
            past_key_values=full_kv,
            position_ids=terminal_positions,
            cache_position=torch.arange(
                physical_len,
                physical_len + terminal_ids.size(1),
                device=model.device,
            ),
            attention_mask=full_attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    if outputs.past_key_values is None:
        raise ValueError("sink terminal prefill returned no KV cache.")
    num_flops += flops_counter.forward_flops(
        batch_size=1,
        seq_len=terminal_ids.size(1),
        cache_len=physical_len,
        logits_rows=1,
    )

    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    generation_input = torch.cat(
        [
            torch.full(
                (1, physical_len), dummy_id, dtype=torch.long, device=model.device
            ),
            terminal_ids,
        ],
        dim=1,
    )
    return PrefillResult(
        logits=outputs.logits,
        past_key_values=outputs.past_key_values,
        generation_input_ids=generation_input,
        position_ids=terminal_positions[:, -1:],
        attention_mask=full_attention_mask,
        flops=num_flops,
    )
