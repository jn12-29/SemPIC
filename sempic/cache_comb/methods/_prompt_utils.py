from typing import NamedTuple

import torch

from ...cache import KVCache, concate_kv_caches
from ...model import SupportedModel
from ...prompt import TokenSpan, TokenizedPrompt
from ..abc import PreparedKVMapping, PrefillResult, TokenizerType
from ..utils.flops import AutoFlopsCalculator


class RecomputeInputs(NamedTuple):
    input_ids: torch.Tensor
    prefix_cache: KVCache
    candidate_indices: tuple[int, ...]
    inline_indices: tuple[int, ...]


def terminal_span(prompt: TokenizedPrompt) -> TokenSpan:
    if not prompt.parts or prompt.parts[-1].kind != "inline":
        raise ValueError("TokenizedPrompt must end with an Inline span.")
    span = prompt.parts[-1]
    if span.start == span.end:
        raise ValueError("TokenizedPrompt must end with a non-empty Inline span.")
    return span


def ids_2d(prompt: TokenizedPrompt, start: int = 0, end: int | None = None) -> torch.Tensor:
    return prompt.input_ids[start:end].unsqueeze(0)


def inline_only_ids(prompt: TokenizedPrompt) -> torch.Tensor:
    chunks = [prompt.input_ids[span.start:span.end] for span in prompt.parts if span.kind == "inline"]
    if not chunks:
        raise ValueError("TokenizedPrompt contains no Inline tokens.")
    return torch.cat(chunks).unsqueeze(0)


def context_parts(prompt: TokenizedPrompt) -> tuple[tuple[int, TokenSpan], ...]:
    return tuple(
        (part_index, span)
        for part_index, span in enumerate(prompt.parts)
        if span.kind == "context"
    )


def require_exact_context_kvs(prompt: TokenizedPrompt, prepared_kvs: PreparedKVMapping) -> None:
    expected = {part_index for part_index, _ in context_parts(prompt)}
    actual = set(prepared_kvs)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Prepared ContextBlock KV mapping mismatch: missing={missing}, extra={extra}.")


def require_one_to_one_context_kvs(
    method_name: str,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
) -> None:
    require_exact_context_kvs(prompt, prepared_kvs)
    for part_index, span in context_parts(prompt):
        kv = prepared_kvs[part_index]
        source_len = span.end - span.start
        physical_len = kv[0].key.size(2)
        position_len = kv[0].position_ids.size(1)
        if physical_len != source_len or position_len != source_len:
            raise ValueError(
                f"{method_name} requires one physical KV position per source token; "
                f"part {part_index} has source={source_len}, "
                f"physical={physical_len}, positions={position_len}."
            )


def prepare_recompute_inputs(
    method_name: str,
    model: SupportedModel,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
) -> RecomputeInputs:
    """Assemble the canonical prefix while preserving typed recompute policy."""
    require_one_to_one_context_kvs(method_name, prompt, prepared_kvs)
    contexts = context_parts(prompt)
    if not contexts:
        raise ValueError(f"{method_name} requires at least one ContextBlock.")

    template = prepared_kvs[contexts[0][0]][0]
    prefix_caches: list[KVCache] = []
    candidate_indices: list[int] = []
    inline_indices: list[int] = []

    for part_index, span in enumerate(prompt.parts[:-1]):
        span_positions = torch.arange(span.start, span.end, device=model.device).unsqueeze(0)
        if span.kind == "context":
            block = prepared_kvs[part_index].copy(clone_tensor=True)
            prefix_caches.append(block)
            if span.start != 0:
                candidate_indices.extend(range(span.start, span.end))
        else:
            prefix_caches.append(KVCache.create_dummy(
                num_layers=len(prepared_kvs[contexts[0][0]].layers),
                batch_size=1,
                num_heads=template.key.size(1),
                key_head_dim=template.key.size(3),
                value_head_dim=template.value.size(3),
                seq_len=span.end - span.start,
                position_ids=span_positions,
                dtype=template.key.dtype,
                device=model.device,
            ))
            inline_indices.extend(range(span.start, span.end))

    return RecomputeInputs(
        input_ids=ids_2d(prompt).to(model.device),
        prefix_cache=concate_kv_caches(prefix_caches),
        candidate_indices=tuple(candidate_indices),
        inline_indices=tuple(inline_indices),
    )


def recompute_prefix_indices(
    inputs: RecomputeInputs,
    selected_context_indices: list[int],
) -> list[int]:
    return sorted(selected_context_indices + list(inputs.inline_indices))


def finish_recomputed_prefill(
    *,
    model: SupportedModel,
    tokenizer: TokenizerType,
    prompt: TokenizedPrompt,
    cache: KVCache,
    terminal_hidden_states: torch.Tensor,
    terminal_ids: torch.Tensor,
    prefix_physical_len: int,
    flops: int,
) -> PrefillResult:
    last_hidden = model.model.norm(terminal_hidden_states[:, -1:, :])
    logits = model.lm_head(last_hidden)
    flops += AutoFlopsCalculator(model).output_flops(
        batch_size=last_hidden.size(0), hidden_rows=1, logits_rows=1
    )
    physical_len = cache[0].key.size(2)
    dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
    dummy_prefix = torch.full(
        (terminal_ids.size(0), prefix_physical_len),
        dummy_id,
        dtype=torch.long,
        device=model.device,
    )
    generation_input_ids = torch.cat([dummy_prefix, terminal_ids], dim=1)
    if generation_input_ids.size(1) != physical_len:
        raise ValueError(
            "Recomputed prompt and physical KV cache lengths do not match: "
            f"input={generation_input_ids.size(1)}, cache={physical_len}."
        )
    return PrefillResult(
        logits=logits,
        past_key_values=cache.to_hf_cache(config=model.config),
        generation_input_ids=generation_input_ids,
        position_ids=torch.tensor(
            [[prompt.input_ids.numel() - 1]], dtype=torch.long, device=model.device
        ),
        attention_mask=torch.ones_like(generation_input_ids),
        flops=flops,
    )
