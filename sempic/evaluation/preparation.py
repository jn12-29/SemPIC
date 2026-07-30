"""Prepare prompt source KV artifacts for evaluation consumers."""

from dataclasses import dataclass

import torch

from ..cache import KVCache, get_kv_caches, quantize_kv_cache_sd
from ..cache.compress import ScorerPress
from ..cache_comb.abc import PreparedKVMapping, TokenizerType
from ..cache_comb.methods import CONTEXT_BLOCK_METHODS, WHOLE_PREFIX_METHODS
from ..model import SupportedModel
from ..packet_wrapper import PacketWrapper
from ..prompt import TokenizedPrompt
from ..utils.lora import get_causal_lm_body, lora_adapters_enabled
from ..utils.runtime import DebugRecorder, kv_caches_summary, tensor_summary
from .runtime import QuantizationConfig


PreparedSourcePart = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class PreparedPromptKVs:
    prepared_kvs: PreparedKVMapping
    num_orig_tokens: int
    num_wrapped_tokens: int


def prepared_source_parts(
    prompt: TokenizedPrompt,
    method_name: str,
) -> list[PreparedSourcePart]:
    terminal = prompt.parts[-1]
    if method_name in WHOLE_PREFIX_METHODS:
        if terminal.start == 0:
            raise ValueError("single_cache requires a non-empty prompt prefix.")
        return [(len(prompt.parts) - 1, 0, terminal.start)]
    if method_name in CONTEXT_BLOCK_METHODS:
        return [
            (part_index, span.start, span.end)
            for part_index, span in enumerate(prompt.parts)
            if span.kind == "context"
        ]
    return []


def prepare_prompt_kvs(
    *,
    model: SupportedModel,
    tokenizer: TokenizerType,
    prompt: TokenizedPrompt,
    method_name: str,
    packet_wrapper: PacketWrapper | None = None,
    compressor: ScorerPress | None = None,
    keep_filler_tokens: bool = False,
    quantization_config: QuantizationConfig | None = None,
    lora_adapter_name: str | None = None,
    debug_recorder: DebugRecorder | None = None,
) -> PreparedPromptKVs:
    source_parts = prepared_source_parts(prompt, method_name)
    source_ids = [
        prompt.input_ids[start:end].to(model.device)
        for _, start, end in source_parts
    ]
    source_lengths = [ids.numel() for ids in source_ids]
    num_orig_tokens = 0
    num_wrapped_tokens = 0

    document_input_ids: torch.Tensor | None = None
    attn_mask = torch.empty((0, 0), dtype=torch.long, device=model.device)
    input_embeds = torch.empty((0, 0, 0), device=model.device)
    if source_ids:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        max_len = max(source_lengths)
        document_input_ids = torch.full(
            (len(source_ids), max_len),
            pad_id,
            dtype=torch.long,
            device=model.device,
        )
        attn_mask = torch.zeros_like(document_input_ids)
        for row, ids in enumerate(source_ids):
            document_input_ids[row, :ids.numel()] = ids
            attn_mask[row, :ids.numel()] = 1
        input_embeds = get_causal_lm_body(model).embed_tokens(document_input_ids)

    if debug_recorder is not None:
        debug_recorder.record_json(
            "token_lengths",
            {
                "document_input_ids": debug_recorder.token_ids(document_input_ids),
                "attention_mask": tensor_summary(attn_mask),
                "input_embeds": tensor_summary(input_embeds),
                "document_token_lengths": [
                    int(attn_mask[row_idx].sum().item())
                    for row_idx in range(attn_mask.size(0))
                ],
            },
        )

    if source_ids and packet_wrapper is not None:
        filler_length = packet_wrapper.header_len + packet_wrapper.trailer_len
        wrapped_input_embeds = torch.zeros(
            (
                input_embeds.size(0),
                input_embeds.size(1) + filler_length,
                input_embeds.size(2),
            ),
        ).to(input_embeds)

        for row in range(input_embeds.size(0)):
            attn_mask_row = attn_mask[row]
            seq_len = int(attn_mask_row.sum().item())
            wrapped_input_embeds[row, :seq_len + filler_length, :] = packet_wrapper.wrap(
                input_embeds[row, :seq_len, :]
            )
            num_orig_tokens += seq_len
            num_wrapped_tokens += seq_len + filler_length

        wrapped_lengths = torch.tensor(
            [length + filler_length for length in source_lengths],
            device=model.device,
        ).unsqueeze(1)
        attn_mask = (
            torch.arange(
                input_embeds.size(1) + filler_length,
                device=model.device,
            ).unsqueeze(0)
            < wrapped_lengths
        ).long()
        input_embeds = wrapped_input_embeds

    if debug_recorder is not None:
        debug_recorder.record_json(
            "document_kv_stage",
            {
                "lora_enabled_for_document_kv": lora_adapter_name is not None,
                "compressor": type(compressor).__name__ if compressor is not None else None,
            },
        )

    if source_ids and compressor is not None:
        kv_caches: list[KVCache] = []
        for batch_index in range(input_embeds.size(0)):
            input_embed = input_embeds[batch_index:batch_index + 1, :, :]
            mask = attn_mask[batch_index:batch_index + 1, :]
            seq_len = int(mask.sum().item())
            input_embed = input_embed[:, :seq_len, :]

            if keep_filler_tokens:
                assert packet_wrapper is not None, (
                    "keep_filler_tokens is only compatible with packet wrapper."
                )
                indices_to_keep = list(range(packet_wrapper.header_len)) + list(
                    range(seq_len - packet_wrapper.trailer_len, seq_len)
                )
            else:
                indices_to_keep = None

            if lora_adapter_name is not None:
                with lora_adapters_enabled(model, adapter_name=lora_adapter_name):
                    kv_cache = get_kv_caches(
                        model=model,
                        input_embeds=input_embed,
                        compressor=compressor,
                        indices_to_keep=indices_to_keep,
                    )[0]
            else:
                kv_cache = get_kv_caches(
                    model=model,
                    input_embeds=input_embed,
                    compressor=compressor,
                    indices_to_keep=indices_to_keep,
                )[0]
            kv_caches.append(kv_cache)
    elif source_ids:
        if lora_adapter_name is not None:
            with lora_adapters_enabled(model, adapter_name=lora_adapter_name):
                kv_caches = get_kv_caches(
                    model=model,
                    input_embeds=input_embeds,
                    attention_mask=attn_mask,
                    compressor=compressor,
                )
        else:
            kv_caches = get_kv_caches(
                model=model,
                input_embeds=input_embeds,
                attention_mask=attn_mask,
                compressor=compressor,
            )
    else:
        kv_caches = []

    if debug_recorder is not None:
        debug_recorder.record_json(
            "document_kv_shapes",
            {"document_kvs": kv_caches_summary(kv_caches)},
        )

    if quantization_config is not None:
        kv_caches = [
            KVCache.from_state_dict(
                quantize_kv_cache_sd(
                    kv_cache.state_dict(),
                    num_bits=quantization_config["num_bits"],
                    axis=quantization_config["axis"],
                    q_group_size=quantization_config["group_size"],
                )
            )
            for kv_cache in kv_caches
        ]
        if debug_recorder is not None:
            debug_recorder.record_json(
                "quantized_document_kv_shapes",
                {"document_kvs": kv_caches_summary(kv_caches)},
            )

    prepared_kvs = {
        part_index: kv_cache
        for (part_index, _, _), kv_cache in zip(source_parts, kv_caches, strict=True)
    }
    return PreparedPromptKVs(
        prepared_kvs=prepared_kvs,
        num_orig_tokens=num_orig_tokens,
        num_wrapped_tokens=num_wrapped_tokens,
    )


__all__ = [
    "PreparedPromptKVs",
    "PreparedSourcePart",
    "prepare_prompt_kvs",
    "prepared_source_parts",
]
