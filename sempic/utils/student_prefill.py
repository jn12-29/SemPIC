from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)

from ..cache import KVCache, get_kv_caches_with_grad
from ..cache.rotate import rerotate_embeddings
from ..packet_wrapper import PacketWrapper
from ..prefill import (
    ContextPlacement,
    InterleavedPrefillLayout,
    PrefillSegment,
    build_interleaved_layout,
)
from ..prompt import TokenizedPrompt
from .generate import GenerationCacheAccess
from .lora import (
    get_causal_lm_body,
    get_model_device,
    lora_adapters_disabled,
    lora_adapters_enabled,
)


TrainAttentionBackend = Literal["flex", "sdpa"]


def _dense_train_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frontiers: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    def score_mod(
        score: torch.Tensor,
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
    ) -> torch.Tensor:
        del head_index
        return torch.where(
            key_index <= frontiers[batch_index, query_index],
            score,
            torch.full_like(score, -torch.inf),
        )

    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        scale=scale,
        enable_gqa=True,
        kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
    )


_COMPILED_DENSE_TRAIN_FLEX_ATTENTION = torch.compile(
    _dense_train_flex_attention,
    dynamic=True,
)


def _collect_train_flex_attention_shapes(
    samples: Sequence[Mapping[str, Any]],
    generation_cache: GenerationCacheAccess,
    packet_wrapper: PacketWrapper | None,
    device: torch.device,
) -> tuple[tuple[int, int], ...]:
    wrapper_tokens = (
        packet_wrapper.header_len + packet_wrapper.trailer_len
        if packet_wrapper is not None
        else 0
    )
    shapes: list[tuple[int, int]] = []
    seen_shapes: set[tuple[int, int]] = set()
    for sample in samples:
        prompt = sample["prompt"]
        if not isinstance(prompt, TokenizedPrompt):
            raise TypeError("Each training sample must contain a TokenizedPrompt.")
        metadata_method = getattr(generation_cache, "metadata", None)
        metadata = (
            metadata_method(sample["semantic_key"])
            if callable(metadata_method)
            else None
        )
        if metadata is not None:
            sequence_lengths = metadata.sequence_lengths
            teacher_sequence = torch.zeros(
                sequence_lengths[0] if sequence_lengths else 0,
                dtype=torch.long,
                device=device,
            )
        else:
            generation = generation_cache.get(sample["semantic_key"])
            if generation is None or not generation["sequences"]:
                raise KeyError(
                    "Missing teacher generation cache entry during FlexAttention probe."
                )
            sequence_lengths = tuple(
                sequence.numel() for sequence in generation["sequences"]
            )
            teacher_sequence = generation["sequences"][0].to(device)
        if len(sequence_lengths) != 1:
            raise ValueError(
                "Train FlexAttention currently requires one teacher sequence per sample."
            )
        context_lengths = {
            (0, part_index): span.end - span.start + wrapper_tokens
            for part_index, span in enumerate(prompt.parts)
            if span.kind == "context"
        }
        layout = build_student_layout(
            [prompt],
            [teacher_sequence],
            context_lengths,
            device,
        )
        shape = (layout.input_ids.size(1), layout.physical_valid.size(1))
        if shape not in seen_shapes:
            seen_shapes.add(shape)
            shapes.append(shape)
    return tuple(shapes)


def probe_train_flex_attention_shapes(
    samples: Sequence[Mapping[str, Any]],
    model: Any,
    generation_cache: GenerationCacheAccess,
    packet_wrapper: PacketWrapper | None,
) -> None:
    device = get_model_device(model)
    if device.type != "cuda":
        return
    causal_lm = _unwrap_causal_lm(model)
    config = causal_lm.config
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // query_heads))
    dtype = next(causal_lm.parameters()).dtype
    scale = float(causal_lm.model.layers[0].self_attn.scaling)
    shapes = _collect_train_flex_attention_shapes(
        samples,
        generation_cache,
        packet_wrapper,
        device,
    )
    for query_length, key_length in shapes:
        frontiers = torch.full(
            (1, query_length),
            key_length - 1,
            dtype=torch.long,
            device=device,
        )
        query = torch.empty_strided(
            (1, query_heads, query_length, head_dim),
            (
                query_length * query_heads * head_dim,
                head_dim,
                query_heads * head_dim,
                1,
            ),
            dtype=dtype,
            device=device,
        ).zero_().requires_grad_()
        key = torch.zeros(
            (1, kv_heads, key_length, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        value = torch.zeros_like(key, requires_grad=True)
        output = _COMPILED_DENSE_TRAIN_FLEX_ATTENTION(
            query,
            key,
            value,
            frontiers,
            scale,
        )
        torch.autograd.grad(output.sum(), (query, key, value))

    torch.cuda.synchronize(device)


@dataclass(frozen=True, slots=True)
class ContextWorkItem:
    sample_index: int
    part_index: int
    input_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class StudentBatchLayout:
    input_ids: torch.Tensor
    query_valid: torch.Tensor
    query_position_ids: torch.Tensor
    context_valid: torch.Tensor
    context_position_ids: torch.Tensor
    physical_valid: torch.Tensor
    physical_position_ids: torch.Tensor
    physical_source_indices: torch.Tensor
    query_frontiers: torch.Tensor
    target_rows: tuple[torch.Tensor, ...]
    context_placements: tuple[tuple[ContextPlacement, ...], ...]
    interleaved_layouts: tuple[InterleavedPrefillLayout, ...]


def collect_context_blocks(
    prompts: Sequence[TokenizedPrompt],
) -> list[ContextWorkItem]:
    items: list[ContextWorkItem] = []
    for sample_index, prompt in enumerate(prompts):
        for part_index, span in enumerate(prompt.parts):
            if span.kind == "context":
                items.append(ContextWorkItem(
                    sample_index=sample_index,
                    part_index=part_index,
                    input_ids=prompt.input_ids[span.start:span.end],
                ))
    return items


def _length_aware_context_embeds(
    source_embeds: torch.Tensor,
    source_lengths: Sequence[int],
    packet_wrapper: PacketWrapper | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rows: list[torch.Tensor] = []
    physical_lengths: list[int] = []
    for row_index, source_length in enumerate(source_lengths):
        row = source_embeds[row_index, :source_length]
        if packet_wrapper is not None:
            header = packet_wrapper.header[0].to(device=row.device, dtype=row.dtype)
            trailer = packet_wrapper.trailer[0].to(device=row.device, dtype=row.dtype)
            row = torch.cat((header, row, trailer), dim=0)
        rows.append(row)
        physical_lengths.append(row.size(0))

    max_length = max(physical_lengths)
    embeds = torch.stack([
        F.pad(row, (0, 0, 0, max_length - row.size(0)))
        for row in rows
    ])
    mask = torch.zeros(
        (len(rows), max_length),
        dtype=torch.long,
        device=source_embeds.device,
    )
    for row_index, length in enumerate(physical_lengths):
        mask[row_index, :length] = 1
    return embeds, mask, physical_lengths


def prepare_context_blocks(
    items: Sequence[ContextWorkItem],
    model: Any,
    lora_enabled: bool,
    lora_adapter_name: str | None,
    packet_wrapper: PacketWrapper | None,
    checkpoint_grad: bool = False,
) -> tuple[dict[tuple[int, int], KVCache], dict[tuple[int, int], int]]:
    if not items:
        return {}, {}

    model_device = get_model_device(model)
    body = get_causal_lm_body(model)
    source_lengths = [int(item.input_ids.numel()) for item in items]
    max_source_length = max(source_lengths)
    padded_ids = torch.zeros(
        (len(items), max_source_length),
        dtype=torch.long,
        device=model_device,
    )
    for row_index, item in enumerate(items):
        length = source_lengths[row_index]
        padded_ids[row_index, :length] = item.input_ids.to(model_device)

    adapter_context = (
        lora_adapters_enabled(model, adapter_name=lora_adapter_name)
        if lora_enabled
        else nullcontext()
    )
    forward_context_factory = (
        (lambda: lora_adapters_enabled(model, adapter_name=lora_adapter_name))
        if lora_enabled and checkpoint_grad
        else None
    )
    with adapter_context:
        source_embeds = body.embed_tokens(padded_ids)
        embeds, attention_mask, physical_lengths = _length_aware_context_embeds(
            source_embeds,
            source_lengths,
            packet_wrapper,
        )
        caches = get_kv_caches_with_grad(
            model,
            input_embeds=embeds,
            attention_mask=attention_mask,
            checkpoint_grad=checkpoint_grad,
            forward_context_factory=forward_context_factory,
        )

    cache_by_item = {
        (item.sample_index, item.part_index): caches[item_index]
        for item_index, item in enumerate(items)
    }
    lengths_by_item = {
        (item.sample_index, item.part_index): physical_lengths[item_index]
        for item_index, item in enumerate(items)
    }
    return cache_by_item, lengths_by_item


def build_student_layout(
    prompts: Sequence[TokenizedPrompt],
    teacher_sequences: Sequence[torch.Tensor],
    context_lengths: Mapping[tuple[int, int], int],
    device: torch.device,
) -> StudentBatchLayout:
    if len(prompts) != len(teacher_sequences) or not prompts:
        raise ValueError("Prompts and teacher sequences must have the same non-zero length.")

    query_ids_by_sample: list[torch.Tensor] = []
    query_positions_by_sample: list[torch.Tensor] = []
    context_positions_by_sample: list[torch.Tensor] = []
    interleaved_layouts: list[InterleavedPrefillLayout] = []
    target_rows: list[torch.Tensor] = []
    placements_by_sample: list[tuple[ContextPlacement, ...]] = []

    for sample_index, (prompt, teacher_sequence) in enumerate(zip(prompts, teacher_sequences, strict=True)):
        if teacher_sequence.ndim != 1 or teacher_sequence.numel() == 0:
            raise ValueError("Training requires one non-empty one-dimensional teacher sequence per sample.")

        query_ids: list[torch.Tensor] = []
        query_positions: list[torch.Tensor] = []
        context_positions: list[torch.Tensor] = []
        segments: list[PrefillSegment] = []
        logical_position = 0
        terminal_query_row: int | None = None

        for part_index, span in enumerate(prompt.parts):
            if span.kind == "inline":
                ids = prompt.input_ids[span.start:span.end].to(device)
                positions = torch.arange(
                    logical_position,
                    logical_position + ids.numel(),
                    dtype=torch.long,
                    device=device,
                )
                query_ids.append(ids)
                query_positions.append(positions)
                segments.append(PrefillSegment(
                    kind="inline",
                    position_ids=positions,
                    canonical_start=span.start,
                    canonical_end=span.end,
                    terminal=part_index == len(prompt.parts) - 1,
                ))
                logical_position += ids.numel()
                if part_index == len(prompt.parts) - 1:
                    terminal_query_row = sum(chunk.numel() for chunk in query_ids) - 1
            else:
                length = context_lengths.get((sample_index, part_index))
                if length is None:
                    raise KeyError(f"Missing prepared length for ContextBlock {(sample_index, part_index)}.")
                positions = torch.arange(
                    logical_position,
                    logical_position + length,
                    dtype=torch.long,
                    device=device,
                )
                context_positions.append(positions)
                segments.append(PrefillSegment(
                    kind="context",
                    position_ids=positions,
                    canonical_start=span.start,
                    canonical_end=span.end,
                    part_index=part_index,
                ))
                logical_position += length

        assert terminal_query_row is not None
        flat_query_ids = torch.cat(query_ids)
        flat_query_positions = torch.cat(query_positions)
        forcing_ids = teacher_sequence[:-1].to(device)
        forcing_positions = torch.arange(
            logical_position,
            logical_position + forcing_ids.numel(),
            dtype=torch.long,
            device=device,
        )
        if forcing_ids.numel():
            segments.append(PrefillSegment(
                kind="inline",
                position_ids=forcing_positions,
            ))
        interleaved_layout = build_interleaved_layout(segments)
        first_forcing_row = flat_query_ids.numel()
        flat_query_ids = torch.cat((flat_query_ids, forcing_ids))
        flat_query_positions = torch.cat((flat_query_positions, forcing_positions))
        rows = torch.cat((
            torch.tensor([terminal_query_row], dtype=torch.long, device=device),
            torch.arange(
                first_forcing_row,
                first_forcing_row + forcing_ids.numel(),
                dtype=torch.long,
                device=device,
            ),
        ))

        query_ids_by_sample.append(flat_query_ids)
        query_positions_by_sample.append(flat_query_positions)
        context_positions_by_sample.append(
            torch.cat(context_positions) if context_positions else torch.empty(0, dtype=torch.long, device=device)
        )
        target_rows.append(rows)
        placements_by_sample.append(interleaved_layout.context_placements)
        interleaved_layouts.append(interleaved_layout)

    input_ids, query_valid = _pad_1d(query_ids_by_sample, fill_value=0)
    query_position_ids, _ = _pad_1d(query_positions_by_sample, fill_value=0)
    context_position_ids, context_valid = _pad_1d(context_positions_by_sample, fill_value=0)
    physical_position_ids, physical_valid = _pad_1d(
        [row.physical_position_ids for row in interleaved_layouts],
        fill_value=0,
    )
    query_frontiers, _ = _pad_1d(
        [row.inline_physical_frontiers for row in interleaved_layouts],
        fill_value=0,
    )
    max_context_length = context_position_ids.size(1)
    max_query_length = input_ids.size(1)
    physical_sources: list[torch.Tensor] = []
    for row in interleaved_layouts:
        sources = row.physical_source_indices.clone()
        inline = ~row.physical_is_context
        sources[inline] += max_context_length - row.context_length
        physical_sources.append(sources)
    physical_source_indices, _ = _pad_1d(
        physical_sources,
        fill_value=max_context_length + max_query_length,
    )
    return StudentBatchLayout(
        input_ids=input_ids,
        query_valid=query_valid,
        query_position_ids=query_position_ids,
        context_valid=context_valid,
        context_position_ids=context_position_ids,
        physical_valid=physical_valid,
        physical_position_ids=physical_position_ids,
        physical_source_indices=physical_source_indices,
        query_frontiers=query_frontiers,
        target_rows=tuple(target_rows),
        context_placements=tuple(placements_by_sample),
        interleaved_layouts=tuple(interleaved_layouts),
    )


def _pad_1d(rows: Sequence[torch.Tensor], fill_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(row.numel() for row in rows)
    values = torch.full(
        (len(rows), max_length),
        fill_value=fill_value,
        dtype=rows[0].dtype,
        device=rows[0].device,
    )
    valid = torch.zeros((len(rows), max_length), dtype=torch.bool, device=rows[0].device)
    for row_index, row in enumerate(rows):
        length = row.numel()
        values[row_index, :length] = row
        valid[row_index, :length] = True
    return values, valid


def build_logical_causal_mask(
    query_position_ids: torch.Tensor,
    context_position_ids: torch.Tensor,
    query_valid: torch.Tensor,
    context_valid: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    if query_position_ids.shape != query_valid.shape:
        raise ValueError("Query positions and validity mask must have the same shape.")
    if context_position_ids.shape != context_valid.shape:
        raise ValueError("Context positions and validity mask must have the same shape.")
    key_positions = torch.cat((context_position_ids, query_position_ids), dim=1)
    key_valid = torch.cat((context_valid, query_valid), dim=1)
    visible = key_positions.unsqueeze(1) <= query_position_ids.unsqueeze(2)
    visible &= key_valid.unsqueeze(1)
    visible &= query_valid.unsqueeze(2)
    mask = torch.zeros(visible.shape, dtype=dtype, device=query_position_ids.device)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def _unwrap_causal_lm(model: Any) -> LlamaForCausalLM | Qwen3ForCausalLM:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if isinstance(base_model, (LlamaForCausalLM, Qwen3ForCausalLM)):
        return base_model
    raise TypeError(f"Unsupported student prefill model type: {type(base_model).__name__}.")


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    batch_size, num_heads, sequence_length, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch_size, num_heads, groups, sequence_length, head_dim)
        .reshape(batch_size, num_heads * groups, sequence_length, head_dim)
    )


@dataclass(frozen=True, slots=True)
class PackedContextBatch:
    key: torch.Tensor
    value: torch.Tensor


def pack_rerotated_contexts(
    layout: StudentBatchLayout,
    prepared_contexts: Mapping[tuple[int, int], KVCache],
    num_layers: int,
    body: Any,
    nope_dim: int | None,
) -> PackedContextBatch | None:
    occurrence_keys: list[torch.Tensor] = []
    occurrence_values: list[torch.Tensor] = []
    occurrence_old_positions: list[torch.Tensor] = []
    occurrence_new_positions: list[torch.Tensor] = []
    row_lengths: list[int] = []

    for sample_index, placements in enumerate(layout.context_placements):
        row_length = 0
        for placement in placements:
            cache = prepared_contexts[(sample_index, placement.part_index)].copy()
            layer_values = [cache[layer_index] for layer_index in range(num_layers)]
            expected_length = placement.source_end - placement.source_start
            if any(item.key.size(0) != 1 for item in layer_values):
                raise ValueError("Prepared Context KV must have batch size one.")
            if any(item.key.size(2) != expected_length for item in layer_values):
                raise ValueError("Prepared Context KV length does not match its placement.")
            occurrence_keys.append(torch.stack([item.key[0] for item in layer_values]))
            occurrence_values.append(torch.stack([item.value[0] for item in layer_values]))
            occurrence_old_positions.append(torch.stack([
                item.position_ids[0] for item in layer_values
            ]))
            occurrence_new_positions.append(
                placement.position_ids.unsqueeze(0).expand(num_layers, -1)
            )
            row_length += expected_length
        row_lengths.append(row_length)

    if not occurrence_keys:
        return None

    packed_key = torch.cat(occurrence_keys, dim=2)
    packed_value = torch.cat(occurrence_values, dim=2)
    old_positions = torch.cat(occurrence_old_positions, dim=1)
    new_positions = torch.cat(occurrence_new_positions, dim=1)
    rotated_key = rerotate_embeddings(
        packed_key,
        body.rotary_emb,
        old_positions,
        new_positions,
        nope_dim=nope_dim,
    )

    max_length = layout.context_position_ids.size(1)
    rows_key: list[torch.Tensor] = []
    rows_value: list[torch.Tensor] = []
    offset = 0
    for row_length in row_lengths:
        rows_key.append(F.pad(
            rotated_key[:, :, offset:offset + row_length],
            (0, 0, 0, max_length - row_length),
        ))
        rows_value.append(F.pad(
            packed_value[:, :, offset:offset + row_length],
            (0, 0, 0, max_length - row_length),
        ))
        offset += row_length
    return PackedContextBatch(
        key=torch.stack(rows_key, dim=1),
        value=torch.stack(rows_value, dim=1),
    )


def build_physical_causal_mask(
    query_frontiers: torch.Tensor,
    query_valid: torch.Tensor,
    physical_valid: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    physical_indices = torch.arange(
        physical_valid.size(1),
        device=physical_valid.device,
    )
    visible = physical_indices.view(1, 1, -1) <= query_frontiers.unsqueeze(2)
    visible &= physical_valid.unsqueeze(1)
    visible &= query_valid.unsqueeze(2)
    mask = torch.zeros(visible.shape, dtype=dtype, device=physical_valid.device)
    mask.masked_fill_(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(1)


def _interleave_layer_kv(
    layout: StudentBatchLayout,
    context_key: torch.Tensor,
    context_value: torch.Tensor,
    inline_key: torch.Tensor,
    inline_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sentinel_shape = (inline_key.size(0), inline_key.size(1), 1, inline_key.size(3))
    sentinel_key = inline_key.new_zeros(sentinel_shape)
    sentinel_value = inline_value.new_zeros(sentinel_shape)
    compact_key = torch.cat((context_key, inline_key, sentinel_key), dim=2)
    compact_value = torch.cat((context_value, inline_value, sentinel_value), dim=2)
    gather_index = layout.physical_source_indices[:, None, :, None].expand(
        -1, compact_key.size(1), -1, compact_key.size(3)
    )
    return (
        compact_key.gather(2, gather_index),
        compact_value.gather(2, gather_index),
    )


def _eager_attention(
    attention: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    key = _repeat_kv(key, attention.num_key_value_groups)
    value = _repeat_kv(value, attention.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * attention.scaling
    scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    dropout = float(getattr(attention, "attention_dropout", 0.0))
    probabilities = F.dropout(probabilities, p=dropout, training=attention.training)
    return torch.matmul(probabilities, value).transpose(1, 2).contiguous()


def run_inline_prefill(
    model: Any,
    layout: StudentBatchLayout,
    prepared_contexts: Mapping[tuple[int, int], KVCache],
    attention_backend: TrainAttentionBackend = "sdpa",
) -> tuple[torch.Tensor, ...]:
    causal_lm = _unwrap_causal_lm(model)
    body = causal_lm.model
    for layer_index, decoder_layer in enumerate(body.layers):
        attention_type = getattr(decoder_layer, "attention_type", "full_attention")
        if attention_type != "full_attention":
            raise ValueError(
                f"Student interleaved prefill does not support {attention_type!r} "
                f"at layer {layer_index}."
            )
    hidden_states = body.embed_tokens(layout.input_ids)
    position_embeddings = body.rotary_emb(hidden_states, layout.query_position_ids)
    nope_dim = getattr(causal_lm.config, "qk_nope_head_dim", None)
    packed_context = pack_rerotated_contexts(
        layout,
        prepared_contexts,
        len(body.layers),
        body,
        nope_dim,
    )
    attention_mask = build_physical_causal_mask(
        layout.query_frontiers,
        layout.query_valid,
        layout.physical_valid,
        hidden_states.dtype,
    )

    for layer_index, decoder_layer in enumerate(body.layers):
        residual = hidden_states
        normalized = decoder_layer.input_layernorm(hidden_states)
        attention = decoder_layer.self_attn
        input_shape = normalized.shape[:-1]
        query_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(normalized).view(query_shape).transpose(1, 2)
        key = attention.k_proj(normalized).view(query_shape).transpose(1, 2)
        value = attention.v_proj(normalized).view(query_shape).transpose(1, 2)

        if isinstance(decoder_layer, Qwen3DecoderLayer):
            query = attention.q_norm(query.transpose(1, 2)).transpose(1, 2)
            key = attention.k_norm(key.transpose(1, 2)).transpose(1, 2)
            query, key = qwen3_apply_rotary_pos_emb(query, key, *position_embeddings)
        elif isinstance(decoder_layer, LlamaDecoderLayer):
            query, key = llama_apply_rotary_pos_emb(query, key, *position_embeddings)
        else:
            raise TypeError(f"Unsupported decoder layer type: {type(decoder_layer).__name__}.")

        if packed_context is None:
            context_key = key.new_zeros((key.size(0), key.size(1), 0, key.size(3)))
            context_value = value.new_zeros((value.size(0), value.size(1), 0, value.size(3)))
        else:
            context_key = packed_context.key[layer_index]
            context_value = packed_context.value[layer_index]
        combined_key, combined_value = _interleave_layer_kv(
            layout,
            context_key,
            context_value,
            key,
            value,
        )
        if hidden_states.device.type == "cuda" and attention_backend == "flex":
            if hidden_states.size(0) != 1:
                raise ValueError("Train FlexAttention currently requires forward_batch_size=1.")
            dropout = float(getattr(attention, "attention_dropout", 0.0))
            if attention.training and dropout:
                raise ValueError("Train FlexAttention does not support attention dropout.")
            attended = _COMPILED_DENSE_TRAIN_FLEX_ATTENTION(
                query,
                combined_key,
                combined_value,
                layout.query_frontiers,
                attention.scaling,
            ).transpose(1, 2).contiguous()
        elif hidden_states.device.type == "cuda":
            dropout = float(getattr(attention, "attention_dropout", 0.0))
            attended = F.scaled_dot_product_attention(
                query,
                combined_key,
                combined_value,
                attn_mask=attention_mask,
                dropout_p=dropout if attention.training else 0.0,
                scale=attention.scaling,
                enable_gqa=True,
            ).transpose(1, 2).contiguous()
        else:
            attended = _eager_attention(
                attention,
                query,
                combined_key,
                combined_value,
                attention_mask,
            )
        attended = attended.reshape(*input_shape, -1)
        hidden_states = residual + attention.o_proj(attended)
        residual = hidden_states
        hidden_states = residual + decoder_layer.mlp(decoder_layer.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states * layout.query_valid.unsqueeze(-1)

    hidden_states = body.norm(hidden_states)
    return tuple(
        causal_lm.lm_head(hidden_states[sample_index].index_select(0, rows))
        for sample_index, rows in enumerate(layout.target_rows)
    )


def batched_student_loss(
    samples: Sequence[Mapping[str, Any]],
    model: Any,
    generation_cache: GenerationCacheAccess,
    loss_config: Mapping[str, Any],
    lora_enabled: bool,
    lora_adapter_name: str | None,
    packet_wrapper: PacketWrapper | None,
    kv_gradient_checkpointing: bool = False,
    debug_recorders: Sequence[Any | None] | None = None,
    attention_backend: TrainAttentionBackend = "sdpa",
) -> tuple[torch.Tensor, int, list[float] | None]:
    if not samples:
        raise ValueError("Student microbatch must contain at least one sample.")
    if debug_recorders is not None and len(debug_recorders) != len(samples):
        raise ValueError("debug_recorders must match the microbatch size.")

    device = get_model_device(model)
    source_prompts: list[TokenizedPrompt] = []
    target_prompts: list[TokenizedPrompt] = []
    target_owners: list[int] = []
    target_indices: list[int] = []
    target_counts: list[int] = []
    teacher_sequences: list[torch.Tensor] = []
    teacher_logits: list[torch.Tensor | None] = []
    for sample_index, sample in enumerate(samples):
        prompt = sample["prompt"]
        if not isinstance(prompt, TokenizedPrompt):
            raise TypeError("Each training sample must contain a TokenizedPrompt.")
        generation = generation_cache.get(
            sample["semantic_key"],
            device=device,
        )
        if generation is None:
            raise KeyError("Missing teacher generation cache entry.")
        if not generation["sequences"]:
            raise ValueError("Training requires at least one teacher sequence per sample.")
        if (
            loss_config["type"] == "kl"
            and len(generation["logits"]) != len(generation["sequences"])
        ):
            raise ValueError("KL loss requires one teacher logits tensor per sequence.")
        source_prompts.append(prompt)
        target_counts.append(len(generation["sequences"]))
        for target_index, teacher_sequence in enumerate(generation["sequences"]):
            if teacher_sequence.ndim != 1 or teacher_sequence.numel() == 0:
                raise ValueError("Teacher sequences must be non-empty one-dimensional tensors.")
            if teacher_sequence.dtype != torch.long:
                raise ValueError("Teacher sequences must use torch.long token IDs.")
            vocab_size = int(model.config.vocab_size)
            if bool((teacher_sequence < 0).any()) or bool((teacher_sequence >= vocab_size).any()):
                raise ValueError("Teacher sequence token IDs exceed the student vocabulary.")
            target_logits = (
                generation["logits"][target_index]
                if loss_config["type"] == "kl"
                else None
            )
            if target_logits is not None:
                if target_logits.ndim != 2 or not target_logits.is_floating_point():
                    raise ValueError(
                        "Teacher logits must be a two-dimensional floating-point tensor."
                    )
                if target_logits.shape != (teacher_sequence.numel(), vocab_size):
                    raise ValueError(
                        "Teacher logits must match the teacher sequence length and student vocabulary."
                    )
            target_prompts.append(prompt)
            target_owners.append(sample_index)
            target_indices.append(target_index)
            teacher_sequences.append(teacher_sequence.to(device))
            teacher_logits.append(
                target_logits.to(device) if target_logits is not None else None
            )

    context_items = collect_context_blocks(source_prompts)
    prepared_contexts, context_lengths = prepare_context_blocks(
        context_items,
        model,
        lora_enabled,
        lora_adapter_name,
        packet_wrapper,
        checkpoint_grad=kv_gradient_checkpointing,
    )
    target_contexts: dict[tuple[int, int], KVCache] = {}
    target_context_lengths: dict[tuple[int, int], int] = {}
    for target_row, source_row in enumerate(target_owners):
        for part_index, span in enumerate(source_prompts[source_row].parts):
            if span.kind != "context":
                continue
            source_key = (source_row, part_index)
            target_key = (target_row, part_index)
            target_contexts[target_key] = prepared_contexts[source_key]
            target_context_lengths[target_key] = context_lengths[source_key]

    layout = build_student_layout(
        target_prompts,
        teacher_sequences,
        target_context_lengths,
        device,
    )

    adapter_context = lora_adapters_disabled(model) if lora_enabled else nullcontext()
    with adapter_context:
        logits_by_target = run_inline_prefill(
            model,
            layout,
            target_contexts,
            attention_backend=attention_backend,
        )

    losses_by_sample: list[list[torch.Tensor]] = [[] for _ in samples]
    total_tokens = 0
    for target_row, (student_logits, teacher_sequence, target_logits) in enumerate(
        zip(logits_by_target, teacher_sequences, teacher_logits, strict=True)
    ):
        if student_logits.size(0) != teacher_sequence.numel():
            raise ValueError("Student logits and teacher sequence lengths differ.")
        if loss_config["type"] == "kl":
            assert target_logits is not None
            if target_logits.shape != student_logits.shape:
                raise ValueError(
                    f"Student/teacher logits shape mismatch: {student_logits.shape} vs {target_logits.shape}."
                )
            tau = float(loss_config["tau"])
            teacher_probs = F.softmax(target_logits / tau, dim=-1)
            student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
            target_loss = (tau * tau) * F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="none",
            ).sum()
        elif loss_config["type"] == "ce":
            target_loss = F.cross_entropy(
                student_logits,
                teacher_sequence,
                reduction="sum",
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_config['type']!r}.")
        sample_index = target_owners[target_row]
        losses_by_sample[sample_index].append(target_loss)
        total_tokens += teacher_sequence.numel()

        if debug_recorders is not None:
            recorder = debug_recorders[sample_index]
            if recorder is not None and recorder.enabled:
                debug_name = (
                    "student_prefill"
                    if target_counts[sample_index] == 1
                    else f"student_prefill_target_{target_indices[target_row]}"
                )
                recorder.record_json(debug_name, {
                    "query_length": int(layout.query_valid[target_row].sum().item()),
                    "context_length": int(layout.context_valid[target_row].sum().item()),
                    "target_rows": layout.target_rows[target_row],
                    "student_logits": student_logits,
                })

    sample_losses = [torch.stack(losses).sum() for losses in losses_by_sample]
    total_loss = torch.stack(sample_losses).sum()
    detached_losses = [float(loss.detach().item()) for loss in sample_losses]
    return total_loss, total_tokens, detached_losses
