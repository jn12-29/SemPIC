from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
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

from ..cache import KVCache, KeyValue
from ..cache.rotate import rerotate_embeddings, rerotate_kv_flops
from ..model import SupportedModel
from ..prefill.layout import (
    InterleavedPrefillLayout,
    PrefillSegment,
    build_interleaved_layout,
)
from ..prompt import TokenizedPrompt
from .abc import PrefillResult, PreparedKVMapping, TokenizerType
from .utils.flops import AutoFlopsCalculator


BackendName = Literal["flex", "observable_eager"]


def _terminal_span(prompt: TokenizedPrompt):
    if not prompt.parts or prompt.parts[-1].kind != "inline":
        raise ValueError("TokenizedPrompt must end with an Inline span.")
    terminal = prompt.parts[-1]
    if terminal.start == terminal.end:
        raise ValueError("TokenizedPrompt must end with a non-empty Inline span.")
    return terminal


def _require_exact_context_kvs(
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
) -> None:
    expected = {
        part_index
        for part_index, span in enumerate(prompt.parts)
        if span.kind == "context"
    }
    actual = set(prepared_kvs)
    if actual != expected:
        raise ValueError(
            "Prepared ContextBlock KV mapping mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )


def _request_attention_shape(
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
) -> tuple[int, int]:
    _require_exact_context_kvs(prompt, prepared_kvs)
    _terminal_span(prompt)
    inline_length = 0
    physical_length = 0
    for part_index, span in enumerate(prompt.parts):
        span_length = span.end - span.start
        if span_length <= 0:
            raise ValueError("Compact prefill requires non-empty prompt spans.")
        if span.kind == "inline":
            inline_length += span_length
            physical_length += span_length
            continue
        cache = prepared_kvs[part_index]
        if not cache.layers:
            raise ValueError(f"Prepared ContextBlock {part_index} contains no layers.")
        context_length = cache.get_seq_length(cache.layers[0])
        if context_length == 0:
            raise ValueError(
                f"Prepared ContextBlock {part_index} must not be empty."
            )
        physical_length += context_length
    return inline_length, physical_length


def _reject_unexpected_method_kwargs(method_name: object, kwargs: object) -> None:
    if not isinstance(method_name, str):
        raise TypeError("Compact prefill requires method_name.")
    if kwargs:
        raise ValueError(f"{method_name} compact prefill got unexpected kwargs: {kwargs}")


@dataclass(frozen=True, slots=True)
class LayerPrefillObservation:
    layer_index: int
    selected_query_indices: torch.Tensor
    scaled_masked_logits: torch.Tensor
    probabilities: torch.Tensor
    keep_mask: torch.Tensor
    values: torch.Tensor
    query_to_kv_head: torch.Tensor
    layout: InterleavedPrefillLayout


class PrefillObserver(Protocol):
    def __call__(self, observation: LayerPrefillObservation) -> None: ...


def _unwrap_causal_lm(model: SupportedModel) -> LlamaForCausalLM | Qwen3ForCausalLM:
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if isinstance(base_model, (LlamaForCausalLM, Qwen3ForCausalLM)):
        return base_model
    raise TypeError(f"Unsupported compact prefill model type: {type(base_model).__name__}.")


def _cache_views(prepared_kvs: PreparedKVMapping) -> dict[int, KVCache]:
    # Consolidation, when needed, happens only in these shallow views. The prepared
    # cache objects and all of their tensors remain untouched.
    return {part_index: cache.copy() for part_index, cache in prepared_kvs.items()}


def _build_eval_layout(
    model: SupportedModel,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    cache_views: Mapping[int, KVCache],
) -> InterleavedPrefillLayout:
    _require_exact_context_kvs(prompt, prepared_kvs)
    context_indices = [
        part_index
        for part_index, span in enumerate(prompt.parts)
        if span.kind == "context"
    ]
    extents: list[int] = []
    if context_indices:
        maxima = []
        for part_index in context_indices:
            cache = cache_views[part_index]
            if not cache.layers:
                raise ValueError(f"Prepared ContextBlock {part_index} contains no layers.")
            positions = cache[cache.layers[0]].position_ids
            if positions.size(0) != 1 or positions.size(1) == 0:
                raise ValueError(
                    "Compact prefill requires a non-empty one-row position map for "
                    f"ContextBlock {part_index}; got {tuple(positions.shape)}."
                )
            maxima.append(positions.max())
        # One synchronization covers every ContextBlock extent. Do not introduce a
        # per-document .item() in this request path.
        extents = (torch.stack(maxima) + 1).tolist()

    terminal = _terminal_span(prompt)
    next_position = 0
    context_cursor = 0
    segments: list[PrefillSegment] = []
    for part_index, span in enumerate(prompt.parts):
        if span.kind == "context":
            cache = cache_views[part_index]
            old_positions = cache[cache.layers[0]].position_ids.squeeze(0)
            positions = old_positions + next_position
            segments.append(PrefillSegment(
                kind="context",
                position_ids=positions,
                canonical_start=span.start,
                canonical_end=span.end,
                part_index=part_index,
            ))
            source_length = span.end - span.start
            next_position += max(source_length, extents[context_cursor])
            context_cursor += 1
        else:
            length = span.end - span.start
            positions = torch.arange(
                next_position,
                next_position + length,
                dtype=torch.long,
                device=model.device,
            )
            segments.append(PrefillSegment(
                kind="inline",
                position_ids=positions,
                canonical_start=span.start,
                canonical_end=span.end,
                terminal=span is terminal,
            ))
            next_position += length
    return build_interleaved_layout(segments)


def _pack_context(
    causal_lm: LlamaForCausalLM | Qwen3ForCausalLM,
    layout: InterleavedPrefillLayout,
    cache_views: Mapping[int, KVCache],
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    if not layout.context_placements:
        return None, None, 0

    body = causal_lm.model
    num_layers = len(body.layers)
    key_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    old_position_chunks: list[torch.Tensor] = []
    flops = 0
    nope_dim = getattr(causal_lm.config, "qk_nope_head_dim", None)
    expected_layers = list(range(num_layers))
    for placement in layout.context_placements:
        cache = cache_views[placement.part_index]
        if cache.layers != expected_layers:
            raise ValueError(
                f"ContextBlock {placement.part_index} layers {cache.layers} do not "
                f"match model layers {expected_layers}."
            )
        layer_values = [cache[layer_index] for layer_index in expected_layers]
        reference_positions = layer_values[0].position_ids
        if any(
            value.position_ids.shape != reference_positions.shape
            for value in layer_values[1:]
        ):
            raise ValueError(
                f"ContextBlock {placement.part_index} position ID shapes differ across layers."
            )
        key_chunks.append(torch.stack([value.key[0] for value in layer_values]))
        value_chunks.append(torch.stack([value.value[0] for value in layer_values]))
        old_position_chunks.append(reference_positions)
        flops += rerotate_kv_flops(cache, nope_dim=nope_dim)

    packed_key = torch.cat(key_chunks, dim=2)
    packed_value = torch.cat(value_chunks, dim=2)
    old_positions = torch.cat(old_position_chunks, dim=1)
    new_positions = layout.context_position_ids.unsqueeze(0)
    if packed_key.size(2) != layout.context_length:
        raise ValueError("Packed Context K/V does not match the shared prefill layout.")
    rerotated_key = rerotate_embeddings(
        packed_key,
        body.rotary_emb,
        old_positions,
        new_positions,
        nope_dim=nope_dim,
    )
    return rerotated_key, packed_value, flops


def _dense_frontier_attention(
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
        del batch_index, head_index
        return torch.where(
            key_index <= frontiers[query_index],
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


class _FlexBackend:
    def __init__(self, causal_lm: LlamaForCausalLM | Qwen3ForCausalLM):
        self.device = causal_lm.device
        self.allow_compile = False
        self._attention = _dense_frontier_attention
        if self.device.type == "cuda":
            self._attention = torch.compile(_dense_frontier_attention, dynamic=True)

    def attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        frontiers: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        if self.device.type != "cuda":
            query_heads = query.size(1)
            kv_heads = key.size(1)
            if query_heads % kv_heads:
                raise ValueError("Query heads must be divisible by KV heads for GQA.")
            query_to_kv_head = (
                torch.arange(query_heads, device=query.device) // (query_heads // kv_heads)
            )
            expanded_key = key.index_select(1, query_to_kv_head)
            expanded_value = value.index_select(1, query_to_kv_head)
            scores = torch.matmul(query, expanded_key.transpose(2, 3)) * scale
            key_indices = torch.arange(key.size(2), device=query.device)
            keep = key_indices.view(1, 1, 1, -1) <= frontiers.view(1, 1, -1, 1)
            probabilities = torch.softmax(
                scores.masked_fill(~keep, torch.finfo(scores.dtype).min),
                dim=-1,
                dtype=torch.float32,
            ).to(query.dtype)
            return torch.matmul(probabilities, expanded_value).transpose(1, 2).contiguous()
        if self.allow_compile:
            attended = self._attention(query, key, value, frontiers, scale)
        else:
            with torch.compiler.set_stance("fail_on_recompile"):
                attended = self._attention(query, key, value, frontiers, scale)
        return attended.transpose(1, 2).contiguous()

    @torch.no_grad()
    def warm_shape(
        self,
        query_length: int,
        key_length: int,
        causal_lm: LlamaForCausalLM | Qwen3ForCausalLM,
    ) -> None:
        if self.device.type != "cuda":
            return
        config = causal_lm.config
        query_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", config.hidden_size // query_heads))
        dtype = next(causal_lm.parameters()).dtype
        query = torch.empty_strided(
            (1, query_heads, query_length, head_dim),
            (
                query_length * query_heads * head_dim,
                head_dim,
                query_heads * head_dim,
                1,
            ),
            dtype=dtype,
            device=self.device,
        ).zero_()
        key = torch.zeros(
            (1, kv_heads, key_length, head_dim),
            dtype=dtype,
            device=self.device,
        )
        value = torch.zeros_like(key)
        frontiers = torch.full(
            (query_length,),
            key_length - 1,
            dtype=torch.long,
            device=self.device,
        )
        self.allow_compile = True
        try:
            self._attention(
                query,
                key,
                value,
                frontiers,
                float(causal_lm.model.layers[0].self_attn.scaling),
            )
            torch.cuda.synchronize(self.device)
        finally:
            self.allow_compile = False


def _observable_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frontiers: torch.Tensor,
    scale: float,
    layer_index: int,
    selected_query_indices: torch.Tensor,
    layout: InterleavedPrefillLayout,
    observer: PrefillObserver | None,
) -> torch.Tensor:
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads for GQA.")
    groups = query_heads // kv_heads
    query_to_kv_head = torch.arange(query_heads, device=query.device) // groups
    expanded_key = key.index_select(1, query_to_kv_head)
    expanded_value = value.index_select(1, query_to_kv_head)
    logits = torch.matmul(query, expanded_key.transpose(2, 3)) * scale
    key_indices = torch.arange(key.size(2), device=query.device)
    keep_mask = key_indices.view(1, 1, 1, -1) <= frontiers.view(1, 1, -1, 1)
    masked_logits = logits.masked_fill(~keep_mask, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked_logits, dim=-1, dtype=torch.float32).to(query.dtype)
    attended = torch.matmul(probabilities, expanded_value).transpose(1, 2).contiguous()

    if observer is not None:
        observer(LayerPrefillObservation(
            layer_index=layer_index,
            selected_query_indices=selected_query_indices,
            scaled_masked_logits=masked_logits.index_select(2, selected_query_indices),
            probabilities=probabilities.index_select(2, selected_query_indices),
            keep_mask=keep_mask.index_select(2, selected_query_indices),
            values=value,
            query_to_kv_head=query_to_kv_head,
            layout=layout,
        ))
    return attended


class CompactPrefillExecutor:
    def __init__(self, model: SupportedModel, backend: BackendName = "flex"):
        self.model = model
        self.causal_lm = _unwrap_causal_lm(model)
        self.backend = backend
        self._flex = _FlexBackend(self.causal_lm) if backend == "flex" else None
        if backend not in ("flex", "observable_eager"):
            raise ValueError(f"Unsupported compact prefill backend: {backend}")
        for layer_index, decoder_layer in enumerate(self.causal_lm.model.layers):
            attention_type = getattr(decoder_layer, "attention_type", "full_attention")
            if attention_type != "full_attention":
                raise NotImplementedError(
                    "Compact prefill does not yet support sliding attention; "
                    f"layer {layer_index} uses {attention_type!r}."
                )

    def build_layout(
        self,
        prompt: TokenizedPrompt,
        prepared_kvs: PreparedKVMapping,
    ) -> InterleavedPrefillLayout:
        cache_views = _cache_views(prepared_kvs)
        return _build_eval_layout(self.model, prompt, prepared_kvs, cache_views)

    @torch.no_grad()
    def warm_request(self, **call_kwargs: object) -> None:
        _reject_unexpected_method_kwargs(
            call_kwargs.get("method_name"),
            call_kwargs.get("kwargs"),
        )
        if self._flex is None or self.model.device.type != "cuda":
            return
        prompt = call_kwargs.get("prompt")
        prepared_kvs = call_kwargs.get("prepared_kvs")
        if not isinstance(prompt, TokenizedPrompt) or not isinstance(prepared_kvs, Mapping):
            raise TypeError("Compact prefill warmup requires prompt and prepared_kvs.")
        query_length, key_length = _request_attention_shape(prompt, prepared_kvs)
        self._flex.warm_shape(query_length, key_length, self.causal_lm)

    @torch.no_grad()
    def prefill(
        self,
        *,
        method_name: str,
        model: SupportedModel | None = None,
        tokenizer: TokenizerType,
        prompt: TokenizedPrompt,
        prepared_kvs: PreparedKVMapping,
        generation_config: object | None = None,
        answer: str = "",
        answer_postprocess_func: Callable[[str, str], tuple[str, str]] | None = None,
        kwargs: dict | None = None,
        observer: PrefillObserver | None = None,
        selected_q_indices: torch.Tensor | None = None,
    ) -> PrefillResult:
        if model is not None and model is not self.model:
            raise ValueError("Compact prefill executor was called with a different model.")
        del generation_config, answer, answer_postprocess_func
        _reject_unexpected_method_kwargs(method_name, kwargs)
        if self.backend == "flex" and observer is not None:
            raise ValueError("The FlexAttention backend does not support attention observation.")
        if observer is not None and selected_q_indices is None:
            raise ValueError("Attention observation requires selected_q_indices.")

        cache_views = _cache_views(prepared_kvs)
        layout = _build_eval_layout(self.model, prompt, prepared_kvs, cache_views)
        inline_ids = prompt.input_ids.to(self.model.device).index_select(
            0, layout.inline_canonical_indices
        ).unsqueeze(0)
        hidden_states = self.causal_lm.model.embed_tokens(inline_ids)
        position_ids = layout.inline_position_ids.unsqueeze(0)
        position_embeddings = self.causal_lm.model.rotary_emb(hidden_states, position_ids)
        context_key, context_value, flops = _pack_context(
            self.causal_lm, layout, cache_views
        )

        selected = selected_q_indices
        if selected is None:
            selected = torch.empty((0,), dtype=torch.long, device=self.model.device)
        else:
            selected = selected.to(device=self.model.device, dtype=torch.long)

        physical_source_indices = layout.physical_source_indices.to(self.model.device)
        complete_cache = KVCache()
        for layer_index, decoder_layer in enumerate(self.causal_lm.model.layers):
            residual = hidden_states
            normalized = decoder_layer.input_layernorm(hidden_states)
            attention = decoder_layer.self_attn
            input_shape = normalized.shape[:-1]
            projection_shape = (*input_shape, -1, attention.head_dim)
            query = attention.q_proj(normalized).view(projection_shape).transpose(1, 2)
            key = attention.k_proj(normalized).view(projection_shape).transpose(1, 2)
            value = attention.v_proj(normalized).view(projection_shape).transpose(1, 2)
            if isinstance(decoder_layer, Qwen3DecoderLayer):
                query = attention.q_norm(query.transpose(1, 2)).transpose(1, 2)
                key = attention.k_norm(key.transpose(1, 2)).transpose(1, 2)
                query, key = qwen3_apply_rotary_pos_emb(query, key, *position_embeddings)
            elif isinstance(decoder_layer, LlamaDecoderLayer):
                query, key = llama_apply_rotary_pos_emb(query, key, *position_embeddings)
            else:
                raise TypeError(
                    f"Unsupported decoder layer type: {type(decoder_layer).__name__}."
                )

            if context_key is None:
                compact_key, compact_value = key, value
            else:
                assert context_value is not None
                compact_key = torch.cat((context_key[layer_index].unsqueeze(0), key), dim=2)
                compact_value = torch.cat((context_value[layer_index].unsqueeze(0), value), dim=2)
            physical_key = compact_key.index_select(2, physical_source_indices)
            physical_value = compact_value.index_select(2, physical_source_indices)

            if self.backend == "flex":
                assert self._flex is not None
                attended = self._flex.attend(
                    query,
                    physical_key,
                    physical_value,
                    layout.inline_physical_frontiers,
                    attention.scaling,
                )
            else:
                attended = _observable_attention(
                    query=query,
                    key=physical_key,
                    value=physical_value,
                    frontiers=layout.inline_physical_frontiers,
                    scale=attention.scaling,
                    layer_index=layer_index,
                    selected_query_indices=selected,
                    layout=layout,
                    observer=observer,
                )
            hidden_states = residual + attention.o_proj(attended.reshape(*input_shape, -1))
            residual = hidden_states
            hidden_states = residual + decoder_layer.mlp(
                decoder_layer.post_attention_layernorm(hidden_states)
            )
            complete_cache.update(layer_index, KeyValue(
                key=physical_key,
                value=physical_value,
                position_ids=layout.physical_position_ids.unsqueeze(0),
            ))

        terminal_hidden = hidden_states[:, layout.terminal_inline_row : layout.terminal_inline_row + 1]
        logits = self.causal_lm.lm_head(self.causal_lm.model.norm(terminal_hidden))
        calculator = AutoFlopsCalculator(self.causal_lm)
        flops += calculator.total_flops(
            batch_size=1,
            seq_len=layout.inline_length,
            cache_len=layout.context_length,
        )
        flops += calculator.output_flops(batch_size=1, hidden_rows=1, logits_rows=1)

        terminal = _terminal_span(prompt)
        terminal_ids = prompt.input_ids[terminal.start : terminal.end].unsqueeze(0).to(self.model.device)
        prefix_length = layout.physical_length - terminal_ids.size(1)
        dummy_id = 1 if tokenizer.pad_token_id == 0 else 0
        generation_input_ids = torch.cat((
            torch.full(
                (1, prefix_length),
                dummy_id,
                dtype=torch.long,
                device=self.model.device,
            ),
            terminal_ids,
        ), dim=1)
        if generation_input_ids.size(1) != layout.physical_length:
            raise ValueError("Generation input and compact prefill cache lengths differ.")
        return PrefillResult(
            logits=logits,
            past_key_values=complete_cache.to_hf_cache(config=self.causal_lm.config),
            generation_input_ids=generation_input_ids,
            position_ids=layout.inline_position_ids[
                layout.terminal_inline_row : layout.terminal_inline_row + 1
            ].view(1, 1),
            attention_mask=torch.ones_like(generation_input_ids),
            flops=flops,
        )


__all__ = [
    "CompactPrefillExecutor",
    "LayerPrefillObservation",
    "PrefillObserver",
]
