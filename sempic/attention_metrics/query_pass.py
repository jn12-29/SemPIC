"""One real forward per method and query pass with layer-basis callbacks."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import gc
import hashlib
from itertools import islice
from typing import Callable, TypeAlias

import torch

from ..cache_comb import build_cache_comb_executor
from ..cache_comb.abc import PreparedKVMapping
from ..cache_comb.compact_prefill import (
    CompactPrefillExecutor,
    LayerPrefillObservation,
)
from ..evaluation import (
    EvalResourceCache,
    build_eval_generator,
    build_generation_config,
    create_eval_resource_cache,
    load_eval_resources,
    prepare_prompt_kvs,
    release_eval_resource_cache,
)
from ..prompt import TokenSpan, TokenizedPrompt, compile_prompt
from ..utils.lora import lora_adapters_disabled
from .answer import (
    stream_answer_attention,
    stream_shifted_prediction_attention,
    tokenize_gold_answer,
)
from .basis import LayerAttentionBasis, PhysicalChunkScope, PhysicalLayout
from .layout import project_physical_layout
from .profile_identity import normalize_method_key
from .spec import QueryPassSpec


LayerBasisSink: TypeAlias = Callable[[LayerAttentionBasis], None]


@dataclass(frozen=True, slots=True)
class ForwardPassShape:
    layer_count: int
    query_count: int
    query_head_count: int
    layout: PhysicalLayout


@dataclass(frozen=True, slots=True)
class QueryPassForwardMeta:
    sample_id: str
    canonical_token_digest: str
    query_target_digest: str
    chunks: tuple[PhysicalChunkScope, ...]
    layout: PhysicalLayout
    layer_count: int
    query_count: int
    query_head_count: int


def _tensor_digest(values: torch.Tensor) -> str:
    contiguous = values.detach().to(device="cpu", dtype=torch.long).contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _sample_id(entry, prompt: TokenizedPrompt, query_ids: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(entry["query"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(entry["answer"].encode("utf-8"))
    digest.update(b"\0")
    digest.update(_tensor_digest(prompt.input_ids).encode("ascii"))
    digest.update(b"\0")
    digest.update(_tensor_digest(query_ids).encode("ascii"))
    return digest.hexdigest()


def _query_ids(tokenizer, entry, query_pass: QueryPassSpec) -> torch.LongTensor:
    if query_pass.uses_gold_answer:
        return tokenize_gold_answer(tokenizer, entry["answer"])
    return torch.empty(0, dtype=torch.long)


@contextmanager
def _temporary_eager_attention(model):
    previous = getattr(model.config, "_attn_implementation", None)
    setter = getattr(model, "set_attn_implementation", None)
    if not callable(setter):
        raise ValueError("The loaded model cannot switch to eager attention.")
    setter("eager")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("The loaded model did not enable eager attention.")
    try:
        yield
    finally:
        if previous is not None:
            setter(previous)


def _all_inline_prompt(prompt: TokenizedPrompt) -> TokenizedPrompt:
    return TokenizedPrompt(
        input_ids=prompt.input_ids,
        parts=(TokenSpan("inline", 0, prompt.input_ids.numel()),),
    )


def _terminal_query_rows(
    prompt: TokenizedPrompt,
    executor: CompactPrefillExecutor,
    prepared_kvs: PreparedKVMapping,
) -> torch.LongTensor:
    terminal = prompt.parts[-1]
    layout = executor.build_layout(prompt, prepared_kvs)
    return torch.nonzero(
        (layout.inline_canonical_indices >= terminal.start)
        & (layout.inline_canonical_indices < terminal.end),
        as_tuple=False,
    ).flatten()


def _stream_compact_terminal_basis(
    *,
    method_name: str,
    executor: CompactPrefillExecutor,
    tokenizer,
    prompt: TokenizedPrompt,
    prepared_kvs: PreparedKVMapping,
    header_len: int,
    trailer_len: int,
    basis_sink: LayerBasisSink,
) -> ForwardPassShape:
    """Run one compact Inline prefill and stream only its terminal query rows."""
    forward_prompt = _all_inline_prompt(prompt) if method_name == "full_recompute" else prompt
    if method_name == "full_recompute":
        terminal = prompt.parts[-1]
        selected = torch.arange(terminal.start, terminal.end)
    else:
        selected = _terminal_query_rows(prompt, executor, prepared_kvs)
    query_count = int(selected.numel())
    query_head_count: int | None = None
    physical_layout: PhysicalLayout | None = None
    layer_count = 0

    def observe(value: LayerPrefillObservation) -> None:
        nonlocal layer_count, physical_layout, query_head_count
        projected = project_physical_layout(
            prompt=prompt,
            method_name=method_name,
            layout=value.layout,
            header_len=header_len,
            trailer_len=trailer_len,
        )
        if physical_layout is None:
            physical_layout = projected
        elif projected != physical_layout:
            raise ValueError("Physical layout changed across attention layers.")
        if value.layer_index != layer_count:
            raise ValueError("Attention basis callbacks must be complete and ordered.")
        if not torch.equal(value.selected_query_indices.cpu(), selected):
            raise ValueError("Compact prefill observed unexpected query rows.")
        expected = (
            1,
            value.probabilities.size(1),
            query_count,
            projected.physical_length,
        )
        if tuple(value.probabilities.shape) != expected:
            raise ValueError(
                f"Layer {value.layer_index} terminal attention shape mismatch: "
                f"expected {expected}, got {tuple(value.probabilities.shape)}."
            )
        if query_head_count is None:
            query_head_count = value.probabilities.size(1)
        elif value.probabilities.size(1) != query_head_count:
            raise ValueError("Attention query-head count changed across layers.")
        basis_sink(LayerAttentionBasis.from_compact_prefill(
            value=value,
            layout=projected,
        ))
        layer_count += 1

    executor.prefill(
        method_name=method_name,
        tokenizer=tokenizer,
        prompt=forward_prompt,
        prepared_kvs=prepared_kvs,
        observer=observe,
        selected_q_indices=selected,
    )
    if physical_layout is None or query_head_count is None:
        raise ValueError("Compact prefill emitted no attention observations.")
    return ForwardPassShape(
        layer_count=layer_count,
        query_count=query_count,
        query_head_count=query_head_count,
        layout=physical_layout,
    )


class QueryPassForwardProvider:
    """Own shared eval resources and execute one forward per method/query pass."""

    def __init__(
        self,
        *,
        configs: list[tuple[str, dict]],
        resource_cache: EvalResourceCache | None = None,
    ):
        if not configs:
            raise ValueError("Real attention collection requires method configs.")
        self.configs = {}
        for source_config, config in configs:
            method = normalize_method_key(config["cache_comb"]["method"])
            if method in self.configs:
                raise ValueError(f"Duplicate real-forward method config: {method}.")
            self.configs[method] = (source_config, config)
        if "full_recompute" not in self.configs:
            raise ValueError("Real attention collection requires Full Recompute.")
        model_configs = [config["model"] for _, config in self.configs.values()]
        dataset_configs = [config["dataset"] for _, config in self.configs.values()]
        if any(value != model_configs[0] for value in model_configs[1:]):
            raise ValueError("All methods in a partition must share one model config.")
        if any(value != dataset_configs[0] for value in dataset_configs[1:]):
            raise ValueError("All methods in a partition must share one dataset config.")
        if any(
            config["compress"] is not None or config["quantization"] is not None
            for _, config in self.configs.values()
        ):
            raise ValueError(
                "Attention query passes require one physical key per canonical token."
            )

        self._owns_resource_cache = resource_cache is None
        self.resource_cache = resource_cache or create_eval_resource_cache()
        self._closed = False
        try:
            self.resources = {
                method: load_eval_resources(config, self.resource_cache)
                for method, (_, config) in self.configs.items()
            }
        except Exception:
            if self._owns_resource_cache:
                release_eval_resource_cache(self.resource_cache)
            raise
        models = [resources.model for resources in self.resources.values()]
        tokenizers = [resources.tokenizer for resources in self.resources.values()]
        if any(model is not models[0] for model in models[1:]):
            raise ValueError("Attention methods did not reuse one base model instance.")
        if any(tokenizer is not tokenizers[0] for tokenizer in tokenizers[1:]):
            raise ValueError("Attention methods did not reuse one tokenizer instance.")
        self.model = models[0]
        self.tokenizer = tokenizers[0]
        self.model.eval()
        self._terminal_executor = CompactPrefillExecutor(
            self.model, backend="observable_eager"
        )

    @property
    def layer_count(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def query_head_count(self) -> int:
        return int(self.model.config.num_attention_heads)

    def entries(self, max_samples: int | None = None):
        full_config = self.configs["full_recompute"][1]
        entries = build_eval_generator(full_config["dataset"], self.tokenizer)
        return islice(entries, max_samples) if max_samples is not None else entries

    def identify_sample(
        self, sample_index: int, entry, query_pass: QueryPassSpec
    ) -> dict[str, object]:
        prompt = compile_prompt(self.tokenizer, entry["prompt"])
        query_ids = _query_ids(self.tokenizer, entry, query_pass)
        chunks = [
            {
                "chunk_id": f"part-{index}:{span.start}:{span.end}",
                "token_digest": _tensor_digest(prompt.input_ids[span.start:span.end]),
                "token_length": span.end - span.start,
            }
            for index, span in enumerate(prompt.parts[:-1])
            if span.kind == "context"
        ]
        if not chunks:
            raise ValueError("Attention query passes require at least one ContextBlock.")
        return {
            "sample_index": sample_index,
            "sample_id": _sample_id(entry, prompt, query_ids),
            "canonical_token_digest": _tensor_digest(prompt.input_ids),
            "query_target_digest": _tensor_digest(query_ids),
            "chunks": chunks,
        }

    def stream_method(
        self,
        method: str,
        entry,
        query_pass: QueryPassSpec,
        basis_sink: LayerBasisSink,
    ) -> QueryPassForwardMeta:
        try:
            _, config = self.configs[method]
            resources = self.resources[method]
        except KeyError as exc:
            raise ValueError(f"Unknown configured attention method: {method}.") from exc

        prompt = compile_prompt(resources.tokenizer, entry["prompt"]).to(
            resources.model.device
        )
        raw_method = config["cache_comb"]["method"]
        query_ids = _query_ids(resources.tokenizer, entry, query_pass)
        wrapper = resources.packet_wrapper
        callback_count = 0
        callback_layout: PhysicalLayout | None = None

        def consume_basis(event: LayerAttentionBasis) -> None:
            nonlocal callback_count, callback_layout
            if event.layer_index != callback_count:
                raise ValueError("Attention basis callbacks must be complete and ordered.")
            if callback_layout is None:
                callback_layout = event.layout
            elif event.layout != callback_layout:
                raise ValueError("Physical layout changed across attention layers.")
            callback_count += 1
            basis_sink(event)

        def adapter_context():
            return (
                lora_adapters_disabled(resources.model)
                if resources.lora_adapter_name is not None
                else nullcontext()
            )

        if query_pass.kind == "terminal_inline_tokens":
            if raw_method == "full_recompute":
                prepared_kvs = {}
            else:
                prepared_kvs = prepare_prompt_kvs(
                    model=resources.model,
                    tokenizer=resources.tokenizer,
                    prompt=prompt,
                    method_name=raw_method,
                    packet_wrapper=wrapper,
                    lora_adapter_name=resources.lora_adapter_name,
                ).prepared_kvs
            with adapter_context(), torch.no_grad():
                shape = _stream_compact_terminal_basis(
                    method_name=raw_method,
                    executor=self._terminal_executor,
                    tokenizer=resources.tokenizer,
                    prompt=prompt,
                    prepared_kvs=prepared_kvs,
                    header_len=0 if wrapper is None else wrapper.header_len,
                    trailer_len=0 if wrapper is None else wrapper.trailer_len,
                    basis_sink=consume_basis,
                )
        else:
            prepared_kvs = prepare_prompt_kvs(
                model=resources.model,
                tokenizer=resources.tokenizer,
                prompt=prompt,
                method_name=raw_method,
                packet_wrapper=wrapper,
                lora_adapter_name=resources.lora_adapter_name,
            ).prepared_kvs
            with adapter_context(), torch.no_grad():
                call_kwargs = dict(
                    tokenizer=resources.tokenizer,
                    generation_config=build_generation_config(config["model"]),
                    prompt=prompt,
                    prepared_kvs=prepared_kvs,
                    answer=entry["answer"],
                    kwargs=config["cache_comb"]["kwargs"],
                )
                if raw_method == "full_recompute":
                    prefill = build_cache_comb_executor(
                        raw_method, resources.model
                    )(model=resources.model, **call_kwargs)
                else:
                    prefill = self._terminal_executor.prefill(
                        method_name=raw_method, **call_kwargs
                    )
            layout_prompt = (
                _all_inline_prompt(prompt)
                if raw_method == "full_recompute"
                else prompt
            )
            shared_layout = self._terminal_executor.build_layout(
                layout_prompt, prepared_kvs
            )
            prefix_layout = project_physical_layout(
                prompt=prompt,
                method_name=raw_method,
                layout=shared_layout,
                header_len=0 if wrapper is None else wrapper.header_len,
                trailer_len=0 if wrapper is None else wrapper.trailer_len,
            )
            streamers = {
                "gold_answer_literal_tokens": stream_answer_attention,
                "gold_answer_shifted_prediction_queries": (
                    stream_shifted_prediction_attention
                ),
            }
            try:
                streamer = streamers[query_pass.kind]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported attention query pass: {query_pass.kind!r}."
                ) from exc
            with _temporary_eager_attention(resources.model), adapter_context(), torch.no_grad():
                answer_meta = streamer(
                    model=resources.model,
                    prefill=prefill,
                    answer_ids=query_ids,
                    basis_sink=consume_basis,
                    layout=prefix_layout,
                )
            extra_keys = (
                answer_meta.answer_len
                if query_pass.kind == "gold_answer_literal_tokens"
                else answer_meta.answer_len - 1
            )
            shape = ForwardPassShape(
                layer_count=answer_meta.num_layers,
                query_count=answer_meta.answer_len,
                query_head_count=answer_meta.num_query_heads,
                layout=PhysicalLayout(
                    physical_length=answer_meta.prefix_physical_len + extra_keys,
                    chunks=prefix_layout.chunks,
                ),
            )

        if callback_count != shape.layer_count or callback_layout != shape.layout:
            raise ValueError("Query pass did not emit exactly one basis per model layer.")

        return QueryPassForwardMeta(
            sample_id=_sample_id(entry, prompt, query_ids),
            canonical_token_digest=_tensor_digest(prompt.input_ids),
            query_target_digest=_tensor_digest(query_ids),
            chunks=shape.layout.chunks,
            layout=shape.layout,
            layer_count=shape.layer_count,
            query_count=shape.query_count,
            query_head_count=shape.query_head_count,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.resources.clear()
        self._terminal_executor = None
        self.model = None
        self.tokenizer = None
        if self._owns_resource_cache:
            release_eval_resource_cache(self.resource_cache)
        gc.collect()


__all__ = [
    "ForwardPassShape",
    "LayerBasisSink",
    "QueryPassForwardMeta",
    "QueryPassForwardProvider",
]
