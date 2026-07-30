"""Sample-major collection for one model/dataset/query pass partition."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from .profile_storage import (
    load_checkpoint,
    load_partition,
    make_checkpoint,
    save_checkpoint,
    save_partition,
)
from .profiles import make_partition, query_pass_partition_fingerprint
from .reducers import (
    AttentionProfileReducer,
    PicRetrievalReducer,
    RawAttentionProfileReducer,
)
from .spec import QueryPassSpec


REDUCER_FACTORIES = {
    "attention_profile": AttentionProfileReducer,
    "pic_retrieval": PicRetrievalReducer,
    "raw_attention_profile": RawAttentionProfileReducer,
}


def collect_query_pass_partition(
    *,
    partition_identity: dict[str, object],
    query_pass: QueryPassSpec,
    entries: Iterable[object],
    identify_sample: Callable[[int, object, QueryPassSpec], dict[str, object]],
    stream_method: Callable,
    layer_count: int,
    query_head_count: int,
    partition_path: str | Path,
    work_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Collect complete sample bundles; every method forward fans out to reducers."""
    output = Path(partition_path)
    work = Path(work_dir)
    if output.exists() and not overwrite:
        existing = load_partition(output)
        expected = query_pass_partition_fingerprint(
            partition_identity, layer_count, query_head_count
        )
        if existing["partition_fingerprint"] != expected:
            raise ValueError("Existing partition does not match the requested query pass.")
        return output
    work.mkdir(parents=True, exist_ok=True)

    methods = tuple(
        item["method_key"] for item in partition_identity["methods"]
    )
    samples: list[dict[str, object]] = []
    for sample_index, entry in enumerate(entries):
        checkpoint_path = work / f"sample_{sample_index:06d}.pt"
        if checkpoint_path.exists() and not overwrite:
            current_identity = identify_sample(sample_index, entry, query_pass)
            checkpoint = load_checkpoint(
                checkpoint_path,
                partition_identity=partition_identity,
                layer_count=layer_count,
                query_head_count=query_head_count,
                expected_index=sample_index,
            )
            saved = checkpoint["sample"]
            for field in (
                "sample_id", "canonical_token_digest", "query_target_digest"
            ):
                if saved[field] != current_identity[field]:
                    raise ValueError(
                        f"Checkpoint does not match the current dataset entry: {field}."
                    )
            current_chunks = [
                (chunk["chunk_id"], chunk["token_digest"], chunk["token_length"])
                for chunk in current_identity["chunks"]
            ]
            saved_chunks = [
                (chunk["chunk_id"], chunk["token_digest"], chunk["token_length"])
                for chunk in saved["chunks"]
            ]
            if saved_chunks != current_chunks:
                raise ValueError(
                    "Checkpoint does not match current ContextBlock boundaries."
                )
            samples.append(saved)
            continue

        identity = identify_sample(sample_index, entry, query_pass)
        reducers = [REDUCER_FACTORIES[key](methods) for key in query_pass.reducers]
        layouts: dict[str, dict[str, dict[str, int]]] = {}
        query_count: int | None = None
        canonical_chunks: dict[str, tuple[str, int]] = {}
        for method in methods:
            def consume_event(event) -> None:
                for reducer in reducers:
                    reducer.consume(method, event)

            meta = stream_method(
                method,
                entry,
                query_pass,
                consume_event,
            )
            if meta.layer_count != layer_count or meta.query_head_count != query_head_count:
                raise ValueError("Forward metadata does not match partition dimensions.")
            for field in (
                "sample_id", "canonical_token_digest", "query_target_digest"
            ):
                if getattr(meta, field) != identity[field]:
                    raise ValueError(f"Method forward changed sample identity: {field}.")
            if query_count is None:
                query_count = meta.query_count
            elif query_count != meta.query_count:
                raise ValueError("Methods emitted different query counts.")
            method_chunks = {}
            for chunk in meta.layout.chunks:
                current = chunk.token_digest, chunk.token_length
                previous = canonical_chunks.setdefault(chunk.chunk_id, current)
                if previous != current:
                    raise ValueError("Methods did not align the same canonical chunk.")
                layouts.setdefault(chunk.chunk_id, {})[method] = {
                    "pic_start": chunk.pic_start,
                    "pic_end": chunk.pic_end,
                    "scope_start": chunk.scope_start,
                    "scope_end": chunk.scope_end,
                }
                method_chunks[chunk.chunk_id] = current
            if set(method_chunks) != set(canonical_chunks):
                raise ValueError("Methods emitted different canonical chunks.")
            for reducer in reducers:
                reducer.finish_method(method)

        reducer_results = {
            reducer.reducer_key: reducer.finish_sample() for reducer in reducers
        }
        assert query_count is not None
        identified_chunks = [
            (chunk["chunk_id"], chunk["token_digest"], chunk["token_length"])
            for chunk in identity["chunks"]
        ]
        if identified_chunks != [
            (chunk_id, token_digest, token_length)
            for chunk_id, (token_digest, token_length) in canonical_chunks.items()
        ]:
            raise ValueError("Forward chunks do not match identified ContextBlocks.")
        chunks = []
        for chunk_id, (token_digest, token_length) in canonical_chunks.items():
            chunks.append({
                "chunk_id": chunk_id,
                "token_digest": token_digest,
                "token_length": token_length,
                "method_layouts": layouts[chunk_id],
                "reducer_outputs": {
                    key: value[chunk_id] for key, value in reducer_results.items()
                },
            })
        sample = {
            "sample_index": sample_index,
            "sample_id": identity["sample_id"],
            "canonical_token_digest": identity["canonical_token_digest"],
            "query_target_digest": identity["query_target_digest"],
            "query_count": query_count,
            "chunks": chunks,
        }
        checkpoint = make_checkpoint(
            partition_identity=partition_identity,
            layer_count=layer_count,
            query_head_count=query_head_count,
            sample=sample,
        )
        save_checkpoint(
            checkpoint_path,
            checkpoint,
            partition_identity=partition_identity,
            layer_count=layer_count,
            query_head_count=query_head_count,
            expected_index=sample_index,
        )
        samples.append(sample)

    if not samples:
        raise ValueError("Attention collection received no samples.")
    return save_partition(output, make_partition(
        partition_identity=partition_identity,
        layer_count=layer_count,
        query_head_count=query_head_count,
        samples=samples,
    ))


__all__ = ["REDUCER_FACTORIES", "collect_query_pass_partition"]
