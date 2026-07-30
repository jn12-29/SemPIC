"""Strict schema for one query-pass attention statistics partition."""

from __future__ import annotations

from typing import Any

import torch

from .profile_identity import (
    fingerprint,
    non_empty_string,
    non_negative_int,
    strict_dict,
    validate_query_pass_identity,
)


PARTITION_SCHEMA_NAME = "sempic.attention_query_pass_statistics"
PROFILE_SCHEMA_VERSION = 1

_PARTITION_FIELDS = {
    "schema_name",
    "schema_version",
    "partition_identity",
    "partition_fingerprint",
    "layer_count",
    "query_head_count",
    "samples",
}
_SAMPLE_FIELDS = {
    "sample_index",
    "sample_id",
    "canonical_token_digest",
    "query_target_digest",
    "query_count",
    "chunks",
}
_CHUNK_FIELDS = {
    "chunk_id",
    "token_digest",
    "token_length",
    "method_layouts",
    "reducer_outputs",
}
_LAYOUT_FIELDS = {"pic_start", "pic_end", "scope_start", "scope_end"}
_FULL_ATTENTION_FIELDS = {"raw", "chunk_conditional"}
_CANDIDATE_ATTENTION_FIELDS = {
    "raw_absolute_error",
    "chunk_conditional_absolute_error",
}
_RAW_ATTENTION_FIELDS = {"raw"}
_FULL_RETRIEVAL_FIELDS = {
    "query_count",
    "reference_energy_sum",
    "scope_mass_sum",
}
_CANDIDATE_RETRIEVAL_FIELDS = {
    "query_count",
    "squared_error_sum",
    "reference_energy_sum",
    "cosine_distance_sum",
    "cosine_valid_count",
    "absolute_mass_error_sum",
    "full_scope_mass_sum",
    "candidate_scope_mass_sum",
}


def query_pass_partition_fingerprint(
    identity: dict[str, object], layer_count: int, query_head_count: int
) -> str:
    return fingerprint({
        "partition_identity": identity,
        "layer_count": layer_count,
        "query_head_count": query_head_count,
    })


def validate_sample_record(
    value: object,
    *,
    methods: tuple[str, ...],
    layer_count: int,
    query_head_count: int,
    expected_index: int | None = None,
    reducers: tuple[str, ...] = ("attention_profile", "pic_retrieval"),
) -> dict[str, object]:
    """Validate one complete sample/query-pass reducer bundle."""

    sample = strict_dict(value, _SAMPLE_FIELDS, "sample")
    sample_index = non_negative_int(sample["sample_index"], "sample_index")
    if expected_index is not None and sample_index != expected_index:
        raise ValueError("samples must preserve contiguous sample_index order.")
    for field in ("sample_id", "canonical_token_digest", "query_target_digest"):
        non_empty_string(sample[field], field)
    query_count = non_negative_int(sample["query_count"], "query_count")
    if query_count == 0:
        raise ValueError("query_count must be positive.")

    chunks = sample["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("sample.chunks must be a non-empty list.")
    chunk_ids: set[str] = set()
    for chunk_index, raw_chunk in enumerate(chunks):
        chunk = strict_dict(raw_chunk, _CHUNK_FIELDS, f"chunks[{chunk_index}]")
        chunk_id = non_empty_string(chunk["chunk_id"], "chunk_id")
        if chunk_id in chunk_ids:
            raise ValueError("Chunk IDs must be unique within a sample.")
        chunk_ids.add(chunk_id)
        non_empty_string(chunk["token_digest"], "token_digest")
        token_length = non_negative_int(chunk["token_length"], "token_length")
        if token_length == 0:
            raise ValueError("token_length must be positive.")
        _validate_layouts(chunk["method_layouts"], methods, token_length)
        _validate_reducer_outputs(
            chunk["reducer_outputs"],
            methods=methods,
            layer_count=layer_count,
            query_head_count=query_head_count,
            token_length=token_length,
            query_count=query_count,
            reducers=reducers,
        )
    return sample


def _validate_layouts(
    value: object, methods: tuple[str, ...], token_length: int
) -> None:
    layouts = _exact_method_dict(value, methods, "method_layouts")
    for method, raw_layout in layouts.items():
        layout = strict_dict(raw_layout, _LAYOUT_FIELDS, f"method_layouts.{method}")
        pic_start = non_negative_int(layout["pic_start"], "pic_start")
        pic_end = non_negative_int(layout["pic_end"], "pic_end")
        scope_start = non_negative_int(layout["scope_start"], "scope_start")
        scope_end = non_negative_int(layout["scope_end"], "scope_end")
        if pic_end - pic_start != token_length:
            raise ValueError("PIC layout length must equal the canonical token_length.")
        if not (scope_start <= pic_start < pic_end <= scope_end):
            raise ValueError("Retrieval scope must contain the complete canonical PIC.")
        if method not in {"kvpacket", "sempic_kvpacket"} and (
            scope_start != pic_start or scope_end != pic_end
        ):
            raise ValueError("Wrapper-free method retrieval scope must equal its PIC.")
        if method in {"kvpacket", "sempic_kvpacket"} and not (
            scope_start < pic_start and pic_end < scope_end
        ):
            raise ValueError("Wrapper method retrieval scope must include head and tail.")


def _validate_reducer_outputs(
    value: object,
    *,
    methods: tuple[str, ...],
    layer_count: int,
    query_head_count: int,
    token_length: int,
    query_count: int,
    reducers: tuple[str, ...],
) -> None:
    outputs = strict_dict(value, set(reducers), "reducer_outputs")
    profile_shape = (layer_count, token_length)
    head_shape = (layer_count, query_head_count)
    if "attention_profile" in outputs:
        attention = _exact_method_dict(
            outputs["attention_profile"], methods, "attention_profile"
        )
        for method in methods:
            fields = _FULL_ATTENTION_FIELDS if method == "full_recompute" else _CANDIDATE_ATTENTION_FIELDS
            record = strict_dict(attention[method], fields, f"attention_profile.{method}")
            for name, tensor in record.items():
                _validate_float_tensor(tensor, profile_shape, f"{method}.{name}", maximum=1.0)
    if "raw_attention_profile" in outputs:
        raw_attention = _exact_method_dict(
            outputs["raw_attention_profile"], methods, "raw_attention_profile"
        )
        for method in methods:
            record = strict_dict(
                raw_attention[method],
                _RAW_ATTENTION_FIELDS,
                f"raw_attention_profile.{method}",
            )
            _validate_float_tensor(
                record["raw"], profile_shape, f"{method}.raw", maximum=1.0
            )
    if "pic_retrieval" in outputs:
        retrieval = _exact_method_dict(outputs["pic_retrieval"], methods, "pic_retrieval")
        for method in methods:
            fields = _FULL_RETRIEVAL_FIELDS if method == "full_recompute" else _CANDIDATE_RETRIEVAL_FIELDS
            record = strict_dict(retrieval[method], fields, f"pic_retrieval.{method}")
            reducer_query_count = non_negative_int(record["query_count"], f"pic_retrieval.{method}.query_count")
            if reducer_query_count != query_count:
                raise ValueError("Reducer query_count must equal sample.query_count.")
            for name, tensor in record.items():
                if name == "query_count":
                    continue
                if name == "cosine_valid_count":
                    _validate_count_tensor(tensor, head_shape, query_count, name)
                else:
                    maximum = float(query_count) if "mass" in name else (float(2 * query_count) if name == "cosine_distance_sum" else None)
                    _validate_float_tensor(tensor, head_shape, f"{method}.{name}", maximum=maximum)


def _exact_method_dict(
    value: object, methods: tuple[str, ...], name: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or tuple(value) != methods:
        raise ValueError(f"{name} keys must exact-match methods in execution order.")
    return value


def _validate_float_tensor(
    value: object,
    shape: tuple[int, int],
    name: str,
    *,
    maximum: float | None = None,
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or tuple(value.shape) != shape
        or not torch.isfinite(value).all().item()
        or torch.any(value < 0).item()
        or (maximum is not None and torch.any(value > maximum).item())
    ):
        range_description = (
            "non-negative" if maximum is None else f"within [0, {maximum}]"
        )
        raise ValueError(
            f"{name} must be a finite {range_description} CPU float32 tensor "
            f"of shape {shape}."
        )
    return value


def _validate_count_tensor(
    value: object, shape: tuple[int, int], query_count: int, name: str
) -> torch.Tensor:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.int64
        or value.device.type != "cpu"
        or tuple(value.shape) != shape
        or torch.any(value < 0).item()
        or torch.any(value > query_count).item()
    ):
        raise ValueError(
            f"{name} must be a CPU int64 tensor of shape {shape} within query_count."
        )
    return value


def make_partition(
    *,
    partition_identity: dict[str, object],
    layer_count: int,
    query_head_count: int,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    identity = validate_query_pass_identity(partition_identity)
    artifact = {
        "schema_name": PARTITION_SCHEMA_NAME,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "partition_identity": identity,
        "partition_fingerprint": query_pass_partition_fingerprint(
            identity, layer_count, query_head_count
        ),
        "layer_count": layer_count,
        "query_head_count": query_head_count,
        "samples": samples,
    }
    return validate_partition(artifact)


def validate_partition(value: object) -> dict[str, object]:
    artifact = strict_dict(value, _PARTITION_FIELDS, "statistics partition")
    if artifact["schema_name"] != PARTITION_SCHEMA_NAME:
        raise ValueError("Unsupported attention query-pass statistics schema.")
    if artifact["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unsupported attention query-pass statistics version.")
    identity = validate_query_pass_identity(artifact["partition_identity"])
    layer_count = non_negative_int(artifact["layer_count"], "layer_count")
    query_head_count = non_negative_int(
        artifact["query_head_count"], "query_head_count"
    )
    if layer_count == 0 or query_head_count == 0:
        raise ValueError("layer_count and query_head_count must be positive.")
    if artifact["partition_fingerprint"] != query_pass_partition_fingerprint(
        identity, layer_count, query_head_count
    ):
        raise ValueError("partition_fingerprint mismatch.")
    methods = tuple(method["method_key"] for method in identity["methods"])
    reducers = tuple(identity["query_spec"]["reducers"])

    samples = artifact["samples"]
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a non-empty list.")
    max_samples = identity["max_samples"]
    if max_samples is not None and len(samples) > max_samples:
        raise ValueError("Partition contains more samples than max_samples.")
    sample_ids: set[str] = set()
    for index, sample in enumerate(samples):
        validated = validate_sample_record(
            sample,
            methods=methods,
            layer_count=layer_count,
            query_head_count=query_head_count,
            expected_index=index,
            reducers=reducers,
        )
        sample_id = validated["sample_id"]
        if sample_id in sample_ids:
            raise ValueError("Sample IDs must be unique.")
        sample_ids.add(sample_id)
    return artifact


__all__ = [
    "PARTITION_SCHEMA_NAME",
    "PROFILE_SCHEMA_VERSION",
    "make_partition",
    "query_pass_partition_fingerprint",
    "validate_partition",
    "validate_query_pass_identity",
    "validate_sample_record",
]
