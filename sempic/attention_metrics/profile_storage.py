"""Weights-only atomic I/O for query-pass statistics and checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

import torch

from .profile_identity import non_negative_int, strict_dict
from .profiles import (
    PROFILE_SCHEMA_VERSION,
    query_pass_partition_fingerprint,
    validate_partition,
    validate_query_pass_identity,
    validate_sample_record,
)


CHECKPOINT_SCHEMA_NAME = "sempic.attention_query_pass_checkpoint"
_CHECKPOINT_FIELDS = {
    "schema_name",
    "schema_version",
    "partition_fingerprint",
    "layer_count",
    "query_head_count",
    "sample",
}
_CHECKPOINT_NAME = re.compile(r"^sample_(\d{6})\.pt$")


def make_checkpoint(
    *,
    partition_identity: dict[str, object],
    layer_count: int,
    query_head_count: int,
    sample: dict[str, object],
) -> dict[str, object]:
    identity = validate_query_pass_identity(partition_identity)
    layer_count, query_head_count = _validate_dimensions(
        layer_count, query_head_count
    )
    methods = _method_keys(identity)
    validate_sample_record(
        sample,
        methods=methods,
        layer_count=layer_count,
        query_head_count=query_head_count,
        reducers=tuple(identity["query_spec"]["reducers"]),
    )
    return {
        "schema_name": CHECKPOINT_SCHEMA_NAME,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "partition_fingerprint": query_pass_partition_fingerprint(
            identity, layer_count, query_head_count
        ),
        "layer_count": layer_count,
        "query_head_count": query_head_count,
        "sample": sample,
    }


def validate_checkpoint(
    value: object,
    *,
    partition_identity: dict[str, object],
    layer_count: int,
    query_head_count: int,
    expected_index: int | None = None,
) -> dict[str, object]:
    identity = validate_query_pass_identity(partition_identity)
    expected_dimensions = _validate_dimensions(layer_count, query_head_count)
    checkpoint = strict_dict(value, _CHECKPOINT_FIELDS, "query-pass checkpoint")
    if checkpoint["schema_name"] != CHECKPOINT_SCHEMA_NAME:
        raise ValueError("Unsupported attention query-pass checkpoint schema.")
    if checkpoint["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unsupported attention query-pass checkpoint version.")
    checkpoint_dimensions = _validate_dimensions(
        checkpoint["layer_count"], checkpoint["query_head_count"]
    )
    if checkpoint_dimensions != expected_dimensions:
        raise ValueError("Checkpoint dimensions do not match the partition.")
    layer_count, query_head_count = checkpoint_dimensions
    if checkpoint["partition_fingerprint"] != query_pass_partition_fingerprint(
        identity, layer_count, query_head_count
    ):
        raise ValueError("Checkpoint partition_fingerprint mismatch.")
    validate_sample_record(
        checkpoint["sample"],
        methods=_method_keys(identity),
        layer_count=layer_count,
        query_head_count=query_head_count,
        expected_index=expected_index,
        reducers=tuple(identity["query_spec"]["reducers"]),
    )
    return checkpoint


def _method_keys(identity: dict[str, object]) -> tuple[str, ...]:
    return tuple(method["method_key"] for method in identity["methods"])


def _validate_dimensions(layer_count: object, query_head_count: object) -> tuple[int, int]:
    layer_count = non_negative_int(layer_count, "layer_count")
    query_head_count = non_negative_int(query_head_count, "query_head_count")
    if layer_count == 0 or query_head_count == 0:
        raise ValueError("layer_count and query_head_count must be positive.")
    return layer_count, query_head_count


def _atomic_torch_save(path: str | Path, value: object) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save(value, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def save_partition(path: str | Path, artifact: dict[str, object]) -> Path:
    return _atomic_torch_save(path, validate_partition(artifact))


def load_partition(path: str | Path) -> dict[str, object]:
    return validate_partition(
        torch.load(Path(path), map_location="cpu", weights_only=True)
    )


def save_checkpoint(
    path: str | Path,
    checkpoint: dict[str, object],
    *,
    partition_identity: dict[str, object],
    layer_count: int,
    query_head_count: int,
    expected_index: int,
) -> Path:
    _validate_checkpoint_name(path, expected_index)
    validator = lambda value: validate_checkpoint(
        value,
        partition_identity=partition_identity,
        layer_count=layer_count,
        query_head_count=query_head_count,
        expected_index=expected_index,
    )
    return _atomic_torch_save(path, validator(checkpoint))


def load_checkpoint(
    path: str | Path,
    *,
    partition_identity: dict[str, object],
    layer_count: int,
    query_head_count: int,
    expected_index: int,
) -> dict[str, object]:
    _validate_checkpoint_name(path, expected_index)
    return validate_checkpoint(
        torch.load(Path(path), map_location="cpu", weights_only=True),
        partition_identity=partition_identity,
        layer_count=layer_count,
        query_head_count=query_head_count,
        expected_index=expected_index,
    )


def _validate_checkpoint_name(path: str | Path, expected_index: int) -> None:
    match = _CHECKPOINT_NAME.fullmatch(Path(path).name)
    if match is None or int(match.group(1)) != expected_index:
        raise ValueError("Checkpoint filename must match its sample index.")


__all__ = [
    "CHECKPOINT_SCHEMA_NAME",
    "load_checkpoint",
    "load_partition",
    "make_checkpoint",
    "save_checkpoint",
    "save_partition",
    "validate_checkpoint",
]
