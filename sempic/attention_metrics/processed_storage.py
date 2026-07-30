"""Strict validation and atomic storage for flat processed metric records."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import tempfile

import torch

from .processing import normalize_processing_config
from .profile_identity import (
    fingerprint,
    normalize_method_key,
    sanitized_id,
    strict_dict,
)


_ARTIFACT_FIELDS = {
    "processing_config",
    "processing_fingerprint",
    "source_partitions",
    "metric_specs",
    "records",
}
_SOURCE_FIELDS = {
    "partition_fingerprint",
    "model_id",
    "dataset_id",
    "query_pass_id",
}
_RECORD_FIELDS = {
    "model_id",
    "dataset_id",
    "query_pass_id",
    "metric_key",
    "view_key",
    "method_key",
    "facets",
    "axes",
    "coordinates",
    "mean",
    "sem",
    "count",
}
_METRIC_SPEC_FIELDS = {"label", "value_label", "axis_policy"}
_VIEW_AXES = {
    "layer_position_heatmap": ("layer", "position_bin"),
    "layer_head_heatmap": ("layer", "query_head"),
    "layer_curve": ("layer",),
    "global_bar": (),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_processed_metrics(value: object) -> dict[str, object]:
    """Validate the complete current processed artifact."""
    artifact = strict_dict(value, _ARTIFACT_FIELDS, "processed metrics")
    config = normalize_processing_config(artifact["processing_config"])
    if artifact["processing_config"] != config:
        raise ValueError("processing_config is not canonical.")
    if artifact["processing_fingerprint"] != fingerprint(config):
        raise ValueError("processing_fingerprint mismatch.")

    sources = artifact["source_partitions"]
    metric_specs = artifact["metric_specs"]
    records = artifact["records"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_partitions must be a non-empty list.")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list.")
    if not isinstance(metric_specs, dict) or not metric_specs:
        raise ValueError("metric_specs must be a non-empty dictionary.")
    for metric_key, spec_value in metric_specs.items():
        if not isinstance(metric_key, str) or not _KEY.fullmatch(metric_key):
            raise ValueError("metric_specs keys must be lower-snake-case.")
        spec = strict_dict(spec_value, _METRIC_SPEC_FIELDS, "metric spec")
        if any(
            not isinstance(spec[field], str) or not spec[field]
            for field in ("label", "value_label")
        ) or spec["axis_policy"] != "nonnegative_auto":
            raise ValueError("Metric spec metadata is invalid.")

    source_identities: list[tuple[str, str, str]] = []
    source_fingerprints: list[str] = []
    for source_value in sources:
        source = strict_dict(source_value, _SOURCE_FIELDS, "source partition")
        identity = _validate_identity(source, "source partition")
        source_identities.append(identity)
        source_fingerprint = source["partition_fingerprint"]
        if not isinstance(source_fingerprint, str) or not _SHA256.fullmatch(
            source_fingerprint
        ):
            raise ValueError("Source partition fingerprint must be lowercase SHA-256.")
        source_fingerprints.append(source_fingerprint)
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("source_partitions must have unique identities.")
    if len(set(source_fingerprints)) != len(source_fingerprints):
        raise ValueError("source_partitions must have unique fingerprints.")

    source_identity_set = set(source_identities)
    record_keys = set()
    covered_sources = set()
    for record_value in records:
        record = strict_dict(record_value, _RECORD_FIELDS, "processed metric record")
        identity = _validate_identity(record, "processed metric record")
        if identity not in source_identity_set:
            raise ValueError("Metric record identity has no source partition.")
        covered_sources.add(identity)
        record_key = _validate_record(record, metric_specs)
        if record_key in record_keys:
            raise ValueError("Processed metric records must have unique identities.")
        record_keys.add(record_key)
    if covered_sources != source_identity_set:
        raise ValueError("Every source partition must contribute a metric record.")
    if {record["metric_key"] for record in records} != set(metric_specs):
        raise ValueError("metric_specs must exactly describe the emitted records.")
    return artifact


def _validate_identity(
    value: dict[str, object], name: str
) -> tuple[str, str, str]:
    fields = ("model_id", "dataset_id", "query_pass_id")
    for field in fields:
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{name}.{field} must be a non-empty string.")
    if sanitized_id(value["model_id"]) != value["model_id"]:
        raise ValueError(f"{name}.model_id must be a canonical path ID.")
    return tuple(value[field] for field in fields)


def _validate_record(
    record: dict[str, object], metric_specs: dict[str, object]
) -> tuple[object, ...]:
    for field in ("metric_key", "view_key"):
        item = record[field]
        if not isinstance(item, str) or not _KEY.fullmatch(item):
            raise ValueError(f"record.{field} must be a lower-snake-case key.")
    if record["metric_key"] not in metric_specs:
        raise ValueError("record.metric_key has no metric spec.")
    method = record["method_key"]
    if not isinstance(method, str) or normalize_method_key(method) != method:
        raise ValueError("record.method_key must be a canonical method key.")

    facets = record["facets"]
    if not isinstance(facets, dict) or any(
        not isinstance(key, str)
        or not _KEY.fullmatch(key)
        or not _is_facet_value(item)
        for key, item in facets.items()
    ):
        raise ValueError("record.facets must map lower-snake-case keys to scalars.")

    axes = record["axes"]
    expected_axes = _VIEW_AXES.get(record["view_key"])
    if expected_axes is None:
        raise ValueError("Unsupported processed metric view.")
    if not isinstance(axes, list) or tuple(axes) != expected_axes:
        raise ValueError("record.axes do not match record.view_key.")

    coordinates = record["coordinates"]
    if not isinstance(coordinates, dict) or list(coordinates) != axes:
        raise ValueError("record.coordinates must exactly follow record.axes.")
    shape = []
    for axis in axes:
        values = coordinates[axis]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Coordinates for {axis} must be a non-empty list.")
        if any(not _is_coordinate_value(item) for item in values):
            raise ValueError(f"Coordinates for {axis} contain invalid values.")
        if len({_coordinate_key(item) for item in values}) != len(values):
            raise ValueError(f"Coordinates for {axis} must be unique.")
        shape.append(len(values))

    _validate_estimate(record, tuple(shape))
    facet_key = tuple(sorted((key, _coordinate_key(item)) for key, item in facets.items()))
    return (
        record["model_id"],
        record["dataset_id"],
        record["query_pass_id"],
        record["metric_key"],
        record["view_key"],
        record["method_key"],
        facet_key,
    )


def _validate_estimate(record: dict[str, object], shape: tuple[int, ...]) -> None:
    for field in ("mean", "sem"):
        tensor = record[field]
        if type(tensor) is not torch.Tensor or tensor.dtype != torch.float64:
            raise ValueError(f"record.{field} must be a float64 tensor.")
        if tuple(tensor.shape) != shape or torch.isinf(tensor).any().item():
            raise ValueError(f"record.{field} has invalid shape or values.")
    count = record["count"]
    if (
        type(count) is not torch.Tensor
        or count.dtype != torch.int64
        or tuple(count.shape) != shape
        or (count < 0).any().item()
    ):
        raise ValueError("record.count must be a non-negative int64 tensor.")
    empty = count == 0
    if not torch.equal(torch.isnan(record["mean"]), empty) or not torch.equal(
        torch.isnan(record["sem"]), empty
    ):
        raise ValueError("Empty cells must be NaN and only empty cells may be NaN.")
    if not torch.all(record["sem"][count == 1] == 0).item():
        raise ValueError("Single-sample cells must have zero SEM.")
    if not torch.all(record["sem"][count > 0] >= 0).item():
        raise ValueError("Observed SEM must be non-negative.")
    if not torch.all(record["mean"][count > 0] >= 0).item():
        raise ValueError("nonnegative_auto metrics cannot have negative means.")


def _is_facet_value(value: object) -> bool:
    return value is None or isinstance(value, (str, bool, int, float)) and not (
        isinstance(value, float) and not math.isfinite(value)
    )


def _is_coordinate_value(value: object) -> bool:
    return (
        isinstance(value, (str, int, float))
        and not isinstance(value, bool)
        and not (isinstance(value, float) and not math.isfinite(value))
    )


def _coordinate_key(value: object) -> tuple[str, object]:
    return type(value).__name__, value


def save_processed_metrics(path: str | Path, artifact: dict[str, object]) -> Path:
    validated = validate_processed_metrics(artifact)
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
        torch.save(validated, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def load_processed_metrics(path: str | Path) -> dict[str, object]:
    return validate_processed_metrics(
        torch.load(Path(path), map_location="cpu", weights_only=True)
    )


__all__ = [
    "load_processed_metrics",
    "save_processed_metrics",
    "validate_processed_metrics",
]
