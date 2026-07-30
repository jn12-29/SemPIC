"""Data contract and aggregation for the paper attention-sink evidence figure."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch


SCHEMA_NAME = "sempic.attention_sink_plot_data"
SCHEMA_VERSION = 2
EXPECTED_DATASETS = ("biography", "hotpot_qa", "musique", "niah")
REQUIRED_METHODS = ("full_recompute", "vanilla_pic", "sempic")
INTERIOR_ERROR_METHODS = ("vanilla_pic", "kvpacket", "sempic")
REQUIRED_MODEL_ID = "Qwen3-4B-Instruct-2507"
METHOD_ALIASES = {"no_recompute": "vanilla_pic"}
LEADING_END = 0.1
INTERIOR_END = 0.9


class IncompletePartitionError(ValueError):
    """Raised when a saved partition contains fewer samples than its frozen scope."""


def normalize_method_key(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def safe_ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None, "undefined_nonfinite_component"
    if denominator <= 0:
        return None, "undefined_nonpositive_denominator"
    return numerator / denominator, "defined"


def recovery_fraction(
    full_f1: float, vanilla_f1: float, sempic_f1: float
) -> tuple[float | None, str]:
    return safe_ratio(sempic_f1 - vanilla_f1, full_f1 - vanilla_f1)


def mean_sem(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Sample values must be a non-empty list of finite numbers.")
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sem": sem, "count": len(values)}


def membership_digest(sample_ids: list[str]) -> str:
    encoded = json.dumps(sample_ids, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _interval_layer_density(
    profile: torch.Tensor, start: float, end: float
) -> torch.Tensor:
    """Return one token-width-normalized density per layer for an interval."""

    if not 0 <= start < end <= 1:
        raise ValueError("Normalized interval must satisfy 0 <= start < end <= 1.")
    token_count = profile.shape[1]
    left = torch.arange(token_count, dtype=torch.float64) / token_count
    right = torch.arange(1, token_count + 1, dtype=torch.float64) / token_count
    overlap = (torch.minimum(right, torch.tensor(end)) - torch.maximum(
        left, torch.tensor(start)
    )).clamp_min(0)
    represented_width = overlap.sum()
    if represented_width <= 0:
        raise ValueError("Normalized interval contains no represented token width.")
    return (profile.double() * overlap).sum(dim=1) / represented_width


def _raw_profiles(chunk: dict[str, Any], methods: tuple[str, ...]) -> dict[str, torch.Tensor]:
    outputs = chunk.get("reducer_outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Chunk reducer_outputs must be an object.")
    raw_output = outputs.get("raw_attention_profile")
    if not isinstance(raw_output, dict) or tuple(raw_output) != methods:
        raise ValueError(
            "raw_attention_profile method keys must exact-match partition execution order."
        )
    token_length = chunk.get("token_length")
    profiles: dict[str, torch.Tensor] = {}
    for method in methods:
        record = raw_output.get(method)
        profile = record.get("raw") if isinstance(record, dict) else None
        if (
            type(profile) is not torch.Tensor
            or profile.dtype != torch.float32
            or profile.device.type != "cpu"
            or profile.ndim != 2
            or profile.shape[1] != token_length
            or not torch.isfinite(profile).all().item()
            or torch.any(profile < 0).item()
            or torch.any(profile > 1).item()
        ):
            raise ValueError(
                f"raw_attention_profile.{method}.raw must be a finite [0,1] "
                "CPU float32 [layer, token] tensor."
            )
        profiles[method] = profile
    return profiles


def aggregate_partition(
    partition: dict[str, Any],
    *,
    num_position_bins: int = 20,
    required_methods: tuple[str, ...] = REQUIRED_METHODS,
) -> dict[str, Any]:
    """Aggregate one shifted-prediction partition using the declared sink estimand."""

    if num_position_bins <= 0:
        raise ValueError("num_position_bins must be positive.")
    identity = partition.get("partition_identity")
    if not isinstance(identity, dict):
        raise ValueError("Partition identity is missing.")
    if identity.get("query_pass_id") != "shifted_prediction":
        raise ValueError("Sink evidence requires query_pass_id=shifted_prediction.")
    method_records = identity.get("methods")
    if not isinstance(method_records, list):
        raise ValueError("Partition methods are missing.")
    methods = tuple(record.get("method_key") for record in method_records)
    if any(not isinstance(method, str) for method in methods):
        raise ValueError("Partition method keys must be strings.")
    if not required_methods or any(
        not isinstance(method, str) or not method for method in required_methods
    ):
        raise ValueError("required_methods must contain non-empty method keys.")
    missing_methods = set(required_methods) - set(methods)
    if missing_methods:
        raise ValueError(f"Partition is missing required methods: {sorted(missing_methods)}")

    samples = partition.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Partition must contain at least one sample.")
    dataset_config = identity.get("dataset_config")
    configured_count = (
        dataset_config.get("num_samples") if isinstance(dataset_config, dict) else None
    )
    max_samples = identity.get("max_samples")
    if (
        isinstance(configured_count, bool)
        or not isinstance(configured_count, int)
        or configured_count <= 0
    ):
        raise ValueError("Partition dataset_config.num_samples must be positive.")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("Partition max_samples must be null or positive.")
    expected_count = (
        configured_count if max_samples is None else min(configured_count, max_samples)
    )
    if len(samples) != expected_count:
        raise IncompletePartitionError(
            "Incomplete partition: actual sample count "
            f"{len(samples)} does not equal frozen expected count {expected_count} "
            f"(dataset.num_samples={configured_count}, max_samples={max_samples})."
        )
    sample_ids: list[str] = []
    method_samples: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    intervals = [
        (index / num_position_bins, (index + 1) / num_position_bins)
        for index in range(num_position_bins)
    ]

    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Partition samples must be objects.")
        sample_id = sample.get("sample_id")
        chunks = sample.get("chunks")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Each sample must have a non-empty sample_id.")
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Each sample must contain at least one chunk.")
        sample_ids.append(sample_id)

        per_method_chunks: dict[str, list[dict[str, Any]]] = {
            method: [] for method in methods
        }
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise ValueError("Chunks must be objects.")
            profiles = _raw_profiles(chunk, methods)
            for method, profile in profiles.items():
                layer_count = partition.get("layer_count")
                if profile.shape[0] != layer_count:
                    raise ValueError("Raw profile layer count does not match partition.")
                per_method_chunks[method].append(
                    {
                        "bins": torch.stack(
                            [
                                _interval_layer_density(profile, start, end)
                                for start, end in intervals
                            ],
                            dim=1,
                        ),
                        "leading": _interval_layer_density(profile, 0.0, LEADING_END),
                        "interior": _interval_layer_density(
                            profile, LEADING_END, INTERIOR_END
                        ),
                    }
                )

        for method in methods:
            chunk_records = per_method_chunks[method]
            # [chunk, layer, bin] -> equal chunk -> equal layer.
            bin_values = torch.stack([record["bins"] for record in chunk_records]).mean(0).mean(0)
            leading = torch.stack(
                [record["leading"] for record in chunk_records]
            ).mean(0).mean()
            interior = torch.stack(
                [record["interior"] for record in chunk_records]
            ).mean(0).mean()
            method_samples[method].append(
                {
                    "sample_id": sample_id,
                    "bins": [float(value) for value in bin_values],
                    "leading": float(leading),
                    "interior": float(interior),
                }
            )

    method_summaries: dict[str, Any] = {}
    for method, sample_records in method_samples.items():
        positions = []
        for index, (start, end) in enumerate(intervals):
            summary = mean_sem([record["bins"][index] for record in sample_records])
            positions.append(
                {"bin_index": index, "start": start, "end": end, **summary}
            )
        regions = {
            region: mean_sem([record[region] for record in sample_records])
            for region in ("leading", "interior")
        }
        method_summaries[method] = {
            "sample_values": sample_records,
            "positions": positions,
            "regions": regions,
        }

    sempic_regions = method_summaries["sempic"]["regions"]
    sink_ratio, sink_status = safe_ratio(
        sempic_regions["leading"]["mean"], sempic_regions["interior"]["mean"]
    )
    return {
        "model_id": identity.get("model_id"),
        "dataset_id": identity.get("dataset_id"),
        "query_pass_id": identity.get("query_pass_id"),
        "seed": dataset_config.get("seed") if isinstance(dataset_config, dict) else None,
        "sample_count": len(sample_ids),
        "expected_sample_count": expected_count,
        "sample_ids": sample_ids,
        "sample_membership_digest": membership_digest(sample_ids),
        "methods": method_summaries,
        "sink_ratio": sink_ratio,
        "sink_ratio_status": sink_status,
        "region_rule": {
            "leading": [0.0, LEADING_END],
            "interior": [LEADING_END, INTERIOR_END],
        },
    }


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{location} must be finite.")
    return number


def validate_plot_data(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Plot data must be an object.")
    if value.get("schema_name") != SCHEMA_NAME or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Plot data must use {SCHEMA_NAME} version {SCHEMA_VERSION}.")
    if value.get("expected_identity_count") != len(EXPECTED_DATASETS):
        raise ValueError("Plot data expected_identity_count must be 4.")
    points = value.get("points")
    if not isinstance(points, list) or len(points) != len(EXPECTED_DATASETS):
        raise ValueError("Plot data must contain all four dataset identities.")
    if tuple(point.get("dataset_id") for point in points) != EXPECTED_DATASETS:
        raise ValueError("Plot data dataset order must match the fixed evidence matrix.")
    model_ids = {point.get("model_id") for point in points}
    if model_ids != {REQUIRED_MODEL_ID} or value.get("model_id") != REQUIRED_MODEL_ID:
        raise ValueError(f"Sink evidence is fixed to model_id={REQUIRED_MODEL_ID}.")
    for point in points:
        status = point.get("status")
        if status not in {"pass", "fail", "blocked", "not-run"}:
            raise ValueError("Each point must have an explicit evidence status.")
        if status != "pass":
            if not isinstance(point.get("status_reason"), str) or not point["status_reason"]:
                raise ValueError("Incomplete points require status_reason.")
            continue
        if point.get("query_pass_id") != "shifted_prediction":
            raise ValueError("Complete sink points require shifted_prediction.")
        behavior = point.get("behavior")
        profiles = point.get("profiles")
        regions = point.get("regions")
        interior_errors = point.get("interior_attention_errors")
        relative_errors = point.get("relative_interior_attention_errors")
        if not all(
            isinstance(items, list)
            for items in (behavior, profiles, regions, interior_errors, relative_errors)
        ):
            raise ValueError(
                "Complete points require behavior, profiles, regions, and interior-error lists."
            )
        if tuple(item.get("method_key") for item in behavior) != REQUIRED_METHODS:
            raise ValueError("Behavior rows must contain Full, Vanilla, and SemPIC in order.")
        if tuple(item.get("method_key") for item in profiles) != REQUIRED_METHODS:
            raise ValueError("Profiles must contain Full, Vanilla, and SemPIC in order.")
        expected_regions = tuple(
            (method, region)
            for method in REQUIRED_METHODS
            for region in ("leading", "interior")
        )
        if tuple(
            (item.get("method_key"), item.get("region")) for item in regions
        ) != expected_regions:
            raise ValueError("Regions must contain leading/interior for all methods in order.")
        sample_count = point.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("Complete points require a positive sample_count.")
        if tuple(item.get("method_key") for item in interior_errors) != INTERIOR_ERROR_METHODS:
            raise ValueError(
                "Interior attention errors must contain Vanilla, KV Packet, and SemPIC in order."
            )
        if tuple(item.get("method_key") for item in relative_errors) != (
            "kvpacket",
            "sempic",
        ):
            raise ValueError(
                "Relative interior attention errors must contain KV Packet and SemPIC in order."
            )
        raw_errors = {}
        raw_measurement_ids = set()
        for row in interior_errors:
            if set(row) != {"method_key", "mean", "sem", "count", "measurement_id"}:
                raise ValueError("Interior attention error rows have an invalid field set.")
            raw_errors[row["method_key"]] = row
            mean = _finite_number(row.get("mean"), "interior_attention_error.mean")
            if mean < 0:
                raise ValueError("Interior attention error mean must be nonnegative.")
            sem = _finite_number(row.get("sem"), "interior_attention_error.sem")
            if sem < 0:
                raise ValueError("Interior attention error SEM must be nonnegative.")
            count = row.get("count")
            if type(count) is not int or count != sample_count:
                raise ValueError("Interior attention error count must equal point sample_count.")
            if not isinstance(row.get("measurement_id"), str) or not row["measurement_id"]:
                raise ValueError("Each interior attention error requires a measurement ID.")
            if row["measurement_id"] in raw_measurement_ids:
                raise ValueError("Interior attention error measurement IDs must be unique.")
            raw_measurement_ids.add(row["measurement_id"])
        for row in relative_errors:
            if set(row) != {
                "method_key",
                "value",
                "status",
                "numerator_measurement_id",
                "denominator_measurement_id",
            }:
                raise ValueError(
                    "Relative interior attention error rows have an invalid field set."
                )
            method = row["method_key"]
            numerator = raw_errors[method]
            denominator = raw_errors["vanilla_pic"]
            expected, expected_status = safe_ratio(
                float(numerator["mean"]), float(denominator["mean"])
            )
            if row["status"] != expected_status:
                raise ValueError(
                    "Relative interior attention error status does not recompute."
                )
            if row["numerator_measurement_id"] != numerator["measurement_id"] or row[
                "denominator_measurement_id"
            ] != denominator["measurement_id"]:
                raise ValueError(
                    "Relative interior attention error provenance IDs do not match components."
                )
            actual = row["value"]
            if expected is None:
                if actual is not None:
                    raise ValueError("Undefined relative interior attention error must be null.")
            elif actual is None or not math.isclose(
                _finite_number(actual, "relative_interior_attention_error.value"),
                expected,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Relative interior attention error does not recompute from raw means."
                )
        for row in behavior:
            _finite_number(row.get("f1"), "behavior.f1")
            if not isinstance(row.get("measurement_id"), str) or not row["measurement_id"]:
                raise ValueError("Each behavior value requires a measurement ID.")
        bin_coordinates = None
        for profile in profiles:
            bins = profile.get("bins")
            if not isinstance(bins, list) or not bins:
                raise ValueError("Each profile requires normalized-position bins.")
            coordinates = []
            for item in bins:
                _finite_number(item.get("mean"), "profile.mean")
                _finite_number(item.get("sem"), "profile.sem")
                start = _finite_number(item.get("start"), "profile.start")
                end = _finite_number(item.get("end"), "profile.end")
                if not 0 <= start < end <= 1:
                    raise ValueError("Profile bins must be normalized intervals.")
                coordinates.append((start, end))
                if item.get("count") != sample_count:
                    raise ValueError("Profile count must equal point sample_count.")
                if not isinstance(item.get("measurement_id"), str) or not item["measurement_id"]:
                    raise ValueError("Each profile value requires a measurement ID.")
            if bin_coordinates is None:
                bin_coordinates = coordinates
            elif coordinates != bin_coordinates:
                raise ValueError("All methods must use identical normalized-position bins.")
        for row in regions:
            _finite_number(row.get("mean"), "region.mean")
            _finite_number(row.get("sem"), "region.sem")
            if row.get("count") != sample_count:
                raise ValueError("Region count must equal point sample_count.")
            if not isinstance(row.get("measurement_id"), str) or not row["measurement_id"]:
                raise ValueError("Each region value requires a measurement ID.")
        for name in ("recovery_fraction", "sink_ratio"):
            status_key = f"{name}_status"
            metric_status = point.get(status_key)
            metric = point.get(name)
            if metric_status == "defined":
                _finite_number(metric, name)
            elif metric is not None or not isinstance(metric_status, str) or not metric_status.startswith("undefined_"):
                raise ValueError(f"{name} must be null with an undefined status or finite and defined.")
            measurement_id = point.get(f"{name}_measurement_id")
            if not isinstance(measurement_id, str) or not measurement_id:
                raise ValueError(f"{name} requires a provenance measurement ID.")
        f1 = {row["method_key"]: float(row["f1"]) for row in behavior}
        expected_recovery, expected_recovery_status = recovery_fraction(
            f1["full_recompute"], f1["vanilla_pic"], f1["sempic"]
        )
        sempic_regions = {
            row["region"]: float(row["mean"])
            for row in regions
            if row["method_key"] == "sempic"
        }
        expected_sink, expected_sink_status = safe_ratio(
            sempic_regions["leading"], sempic_regions["interior"]
        )
        for name, expected, expected_status in (
            ("recovery_fraction", expected_recovery, expected_recovery_status),
            ("sink_ratio", expected_sink, expected_sink_status),
        ):
            if point[f"{name}_status"] != expected_status:
                raise ValueError(f"{name} status does not recompute from exported components.")
            actual = point[name]
            if expected is None:
                if actual is not None:
                    raise ValueError(f"{name} must be null for its exported components.")
            elif actual is None or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{name} does not recompute from exported components.")
        conditions = point.get("coexistence_conditions")
        expected_conditions = {
            "f1_sempic_greater_than_vanilla": f1["sempic"] > f1["vanilla_pic"],
            "sink_ratio_greater_than_one": (
                expected_sink > 1 if expected_sink is not None else None
            ),
        }
        if conditions != expected_conditions:
            raise ValueError("Coexistence conditions do not match exported F1 and S values.")
    return value


def load_plot_data(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {input_path}: {error}") from error
    try:
        return validate_plot_data(data)
    except ValueError as error:
        raise ValueError(f"{input_path}: {error}") from error
