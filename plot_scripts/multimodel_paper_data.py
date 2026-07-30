"""Contract for the three-model paper evidence figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCHEMA_NAME = "sempic.paper_multimodel_evidence"
SCHEMA_VERSION = 1
FULL_SAMPLE_COUNT = 100
MODEL_ORDER = (
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
)
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
F1_FIELDS = (
    "full",
    "no_cache",
    "no_recompute",
    "kvpacket",
    "sempic",
    "joint",
)
POINT_FIELDS = {
    "dataset_id",
    "f1",
    "kv_recovery",
    "sempic_recovery",
    "f1_change",
    "kv_pre_ratio",
    "kv_interior_ratio",
    "kv_rint",
    "sempic_rint",
    "sink_profile",
    "sink_ratio",
    "source_ids",
}


def _finite(value: Any, location: str, *, nonnegative: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric.")
    number = float(value)
    if not math.isfinite(number) or nonnegative and number < 0:
        raise ValueError(f"{location} must be finite and nonnegative.")
    return number


def validate_plot_data(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Plot data must be an object.")
    if (
        value.get("schema_name") != SCHEMA_NAME
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(f"Plot data must use {SCHEMA_NAME} version 1.")
    models = value.get("models")
    if not isinstance(models, list) or tuple(
        model.get("model_id") for model in models if isinstance(model, dict)
    ) != MODEL_ORDER:
        raise ValueError("Plot data must contain the fixed three-model order.")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Plot data must contain provenance sources.")
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each provenance source must be an object.")
        source_id = source.get("source_id")
        path = source.get("path")
        sha256 = source.get("sha256")
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in source_ids
            or not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("Provenance source identity/path/hash is invalid.")
        source_ids.add(source_id)

    for model in models:
        if not isinstance(model.get("display_name"), str) or not model["display_name"]:
            raise ValueError("Each model requires display_name.")
        points = model.get("points")
        if not isinstance(points, list) or tuple(
            point.get("dataset_id") for point in points if isinstance(point, dict)
        ) != DATASET_ORDER:
            raise ValueError("Each model must contain all four datasets in order.")
        for point in points:
            if set(point) != POINT_FIELDS:
                raise ValueError("Evidence point fields do not match the schema.")
            f1 = point["f1"]
            if not isinstance(f1, dict) or tuple(f1) != F1_FIELDS:
                raise ValueError("F1 fields do not match the fixed method order.")
            for field in F1_FIELDS:
                number = _finite(f1[field], f"f1.{field}")
                if number > 1:
                    raise ValueError("F1 values must be in [0,1].")
            denominator = float(f1["full"]) - float(f1["no_recompute"])
            if denominator <= 0:
                raise ValueError("Recovery requires Full F1 > No Recompute F1.")
            expected_derived = {
                "kv_recovery": (
                    float(f1["kvpacket"]) - float(f1["no_recompute"])
                )
                / denominator,
                "sempic_recovery": (
                    float(f1["sempic"]) - float(f1["no_recompute"])
                )
                / denominator,
                "f1_change": float(f1["sempic"]) - float(f1["no_recompute"]),
            }
            for field in (
                "kv_recovery",
                "sempic_recovery",
                "f1_change",
                "kv_pre_ratio",
                "kv_interior_ratio",
                "kv_rint",
                "sempic_rint",
                "sink_ratio",
            ):
                _finite(point[field], field)
            for field, expected in expected_derived.items():
                if not math.isclose(
                    float(point[field]), expected, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"{field} does not recompute from authoritative F1.")
            if not math.isclose(
                float(point["kv_interior_ratio"]),
                float(point["kv_rint"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("kv_interior_ratio and kv_rint must be identical.")
            profile = point["sink_profile"]
            if not isinstance(profile, list) or not profile:
                raise ValueError("sink_profile must contain normalized-position bins.")
            prior_end = 0.0
            for index, item in enumerate(profile):
                if not isinstance(item, dict) or set(item) != {
                    "start",
                    "end",
                    "mean",
                    "sem",
                    "count",
                }:
                    raise ValueError("Sink profile bins have an invalid field set.")
                start = _finite(item["start"], "sink_profile.start")
                end = _finite(item["end"], "sink_profile.end")
                _finite(item["mean"], "sink_profile.mean")
                _finite(item["sem"], "sink_profile.sem")
                if (
                    not math.isclose(start, prior_end, abs_tol=1e-12)
                    or not start < end <= 1
                    or type(item["count"]) is not int
                    or item["count"] != FULL_SAMPLE_COUNT
                ):
                    raise ValueError(f"Invalid sink profile bin {index}.")
                prior_end = end
            if not math.isclose(prior_end, 1.0, abs_tol=1e-12):
                raise ValueError("Sink profile bins must cover [0,1].")
            region_densities = {}
            for region, region_start, region_end in (
                ("pre", 0.0, 0.1),
                ("interior", 0.1, 0.9),
            ):
                weighted_sum = 0.0
                represented_width = 0.0
                for item in profile:
                    overlap = max(
                        0.0,
                        min(float(item["end"]), region_end)
                        - max(float(item["start"]), region_start),
                    )
                    weighted_sum += overlap * float(item["mean"])
                    represented_width += overlap
                if represented_width <= 0:
                    raise ValueError(f"sink_profile does not cover the {region} region.")
                region_densities[region] = weighted_sum / represented_width
            if region_densities["interior"] <= 0 or not math.isclose(
                float(point["sink_ratio"]),
                region_densities["pre"] / region_densities["interior"],
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError("sink_ratio does not recompute from sink_profile.")
            point_sources = point["source_ids"]
            if not isinstance(point_sources, dict) or set(point_sources) != {
                "f1",
                "boundary_attention",
                "interior_attention",
                "sink_attention",
            } or any(source_id not in source_ids for source_id in point_sources.values()):
                raise ValueError("Point source_ids do not resolve to provenance sources.")
            if point_sources["f1"] != "f1_authority":
                raise ValueError("Paper F1 values must resolve to f1_authority.")
    return value


def load_plot_data(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load plot data {source}: {error}") from error
    return validate_plot_data(value)
