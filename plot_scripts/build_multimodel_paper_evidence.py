"""Build the fixed three-model evidence bundle for the paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from sempic.attention_metrics.processed_storage import load_processed_metrics
from sempic.attention_metrics.profile_storage import load_partition
from sempic.utils.run_storage import atomic_write_json

try:
    from plot_scripts.apply_paper_f1_authority import load_authority
    from plot_scripts.attention_sink_data import (
        aggregate_partition,
        load_plot_data as load_qwen4_attention_plot,
    )
    from plot_scripts.multimodel_paper_data import (
        DATASET_ORDER,
        MODEL_ORDER,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        validate_plot_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from apply_paper_f1_authority import load_authority
    from attention_sink_data import (
        aggregate_partition,
        load_plot_data as load_qwen4_attention_plot,
    )
    from multimodel_paper_data import (
        DATASET_ORDER,
        MODEL_ORDER,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        validate_plot_data,
    )


DISPLAY_NAMES = {
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
}
F1_MAP = {
    "full": "full_f1",
    "no_cache": "no_cache_f1",
    "no_recompute": "no_recompute_f1",
    "kvpacket": "kvpacket_f1",
    "sempic": "sempic_f1",
    "joint": "joint_f1",
}
TARGET_FACETS = {
    "attention_view": "raw",
    "edge_ratio": "0.1",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(source_id: str, path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"Evidence source does not exist: {path}")
    return {
        "source_id": source_id,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def _parse_assignments(values: Iterable[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        dataset_id, separator, path_text = value.partition("=")
        if separator != "=" or dataset_id not in DATASET_ORDER or not path_text:
            raise ValueError(f"{option} requires DATASET=PATH for {DATASET_ORDER}.")
        if dataset_id in result:
            raise ValueError(f"Duplicate {option} dataset: {dataset_id}")
        result[dataset_id] = Path(path_text)
    missing = set(DATASET_ORDER) - set(result)
    if missing:
        raise ValueError(f"{option} is missing datasets: {sorted(missing)}")
    return result


def _recovery(full: float, no_recompute: float, method: float) -> float:
    denominator = full - no_recompute
    if denominator <= 0:
        raise ValueError("F1 recovery requires Full F1 > No Recompute F1.")
    return (method - no_recompute) / denominator


def _f1_point(authority: Any) -> dict[str, Any]:
    f1 = {target: float(authority.values[source]) for target, source in F1_MAP.items()}
    return {
        "f1": f1,
        "kv_recovery": _recovery(f1["full"], f1["no_recompute"], f1["kvpacket"]),
        "sempic_recovery": _recovery(f1["full"], f1["no_recompute"], f1["sempic"]),
        "f1_change": f1["sempic"] - f1["no_recompute"],
    }


def _load_boundary_csv(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    result: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "pass":
                continue
            count_fields = (
                "prefix_vanilla_count",
                "prefix_kvpacket_count",
                "interior_vanilla_count",
                "interior_kvpacket_count",
            )
            if any(int(row[field]) != 100 for field in count_fields):
                raise ValueError(f"Boundary evidence is not full100: {row}")
            key = (row["model_id"], row["dataset_id"])
            result[key] = {
                "pre": float(row["prefix_attention_ratio"]),
                "interior": float(row["interior_attention_ratio"]),
            }
    return result


def _load_qwen4_plot(path: Path) -> dict[str, dict[str, Any]]:
    value = load_qwen4_attention_plot(path)
    points: dict[str, dict[str, Any]] = {}
    for point in value.get("points", []):
        if point.get("status") != "pass":
            continue
        if point.get("sample_count") != 100:
            raise ValueError("Qwen3-4B attention evidence is not full100.")
        ratios = {
            item["method_key"]: float(item["value"])
            for item in point["relative_interior_attention_errors"]
        }
        sempic_profile = next(
            profile for profile in point["profiles"] if profile["method_key"] == "sempic"
        )
        points[point["dataset_id"]] = {
            "kv_rint": ratios["kvpacket"],
            "sempic_rint": ratios["sempic"],
            "sink_profile": _profile_bins(sempic_profile["bins"]),
            "sink_ratio": float(point["sink_ratio"]),
        }
    return points


def _profile_bins(bins: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "start": float(item["start"]),
            "end": float(item["end"]),
            "mean": float(item["mean"]),
            "sem": float(item["sem"]),
            "count": int(item["count"]),
        }
        for item in bins
    ]


def _extract_summary_ratios(path: Path) -> dict[str, dict[str, float]]:
    means: dict[tuple[str, str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["model"] != "Qwen3-8B"
                or row["query_pass"] != "shifted_prediction"
                or row["metric"] != "attention_absolute_deviation"
                or row["view"] != "global_bar"
                or row["layer"]
                or row["position_bin"]
                or row["query_head"]
            ):
                continue
            facets = json.loads(row["facets"])
            if facets.get("attention_view") != "raw" or facets.get("edge_ratio") != "0.1":
                continue
            region = facets.get("region")
            if region not in {"prefix", "interior"}:
                continue
            if int(row["count"]) != 100:
                raise ValueError("Qwen3-8B attention evidence is not full100.")
            means[(row["dataset"], row["method"], region)] = float(row["mean"])
    return _ratios_from_means(means, "Qwen3-8B summary")


def _extract_processed_ratios(path: Path, model_id: str, dataset_id: str) -> dict[str, float]:
    payload = load_processed_metrics(path)
    means: dict[tuple[str, str, str], float] = {}
    for record in payload["records"]:
        facets = record["facets"]
        if (
            record["model_id"] == model_id
            and record["dataset_id"] == dataset_id
            and record["query_pass_id"] == "shifted_prediction"
            and record["metric_key"] == "attention_absolute_deviation"
            and record["view_key"] == "global_bar"
            and facets.get("attention_view") == "raw"
            and facets.get("edge_ratio") == "0.1"
            and facets.get("region") in {"prefix", "interior"}
        ):
            if int(record["count"].item()) != 100:
                raise ValueError(f"Llama attention evidence is not full100: {path}")
            means[(dataset_id, record["method_key"], facets["region"])] = float(
                record["mean"].item()
            )
    return _ratios_from_means(means, str(path))[dataset_id]


def _ratios_from_means(
    means: dict[tuple[str, str, str], float], source: str
) -> dict[str, dict[str, float]]:
    datasets = sorted({key[0] for key in means})
    result: dict[str, dict[str, float]] = {}
    for dataset_id in datasets:
        try:
            vanilla_pre = means[(dataset_id, "vanilla_pic", "prefix")]
            vanilla_int = means[(dataset_id, "vanilla_pic", "interior")]
            kv_pre = means[(dataset_id, "kvpacket", "prefix")]
            kv_int = means[(dataset_id, "kvpacket", "interior")]
            sempic_int = means[(dataset_id, "sempic", "interior")]
        except KeyError as error:
            raise ValueError(f"Missing attention scalar {error.args[0]} in {source}.") from error
        if vanilla_pre <= 0 or vanilla_int <= 0:
            raise ValueError(f"Attention reference means must be positive in {source}.")
        values = {
            "kv_pre_ratio": kv_pre / vanilla_pre,
            "kv_rint": kv_int / vanilla_int,
            "sempic_rint": sempic_int / vanilla_int,
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"Invalid attention ratio in {source}.")
        result[dataset_id] = values
    return result


def _sink_point(path: Path, model_id: str, dataset_id: str) -> dict[str, Any]:
    aggregate = aggregate_partition(
        load_partition(path), num_position_bins=20, required_methods=("sempic",)
    )
    if (
        aggregate["model_id"] != model_id
        or aggregate["dataset_id"] != dataset_id
        or aggregate["query_pass_id"] != "shifted_prediction"
        or aggregate["sample_count"] != 100
    ):
        raise ValueError(
            f"Sink partition identity/sample scope does not match "
            f"{model_id}/{dataset_id}/shifted_prediction/full100: {path}"
        )
    if aggregate["sink_ratio_status"] != "defined":
        raise ValueError(f"Sink ratio is undefined for {path}.")
    return {
        "sink_profile": _profile_bins(aggregate["methods"]["sempic"]["positions"]),
        "sink_ratio": float(aggregate["sink_ratio"]),
    }


def build_bundle(
    *,
    authority_path: Path,
    boundary_path: Path,
    qwen4_plot_path: Path,
    qwen8_summary_path: Path,
    qwen8_partitions: dict[str, Path],
    llama_metrics: dict[str, Path],
    llama_partitions: dict[str, Path],
) -> dict[str, Any]:
    authority = load_authority(authority_path)
    boundary = _load_boundary_csv(boundary_path)
    qwen4 = _load_qwen4_plot(qwen4_plot_path)
    qwen8_ratios = _extract_summary_ratios(qwen8_summary_path)

    sources = [
        _source("f1_authority", authority_path),
        _source("qwen_boundary", boundary_path),
        _source("qwen4_attention", qwen4_plot_path),
        _source("qwen8_interior", qwen8_summary_path),
    ]
    for dataset_id in DATASET_ORDER:
        sources.extend(
            (
                _source(f"qwen8_sink_{dataset_id}", qwen8_partitions[dataset_id]),
                _source(f"llama_metrics_{dataset_id}", llama_metrics[dataset_id]),
                _source(f"llama_sink_{dataset_id}", llama_partitions[dataset_id]),
            )
        )

    models = []
    for model_id in MODEL_ORDER:
        points = []
        for dataset_id in DATASET_ORDER:
            point = {"dataset_id": dataset_id}
            point.update(_f1_point(authority[(model_id, dataset_id)]))
            if model_id == MODEL_ORDER[0]:
                boundary_values = boundary[(model_id, dataset_id)]
                attention = qwen4[dataset_id]
                point.update(
                    {
                        "kv_pre_ratio": boundary_values["pre"],
                        "kv_interior_ratio": attention["kv_rint"],
                        **attention,
                        "source_ids": {
                            "f1": "f1_authority",
                            "boundary_attention": "qwen_boundary",
                            "interior_attention": "qwen4_attention",
                            "sink_attention": "qwen4_attention",
                        },
                    }
                )
            elif model_id == MODEL_ORDER[1]:
                ratios = qwen8_ratios[dataset_id]
                point.update(
                    {
                        "kv_pre_ratio": ratios["kv_pre_ratio"],
                        "kv_interior_ratio": ratios["kv_rint"],
                        "kv_rint": ratios["kv_rint"],
                        "sempic_rint": ratios["sempic_rint"],
                        **_sink_point(qwen8_partitions[dataset_id], model_id, dataset_id),
                        "source_ids": {
                            "f1": "f1_authority",
                            "boundary_attention": "qwen8_interior",
                            "interior_attention": "qwen8_interior",
                            "sink_attention": f"qwen8_sink_{dataset_id}",
                        },
                    }
                )
            else:
                ratios = _extract_processed_ratios(
                    llama_metrics[dataset_id], model_id, dataset_id
                )
                point.update(
                    {
                        "kv_pre_ratio": ratios["kv_pre_ratio"],
                        "kv_interior_ratio": ratios["kv_rint"],
                        "kv_rint": ratios["kv_rint"],
                        "sempic_rint": ratios["sempic_rint"],
                        **_sink_point(llama_partitions[dataset_id], model_id, dataset_id),
                        "source_ids": {
                            "f1": "f1_authority",
                            "boundary_attention": f"llama_metrics_{dataset_id}",
                            "interior_attention": f"llama_metrics_{dataset_id}",
                            "sink_attention": f"llama_sink_{dataset_id}",
                        },
                    }
                )
            points.append(point)
        models.append(
            {"model_id": model_id, "display_name": DISPLAY_NAMES[model_id], "points": points}
        )
    return validate_plot_data(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "sources": sources,
            "models": models,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--boundary-plot-data", type=Path, required=True)
    parser.add_argument("--qwen4-plot-data", type=Path, required=True)
    parser.add_argument("--qwen8-summary-csv", type=Path, required=True)
    parser.add_argument("--qwen8-sink-partition", action="append", default=[])
    parser.add_argument("--llama-metrics", action="append", default=[])
    parser.add_argument("--llama-sink-partition", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_bundle(
        authority_path=args.authority,
        boundary_path=args.boundary_plot_data,
        qwen4_plot_path=args.qwen4_plot_data,
        qwen8_summary_path=args.qwen8_summary_csv,
        qwen8_partitions=_parse_assignments(
            args.qwen8_sink_partition, "--qwen8-sink-partition"
        ),
        llama_metrics=_parse_assignments(args.llama_metrics, "--llama-metrics"),
        llama_partitions=_parse_assignments(
            args.llama_sink_partition, "--llama-sink-partition"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, bundle)
    print(args.output)


if __name__ == "__main__":
    main()
