"""Build the four-dataset Qwen3-4B attention-sink evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch

try:
    from plot_scripts.attention_sink_data import (
        EXPECTED_DATASETS,
        INTERIOR_ERROR_METHODS,
        IncompletePartitionError,
        REQUIRED_MODEL_ID,
        REQUIRED_METHODS,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        aggregate_partition,
        membership_digest,
        normalize_method_key,
        recovery_fraction,
        safe_ratio,
        validate_plot_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from attention_sink_data import (
        EXPECTED_DATASETS,
        INTERIOR_ERROR_METHODS,
        IncompletePartitionError,
        REQUIRED_MODEL_ID,
        REQUIRED_METHODS,
        SCHEMA_NAME,
        SCHEMA_VERSION,
        aggregate_partition,
        membership_digest,
        normalize_method_key,
        recovery_fraction,
        safe_ratio,
        validate_plot_data,
    )

from sempic.attention_metrics.profile_storage import load_partition
from sempic.attention_metrics.processed_storage import load_processed_metrics
from sempic.utils.run_storage import atomic_write_json


DEFAULT_OUTPUT_DIR = "evidence_exports/2026-07-26-attention-sink-analysis-export"
CLAIM_ID = "CLAIM-ATTENTION-SINK-NOT-SUFFICIENT"
WORK_ITEM_ID = "EXPORT-01"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object.")
    return value


def _load_sink_partition(path: Path) -> dict[str, Any]:
    """Load current partitions and preserve legacy missing-seed evidence as blocked."""

    try:
        return load_partition(path)
    except ValueError as validation_error:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        identity = raw.get("partition_identity") if isinstance(raw, dict) else None
        if isinstance(identity, dict) and "eval_seed" not in identity:
            return raw
        raise validation_error


def _canonical_path(value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact path must be a non-empty string or null.")
    path = Path(value)
    return str(path.resolve(strict=False))


def _normalized_method_config(config: dict[str, Any]) -> dict[str, Any]:
    cache_comb = config.get("cache_comb")
    if not isinstance(cache_comb, dict):
        raise ValueError("Result config.cache_comb must be an object.")
    method = cache_comb.get("method")
    if not isinstance(method, str):
        raise ValueError("Result config.cache_comb.method must be a string.")
    return {
        "cache_comb": {
            **cache_comb,
            "method": normalize_method_key(method),
        },
        "packet_wrapper": {
            "path": _canonical_path((config.get("packet_wrapper") or {}).get("path"))
        },
        "lora": {"path": _canonical_path((config.get("lora") or {}).get("path"))},
        "compress": config.get("compress"),
        "quantization": config.get("quantization"),
    }


def _normalized_partition_method(method: dict[str, Any]) -> dict[str, Any]:
    resolved = method.get("resolved_method_config")
    if not isinstance(resolved, dict):
        raise ValueError("Partition method lacks resolved_method_config.")
    normalized = json.loads(json.dumps(resolved))
    cache_comb = normalized.get("cache_comb")
    if not isinstance(cache_comb, dict):
        raise ValueError("Partition resolved cache_comb must be an object.")
    cache_comb["method"] = normalize_method_key(cache_comb.get("method"))
    for artifact in ("packet_wrapper", "lora"):
        record = normalized.get(artifact)
        if not isinstance(record, dict):
            raise ValueError(f"Partition resolved {artifact} must be an object.")
        record["path"] = _canonical_path(record.get("path"))
    return normalized


def _validate_result_match(
    *,
    result_path: Path,
    payload: dict[str, Any],
    partition: dict[str, Any],
    expected_method: str,
) -> dict[str, Any]:
    config = payload.get("config")
    result = payload.get("result")
    identity = partition["partition_identity"]
    if not isinstance(config, dict) or not isinstance(result, dict):
        raise ValueError(f"{result_path}: config and result must be objects.")
    if config.get("model") != identity.get("model_config"):
        raise ValueError(f"{result_path}: model config does not match attention partition.")
    if config.get("dataset") != identity.get("dataset_config"):
        raise ValueError(f"{result_path}: dataset config does not match attention partition.")
    expected_seed = identity.get("eval_seed")
    if config.get("seed") != expected_seed:
        raise ValueError(
            f"{result_path}: top-level seed does not match frozen partition eval_seed."
        )
    partition_methods = {
        method["method_key"]: method
        for method in identity["methods"]
        if isinstance(method, dict) and "method_key" in method
    }
    partition_method = partition_methods.get(expected_method)
    if partition_method is None:
        raise ValueError(f"Partition does not contain method {expected_method}.")
    if _normalized_method_config(config) != _normalized_partition_method(partition_method):
        raise ValueError(f"{result_path}: method/artifact config does not match partition.")
    f1 = result.get("f1")
    if (
        isinstance(f1, bool)
        or not isinstance(f1, (int, float))
        or not math.isfinite(f1)
    ):
        raise ValueError(f"{result_path}: result.f1 must be finite.")
    source_config = partition_method.get("source_config")
    resolved = partition_method.get("resolved_method_config")
    checkpoint = "not applicable: method has no learned artifact"
    if expected_method == "sempic":
        checkpoint = resolved["lora"]["path"]
    return {
        "measurement_id": f"behavior:{identity['dataset_id']}:{expected_method}",
        "model_id": identity["model_id"],
        "dataset_id": identity["dataset_id"],
        "method_key": expected_method,
        "quality_metric_name": "corpus_whitespace_token_micro_f1",
        "quality_metric_value": float(f1),
        "eval_seed": identity["eval_seed"],
        "dataset_seed": identity["dataset_config"].get("seed"),
        "source_config": source_config,
        "resolved_config": "embedded:result.config",
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "checkpoint_or_artifact": checkpoint,
        "known_uncertainty": (
            "Historical evaluator output does not record per-sample membership IDs; "
            "membership is config-implied by the matched partition."
        ),
    }


def _method_source(partition: dict[str, Any], method_key: str) -> dict[str, Any]:
    methods = partition["partition_identity"]["methods"]
    record = next(method for method in methods if method["method_key"] == method_key)
    resolved = record["resolved_method_config"]
    checkpoint = "not applicable: method has no learned artifact"
    if method_key == "sempic":
        checkpoint = resolved["lora"]["path"]
    elif method_key == "kvpacket":
        checkpoint = resolved["packet_wrapper"]["path"]
    return {
        "source_config": record["source_config"],
        "checkpoint_or_artifact": checkpoint,
    }


def _processed_interior_errors(
    *,
    model_id: str,
    partitions: dict[str, dict[str, Any]],
    partition_paths: dict[str, Path],
    processed_metrics: dict[str, Any],
    processed_metrics_path: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Bind exact processed records to their source partitions and export D_int."""

    expected_source_map = {}
    for dataset_id, partition in partitions.items():
        identity = partition.get("partition_identity", {})
        source_key = (
            identity.get("model_id"),
            identity.get("dataset_id"),
            identity.get("query_pass_id"),
        )
        if source_key != (model_id, dataset_id, "shifted_prediction"):
            raise ValueError(f"Partition identity mismatch for {dataset_id}.")
        if "attention_profile" not in identity.get("query_spec", {}).get("reducers", []):
            raise ValueError(f"Partition {dataset_id} lacks attention_profile reducer output.")
        method_keys = {
            method.get("method_key")
            for method in identity.get("methods", [])
            if isinstance(method, dict)
        }
        expected_methods = {"full_recompute", *INTERIOR_ERROR_METHODS}
        if method_keys != expected_methods:
            raise ValueError(
                f"Partition {dataset_id} methods must exactly contain Full, Vanilla, "
                "KV Packet, and SemPIC."
            )
        partition_path = partition_paths.get(dataset_id)
        if partition_path is None or not partition_path.is_file():
            raise ValueError(f"Missing source partition path for {dataset_id}.")
        expected_source_map[source_key] = partition["partition_fingerprint"]

    actual_source_map = {
        (source["model_id"], source["dataset_id"], source["query_pass_id"]): source[
            "partition_fingerprint"
        ]
        for source in processed_metrics["source_partitions"]
    }
    if actual_source_map != expected_source_map:
        raise ValueError(
            "Processed metrics source identities/fingerprints do not exactly match supplied partitions."
        )

    target_facets = {
        "attention_view": "raw",
        "edge_ratio": "0.1",
        "region": "interior",
    }
    metrics_sha256 = _sha256(processed_metrics_path)
    errors_by_dataset: dict[str, list[dict[str, Any]]] = {}
    table_rows: list[dict[str, Any]] = []
    for dataset_id, partition in partitions.items():
        identity = partition["partition_identity"]
        partition_path = partition_paths[dataset_id]
        partition_sha256 = _sha256(partition_path)
        methods = {method["method_key"]: method for method in identity["methods"]}
        dataset_errors = []
        for method_key in INTERIOR_ERROR_METHODS:
            matches = [
                record
                for record in processed_metrics["records"]
                if record["model_id"] == model_id
                and record["dataset_id"] == dataset_id
                and record["query_pass_id"] == "shifted_prediction"
                and record["metric_key"] == "attention_absolute_deviation"
                and record["view_key"] == "global_bar"
                and record["method_key"] == method_key
                and record["facets"] == target_facets
                and record["axes"] == []
                and record["coordinates"] == {}
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one interior attention error record for "
                    f"{dataset_id}/{method_key}; found {len(matches)}."
                )
            record = matches[0]
            mean = float(record["mean"].item())
            sem = float(record["sem"].item())
            count = int(record["count"].item())
            if not math.isfinite(mean) or not math.isfinite(sem) or sem < 0 or count <= 0:
                raise ValueError(
                    f"Interior attention error is invalid for {dataset_id}/{method_key}."
                )
            expected_count = len(partition["samples"])
            if count != expected_count:
                raise ValueError(
                    f"Interior attention error count {count} does not match partition "
                    f"sample count {expected_count} for {dataset_id}/{method_key}."
                )
            measurement_id = f"interior_error:{dataset_id}:{method_key}:raw:0.1"
            dataset_errors.append(
                {
                    "method_key": method_key,
                    "mean": mean,
                    "sem": sem,
                    "count": count,
                    "measurement_id": measurement_id,
                }
            )
            candidate = methods[method_key]
            reference = methods["full_recompute"]
            table_rows.append(
                {
                    "measurement_id": measurement_id,
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "query_pass_id": "shifted_prediction",
                    "method_key": method_key,
                    "reference_method_key": "full_recompute",
                    "metric_key": "attention_absolute_deviation",
                    "view_key": "global_bar",
                    "attention_view": "raw",
                    "edge_ratio": "0.1",
                    "region": "interior",
                    "mean": mean,
                    "sem": sem,
                    "count": count,
                    "candidate_source_config": candidate["source_config"],
                    "reference_source_config": reference["source_config"],
                    "candidate_runtime_fingerprint": candidate["runtime_fingerprint"],
                    "reference_runtime_fingerprint": reference["runtime_fingerprint"],
                    "candidate_resolved_method_config": json.dumps(
                        candidate["resolved_method_config"], sort_keys=True
                    ),
                    "reference_resolved_method_config": json.dumps(
                        reference["resolved_method_config"], sort_keys=True
                    ),
                    "partition_path": str(partition_path.resolve()),
                    "partition_sha256": partition_sha256,
                    "partition_identity_fingerprint": partition["partition_fingerprint"],
                    "processed_metrics_path": str(processed_metrics_path.resolve()),
                    "processed_metrics_sha256": metrics_sha256,
                    "processing_fingerprint": processed_metrics["processing_fingerprint"],
                }
            )
        errors_by_dataset[dataset_id] = dataset_errors
    return errors_by_dataset, table_rows


def _git_context() -> tuple[str, str]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        state = subprocess.run(
            ["git", "status", "--short"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not recorded: git unavailable", "not recorded: git unavailable"
    return revision, state or "clean"


def _provenance(
    *,
    measurement_id: str,
    source_config: str,
    result_path: str,
    source_artifact_sha256: str,
    paired_quality_result_path: str,
    paired_quality_result_sha256: str,
    checkpoint: str,
    metric_definition: str,
    aggregation_rule: str,
    unit: str,
    sample_membership: str,
    sample_count: int,
    seed: object,
    known_uncertainty: str,
    code_revision: str,
    working_tree_state: str,
) -> dict[str, Any]:
    return {
        "measurement_id": measurement_id,
        "claim_id": CLAIM_ID,
        "work_item_id": WORK_ITEM_ID,
        "result_path": result_path,
        "source_artifact_sha256": source_artifact_sha256,
        "paired_quality_result_path": paired_quality_result_path,
        "paired_quality_result_sha256": paired_quality_result_sha256,
        "source_config": source_config,
        "resolved_config": "embedded in source partition/result",
        "timestamped_run": "not recorded: supplied result path may be stable or timestamped",
        "cli_args": "not recorded: source run metadata not embedded in partition",
        "run_log": "not recorded: source run metadata not embedded in partition",
        "checkpoint_or_artifact": checkpoint,
        "training_run": (
            str(Path(checkpoint).parent) if not checkpoint.startswith("not applicable:") else checkpoint
        ),
        "metric_definition": metric_definition,
        "aggregation_rule": aggregation_rule,
        "unit": unit,
        "sample_membership": sample_membership,
        "sample_or_repeat_count": sample_count,
        "seed_or_seed_list": seed,
        "code_revision": code_revision,
        "working_tree_state": working_tree_state,
        "source_hardware_and_runtime": (
            "not recorded: supplied source artifact does not preserve inference "
            "hardware/runtime metadata"
        ),
        "export_hardware_and_runtime": "deterministic CPU export of saved statistics",
        "hardware_and_runtime": (
            "Source inference hardware/runtime not recorded; deterministic evidence "
            "export executed on CPU."
        ),
        "known_uncertainty": known_uncertainty,
    }


def build_plot_data(
    *,
    model_id: str,
    partitions: dict[str, dict[str, Any]],
    partition_paths: dict[str, Path],
    results: dict[tuple[str, str], tuple[Path, dict[str, Any]]],
    processed_metrics: dict[str, Any],
    processed_metrics_path: Path,
    num_position_bins: int = 20,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    tables = {
        "behavior": [],
        "positions": [],
        "regions": [],
        "candidates": [],
        "resolved_configs": [],
        "interior_errors": [],
    }
    provenance: list[dict[str, Any]] = []
    code_revision, working_tree_state = _git_context()
    interior_errors_by_dataset, interior_error_rows = _processed_interior_errors(
        model_id=model_id,
        partitions=partitions,
        partition_paths=partition_paths,
        processed_metrics=processed_metrics,
        processed_metrics_path=processed_metrics_path,
    )
    metrics_sha256 = _sha256(processed_metrics_path)
    interior_error_rows_by_dataset = {
        dataset_id: [
            row for row in interior_error_rows if row["dataset_id"] == dataset_id
        ]
        for dataset_id in partitions
    }

    for dataset_id in EXPECTED_DATASETS:
        partition = partitions.get(dataset_id)
        point_base = {"model_id": model_id, "dataset_id": dataset_id}
        if partition is None:
            reason = "No shifted-prediction raw-profile partition was supplied."
            points.append({**point_base, "status": "not-run", "status_reason": reason})
            tables["candidates"].append({**point_base, "status": "not-run", "reason": reason})
            continue
        partition_path = partition_paths.get(dataset_id)
        if partition_path is None or not partition_path.is_file():
            raise ValueError(f"Missing source partition path for {dataset_id}.")
        partition_sha256 = _sha256(partition_path)
        identity = partition.get("partition_identity", {})
        if identity.get("model_id") != model_id or identity.get("dataset_id") != dataset_id:
            raise ValueError(f"Partition identity mismatch for {dataset_id}.")
        if "eval_seed" not in identity:
            reason = (
                "Legacy partition does not record top-level eval_seed; exact behavior "
                "comparability cannot be established."
            )
            points.append({**point_base, "status": "blocked", "status_reason": reason})
            tables["candidates"].append(
                {**point_base, "status": "blocked", "reason": reason}
            )
            continue
        missing_results = [
            method for method in REQUIRED_METHODS if (dataset_id, method) not in results
        ]
        if missing_results:
            reason = f"Missing exact behavior results: {', '.join(missing_results)}"
            points.append({**point_base, "status": "not-run", "status_reason": reason})
            tables["candidates"].append({**point_base, "status": "not-run", "reason": reason})
            continue

        try:
            aggregate = aggregate_partition(
                partition, num_position_bins=num_position_bins
            )
        except IncompletePartitionError as error:
            reason = str(error)
            points.append({**point_base, "status": "blocked", "status_reason": reason})
            tables["candidates"].append(
                {**point_base, "status": "blocked", "reason": reason}
            )
            continue
        behavior = []
        behavior_by_method: dict[str, dict[str, Any]] = {}
        seed_scope = {
            "eval_seed": identity["eval_seed"],
            "dataset_seed": aggregate["seed"],
        }
        for method in REQUIRED_METHODS:
            result_path, payload = results[(dataset_id, method)]
            measurement = _validate_result_match(
                result_path=result_path,
                payload=payload,
                partition=partition,
                expected_method=method,
            )
            tables["behavior"].append(measurement)
            tables["resolved_configs"].append(
                {
                    "measurement_id": measurement["measurement_id"],
                    "source_config": measurement["source_config"],
                    "result_path": measurement["result_path"],
                    "resolved_config": payload["config"],
                }
            )
            behavior_by_method[method] = measurement
            behavior.append(
                {
                    "method_key": method,
                    "f1": measurement["quality_metric_value"],
                    "measurement_id": measurement["measurement_id"],
                    "result_path": measurement["result_path"],
                }
            )
            provenance.append(
                _provenance(
                    measurement_id=measurement["measurement_id"],
                    source_config=measurement["source_config"],
                    result_path=measurement["result_path"],
                    source_artifact_sha256=measurement["result_sha256"],
                    paired_quality_result_path=measurement["result_path"],
                    paired_quality_result_sha256=measurement["result_sha256"],
                    checkpoint=measurement["checkpoint_or_artifact"],
                    metric_definition="run_eval.py corpus whitespace-token micro-F1",
                    aggregation_rule="corpus-level TP/FP/FN accumulation",
                    unit="evaluation configuration",
                    sample_membership=(
                        "config-implied; attention membership digest="
                        + aggregate["sample_membership_digest"]
                    ),
                    sample_count=aggregate["sample_count"],
                    seed=seed_scope,
                    known_uncertainty=measurement["known_uncertainty"],
                    code_revision=code_revision,
                    working_tree_state=working_tree_state,
                )
            )

        full_f1 = behavior_by_method["full_recompute"]["quality_metric_value"]
        vanilla_f1 = behavior_by_method["vanilla_pic"]["quality_metric_value"]
        sempic_f1 = behavior_by_method["sempic"]["quality_metric_value"]
        recovery, recovery_status = recovery_fraction(full_f1, vanilla_f1, sempic_f1)

        plot_profiles = []
        plot_regions = []
        for method in REQUIRED_METHODS:
            method_source = _method_source(partition, method)
            method_result = behavior_by_method[method]
            summary = aggregate["methods"][method]
            profile_bins = []
            for position in summary["positions"]:
                measurement_id = f"position:{dataset_id}:{method}:{position['bin_index']}"
                profile_bins.append({**position, "measurement_id": measurement_id})
                for sample_record in summary["sample_values"]:
                    tables["positions"].append(
                        {
                            **point_base,
                            "method_key": method,
                            "query_pass_id": "shifted_prediction",
                            "sample_or_aggregate_id": sample_record["sample_id"],
                            "logical_position_or_region": f"bin_{position['bin_index']:02d}",
                            "position_start": position["start"],
                            "position_end": position["end"],
                            "sink_metric_name": "raw_attention_density",
                            "sink_metric_value": sample_record["bins"][position["bin_index"]],
                            "quality_metric_name": method_result["quality_metric_name"],
                            "quality_metric_value": method_result["quality_metric_value"],
                            "sample_count": aggregate["sample_count"],
                            "seed": identity["eval_seed"],
                            "dataset_seed": aggregate["seed"],
                            "source_config": method_source["source_config"],
                            "result_path": str(partition_path.resolve()),
                            "partition_sha256": partition_sha256,
                            "quality_result_path": method_result["result_path"],
                            "quality_result_sha256": method_result["result_sha256"],
                            "checkpoint_or_artifact": method_source["checkpoint_or_artifact"],
                            "analysis_code": "plot_scripts/build_attention_sink_data.py",
                            "known_uncertainty": "within-run sample value; no seed-level uncertainty",
                            "measurement_id": measurement_id,
                        }
                    )
                provenance.append(
                    _provenance(
                        measurement_id=measurement_id,
                        source_config=method_source["source_config"],
                        result_path=str(partition_path.resolve()),
                        source_artifact_sha256=partition_sha256,
                        paired_quality_result_path=method_result["result_path"],
                        paired_quality_result_sha256=method_result["result_sha256"],
                        checkpoint=method_source["checkpoint_or_artifact"],
                        metric_definition="raw canonical-PIC attention probability density",
                        aggregation_rule=(
                            "query/head mean in reducer; token-width bin mean; equal chunk; "
                            "equal layer; equal sample mean/SEM"
                        ),
                        unit="evaluation sample",
                        sample_membership=aggregate["sample_membership_digest"],
                        sample_count=aggregate["sample_count"],
                        seed=seed_scope,
                        known_uncertainty="SEM is within-run sample variability, not seed uncertainty.",
                        code_revision=code_revision,
                        working_tree_state=working_tree_state,
                    )
                )
            plot_profiles.append({"method_key": method, "bins": profile_bins})

            for region_name in ("leading", "interior"):
                region = summary["regions"][region_name]
                region_label = "pre-region" if region_name == "leading" else region_name
                measurement_id = f"region:{dataset_id}:{method}:{region_name}"
                plot_region = {
                    "method_key": method,
                    "region": region_name,
                    **region,
                    "measurement_id": measurement_id,
                }
                plot_regions.append(plot_region)
                tables["regions"].append(
                    {
                        **point_base,
                        **plot_region,
                        "query_pass_id": "shifted_prediction",
                        "sample_or_aggregate_id": "aggregate",
                        "logical_position_or_region": region_name,
                        "sink_metric_name": "raw_attention_density",
                        "sink_metric_value": region["mean"],
                        "quality_metric_name": method_result["quality_metric_name"],
                        "quality_metric_value": method_result["quality_metric_value"],
                        "sample_count": aggregate["sample_count"],
                        "seed": identity["eval_seed"],
                        "dataset_seed": aggregate["seed"],
                        "source_config": method_source["source_config"],
                        "result_path": str(partition_path.resolve()),
                        "partition_sha256": partition_sha256,
                        "quality_result_path": method_result["result_path"],
                        "quality_result_sha256": method_result["result_sha256"],
                        "checkpoint_or_artifact": method_source["checkpoint_or_artifact"],
                        "analysis_code": "plot_scripts/build_attention_sink_data.py",
                        "known_uncertainty": "SEM is within-run sample variability, not seed uncertainty.",
                    }
                )
                provenance.append(
                    _provenance(
                        measurement_id=measurement_id,
                        source_config=method_source["source_config"],
                        result_path=str(partition_path.resolve()),
                        source_artifact_sha256=partition_sha256,
                        paired_quality_result_path=method_result["result_path"],
                        paired_quality_result_sha256=method_result["result_sha256"],
                        checkpoint=method_source["checkpoint_or_artifact"],
                        metric_definition=f"token-width-normalized {region_label} raw attention density",
                        aggregation_rule=(
                            "query/head mean in reducer; token-width region mean; equal chunk; "
                            "equal layer; equal sample mean/SEM"
                        ),
                        unit="evaluation sample",
                        sample_membership=aggregate["sample_membership_digest"],
                        sample_count=aggregate["sample_count"],
                        seed=seed_scope,
                        known_uncertainty="SEM is within-run sample variability, not seed uncertainty.",
                        code_revision=code_revision,
                        working_tree_state=working_tree_state,
                    )
                )

        sink_ratio = aggregate["sink_ratio"]
        sink_status = aggregate["sink_ratio_status"]
        coexistence = (
            "supports_coexistence"
            if sempic_f1 > vanilla_f1 and sink_ratio is not None and sink_ratio > 1
            else "does_not_support_coexistence"
        )
        interior_errors = interior_errors_by_dataset[dataset_id]
        dataset_interior_rows = interior_error_rows_by_dataset[dataset_id]
        tables["interior_errors"].extend(dataset_interior_rows)
        for row in dataset_interior_rows:
            method_source = _method_source(partition, row["method_key"])
            record = _provenance(
                measurement_id=row["measurement_id"],
                source_config=row["candidate_source_config"],
                result_path=row["processed_metrics_path"],
                source_artifact_sha256=metrics_sha256,
                paired_quality_result_path=(
                    "not applicable: D_int does not require a behavior result"
                ),
                paired_quality_result_sha256=(
                    "not applicable: no paired behavior result required"
                ),
                checkpoint=method_source["checkpoint_or_artifact"],
                metric_definition=(
                    "D_int(method, Full): equal-sample mean of post-softmax raw "
                    "attention probability absolute deviation from Full Recompute "
                    "over normalized interior [0.1,0.9)"
                ),
                aggregation_rule=(
                    "query/head mean in reducer; token-width interior mean; equal "
                    "chunk; equal layer; equal sample mean/SEM"
                ),
                unit="evaluation sample",
                sample_membership=membership_digest(
                    [sample["sample_id"] for sample in partition["samples"]]
                ),
                sample_count=row["count"],
                seed=seed_scope,
                known_uncertainty=(
                    "SEM is within-run sample variability, not seed uncertainty. "
                    "The partition fingerprint binds partition identity plus declared "
                    "layer/head counts, not sample or reducer tensor shape/content."
                ),
                code_revision=code_revision,
                working_tree_state=working_tree_state,
            )
            record.update(
                {
                    "reference_method_key": "full_recompute",
                    "reference_source_config": row["reference_source_config"],
                    "candidate_runtime_fingerprint": row[
                        "candidate_runtime_fingerprint"
                    ],
                    "reference_runtime_fingerprint": row[
                        "reference_runtime_fingerprint"
                    ],
                    "candidate_resolved_method_config": json.loads(
                        row["candidate_resolved_method_config"]
                    ),
                    "reference_resolved_method_config": json.loads(
                        row["reference_resolved_method_config"]
                    ),
                    "partition_path": row["partition_path"],
                    "partition_sha256": row["partition_sha256"],
                    "partition_identity_fingerprint": row[
                        "partition_identity_fingerprint"
                    ],
                    "processed_metrics_path": row["processed_metrics_path"],
                    "processed_metrics_sha256": row["processed_metrics_sha256"],
                    "processing_fingerprint": row["processing_fingerprint"],
                    "processing_config": processed_metrics["processing_config"],
                    "partition_artifact_snapshots": partition[
                        "partition_identity"
                    ]["artifact_snapshots"],
                    "artifact_snapshot_limitation": (
                        "Artifact snapshots with files=[] bind canonical paths only; "
                        "they are not learned-weight content hashes."
                    ),
                }
            )
            provenance.append(record)
        interior_error_by_method = {
            row["method_key"]: row for row in interior_errors
        }
        relative_interior_errors = []
        for method_key in ("kvpacket", "sempic"):
            numerator = interior_error_by_method[method_key]
            denominator = interior_error_by_method["vanilla_pic"]
            value, status = safe_ratio(numerator["mean"], denominator["mean"])
            relative_interior_errors.append(
                {
                    "method_key": method_key,
                    "value": value,
                    "status": status,
                    "numerator_measurement_id": numerator["measurement_id"],
                    "denominator_measurement_id": denominator["measurement_id"],
                }
            )
        point = {
            **point_base,
            "status": "pass",
            "status_reason": "Complete matched raw-profile and behavior evidence.",
            "query_pass_id": "shifted_prediction",
            "sample_count": aggregate["sample_count"],
            "seed": identity["eval_seed"],
            "dataset_seed": aggregate["seed"],
            "sample_membership_digest": aggregate["sample_membership_digest"],
            "region_rule": aggregate["region_rule"],
            "behavior": behavior,
            "profiles": plot_profiles,
            "regions": plot_regions,
            "interior_attention_errors": interior_errors,
            "relative_interior_attention_errors": relative_interior_errors,
            "recovery_fraction": recovery,
            "recovery_fraction_status": recovery_status,
            "recovery_fraction_measurement_id": f"estimand:{dataset_id}:recovery_fraction",
            "sink_ratio": sink_ratio,
            "sink_ratio_status": sink_status,
            "sink_ratio_measurement_id": f"estimand:{dataset_id}:sink_ratio",
            "coexistence_conditions": {
                "f1_sempic_greater_than_vanilla": sempic_f1 > vanilla_f1,
                "sink_ratio_greater_than_one": (
                    sink_ratio > 1 if sink_ratio is not None else None
                ),
            },
            "interpretation_status": coexistence,
        }
        points.append(point)
        tables["candidates"].append(
            {**point_base, "status": "pass", "reason": point["status_reason"]}
        )

        for name, value, status, definition in (
            (
                "recovery_fraction",
                recovery,
                recovery_status,
                "Recovery=(F1_SemPIC-F1_Vanilla)/(F1_Full-F1_Vanilla)",
            ),
            (
                "sink_ratio",
                sink_ratio,
                sink_status,
                "S=aggregate SemPIC pre-region density/aggregate SemPIC interior density",
            ),
        ):
            if name == "sink_ratio":
                estimand_result_path = str(partition_path.resolve())
                estimand_source_sha256 = partition_sha256
                paired_result_path = behavior_by_method["sempic"]["result_path"]
                paired_result_sha256 = behavior_by_method["sempic"]["result_sha256"]
            else:
                estimand_result_path = (
                    "multiple matched behavior result paths; see behavior measurement IDs"
                )
                estimand_source_sha256 = (
                    "multiple hashes; see behavior measurement provenance"
                )
                paired_result_path = estimand_result_path
                paired_result_sha256 = estimand_source_sha256
            provenance.append(
                _provenance(
                    measurement_id=f"estimand:{dataset_id}:{name}",
                    source_config="multiple matched source configs; see behavior/region measurement IDs",
                    result_path=estimand_result_path,
                    source_artifact_sha256=estimand_source_sha256,
                    paired_quality_result_path=paired_result_path,
                    paired_quality_result_sha256=paired_result_sha256,
                    checkpoint=behavior_by_method["sempic"]["checkpoint_or_artifact"],
                    metric_definition=f"{definition}; status={status}; value={value}",
                    aggregation_rule="derived from exported aggregate components; never clipped",
                    unit="model-dataset configuration",
                    sample_membership=aggregate["sample_membership_digest"],
                    sample_count=aggregate["sample_count"],
                    seed=seed_scope,
                    known_uncertainty="descriptive single-run estimand",
                    code_revision=code_revision,
                    working_tree_state=working_tree_state,
                )
            )

        for relative in relative_interior_errors:
            provenance.append(
                {
                    "measurement_id": (
                        f"estimand:{dataset_id}:relative_interior_attention_error:"
                        f"{relative['method_key']}"
                    ),
                    "claim_id": CLAIM_ID,
                    "work_item_id": WORK_ITEM_ID,
                    "metric_definition": (
                        "R_int(method)=D_int(method,Full)/D_int(Vanilla,Full); "
                        "Relative interior attention error (lower is better)"
                    ),
                    "method_key": relative["method_key"],
                    "value": relative["value"],
                    "status": relative["status"],
                    "numerator_measurement_id": relative[
                        "numerator_measurement_id"
                    ],
                    "denominator_measurement_id": relative[
                        "denominator_measurement_id"
                    ],
                    "uncertainty": (
                        "No ratio SEM is exported because paired-sample covariance is not "
                        "available from aggregate component summaries."
                    ),
                    "code_revision": code_revision,
                    "working_tree_state": working_tree_state,
                }
            )

    data = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "request_id": "2026-07-26-attention-sink-analysis-export",
        "model_id": model_id,
        "expected_identity_count": len(EXPECTED_DATASETS),
        "points": points,
    }
    return validate_plot_data(data), tables, provenance


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_bundle(
    *,
    output_dir: Path,
    plot_data: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    provenance: list[dict[str, Any]],
    input_paths: list[Path],
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(data_dir / "plot_data.json", plot_data)
    _write_csv(data_dir / "candidate_inventory.csv", tables["candidates"])
    if tables["behavior"]:
        _write_csv(data_dir / "behavior_measurements.csv", tables["behavior"])
        _write_csv(data_dir / "position_measurements.csv", tables["positions"])
        _write_csv(data_dir / "region_measurements.csv", tables["regions"])
        _write_csv(
            data_dir / "interior_attention_error_measurements.csv",
            tables["interior_errors"],
        )
        atomic_write_json(data_dir / "resolved_configs.json", tables["resolved_configs"])
    statistics = {
        "model_id": plot_data["model_id"],
        "estimands": [
            {
                "dataset_id": point["dataset_id"],
                "status": point["status"],
                "recovery_fraction": point.get("recovery_fraction"),
                "recovery_fraction_status": point.get("recovery_fraction_status"),
                "sink_ratio": point.get("sink_ratio"),
                "sink_ratio_status": point.get("sink_ratio_status"),
                "interior_attention_errors": point.get("interior_attention_errors"),
                "relative_interior_attention_errors": point.get(
                    "relative_interior_attention_errors"
                ),
                "interpretation_status": point.get("interpretation_status"),
            }
            for point in plot_data["points"]
        ],
    }
    atomic_write_json(data_dir / "statistics.json", statistics)
    with (data_dir / "provenance.jsonl").open("w", encoding="utf-8") as output:
        for record in provenance:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "schema.md").write_text(
        """# Attention Sink Evidence Schema

`data/plot_data.json` is the sole figure input. It contains the fixed Qwen3-4B
Biography, HotpotQA, MuSiQue, and NIAH identities in that order. Every identity
has an explicit `pass`, `fail`, `blocked`, or `not-run` status.

For complete identities, `profiles` and `regions` contain raw post-softmax
canonical-PIC attention densities. Aggregation is query/head mean in the
reducer, token-width-normalized region or bin mean, equal chunk mean, equal
layer mean, then equal sample mean/SEM. The pre region is normalized position
`[0,0.1)` (stored under the machine key `leading`), and the interior is
`[0.1,0.9)`. `S` divides aggregate SemPIC pre-region density by aggregate
SemPIC interior density. Behavioral recovery uses exact matched Full, Vanilla,
and SemPIC F1.
`D_int(method, Full)` is the post-softmax raw attention-probability absolute
deviation from Full Recompute over normalized interior `[0.1,0.9)`. Relative
interior attention error is
`R_int(method)=D_int(method,Full)/D_int(Vanilla,Full)` (lower is better).
Only KV Packet and SemPIC ratios are exported. Ratio SEM is intentionally not
reported because paired-sample covariance is unavailable from the aggregate
component summaries.
Undefined derived values are JSON null with an `undefined_*` status.
The partition identity's top-level `eval_seed` is exported separately from the
dataset sampling seed; exact behavior matching uses `eval_seed` and never
assumes the two seeds are equal.

CSV measurement rows retain method-level source/result links. JSONL provenance
records are keyed by `measurement_id`; sample SEM is within-run variability and
is not training-seed uncertainty. The partition fingerprint binds exact
partition identity plus declared layer/head counts, not sample or reducer tensor
shape/content. Export-time SHA-256 values separately cover the current
processed-metrics and partition files. Artifact snapshots with empty `files`
lists bind canonical paths only, not learned-weight contents.
""",
        encoding="utf-8",
    )
    complete = sum(point["status"] == "pass" for point in plot_data["points"])
    export_status = "complete" if complete == len(EXPECTED_DATASETS) else "partial"
    existing_figures = sorted((output_dir / "figures").glob("attention_sink_diagnostic.*"))
    diagnostic_status = "stale_unverified" if existing_figures else "not-run"
    (output_dir / "completion_report.md").write_text(
        f"""# Attention Sink Evidence Completion Report

- Export status: {export_status}
- Diagnostic figure status: {diagnostic_status}
- Overall evidence status: pending diagnostic generation and rendered verification
- Completed minimum identities: {complete}/{len(EXPECTED_DATASETS)}
- Scope: fixed {plot_data['model_id']} across Biography, HotpotQA, MuSiQue, and NIAH
- Query pass: `shifted_prediction`
- Optional experiment status: not-run
- Verification: builder schema validation completed; rendered verification is separate
- Interpretation: descriptive and non-causal; see `data/statistics.json`
""",
        encoding="utf-8",
    )
    artifacts = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "bundle_manifest.json"
    )
    atomic_write_json(
        output_dir / "bundle_manifest.json",
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "export_status": export_status,
            "diagnostic_figure_status": diagnostic_status,
            "inputs": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in input_paths
            ],
            "artifacts_at_export": artifacts,
        },
    )


def _parse_mapping(values: list[str], *, kind: str) -> dict[Any, Path]:
    parsed: dict[Any, Path] = {}
    for value in values:
        try:
            identity, raw_path = value.split("=", 1)
        except ValueError as error:
            raise ValueError(f"{kind} must use ID=PATH: {value}") from error
        if kind == "partition":
            key: Any = identity
            if key not in EXPECTED_DATASETS:
                raise ValueError(f"Unknown dataset in partition mapping: {key}")
        else:
            try:
                dataset_id, method = identity.split(":", 1)
            except ValueError as error:
                raise ValueError(f"result must use DATASET:METHOD=PATH: {value}") from error
            method = normalize_method_key(method)
            key = (dataset_id, method)
            if dataset_id not in EXPECTED_DATASETS or method not in REQUIRED_METHODS:
                raise ValueError(f"Unknown result identity: {identity}")
        if key in parsed:
            raise ValueError(f"Duplicate {kind} mapping: {identity}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"{kind} path is not a file: {path}")
        parsed[key] = path
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=REQUIRED_MODEL_ID,
        choices=(REQUIRED_MODEL_ID,),
        help="Fixed Qwen3-4B model ID.",
    )
    parser.add_argument(
        "--partition",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Shifted-prediction statistics partition; repeat for each available dataset.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="DATASET:METHOD=PATH",
        help="Exact matched result JSON; method is full_recompute, no_recompute, or sempic.",
    )
    parser.add_argument(
        "--processed-metrics",
        type=Path,
        required=True,
        help="Validated metrics.pt containing Full-relative interior attention errors.",
    )
    parser.add_argument("--num-position-bins", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    partition_paths = _parse_mapping(args.partition, kind="partition")
    result_paths = _parse_mapping(args.result, kind="result")
    output_resolved = args.output_dir.resolve()
    if not args.processed_metrics.is_file():
        raise ValueError(f"processed metrics path is not a file: {args.processed_metrics}")
    input_paths = [
        *partition_paths.values(),
        *result_paths.values(),
        args.processed_metrics,
    ]
    if any(output_resolved == path.resolve() or output_resolved in path.resolve().parents for path in input_paths):
        raise ValueError("Output directory must not be an input file or its parent.")
    partitions = {
        dataset: _load_sink_partition(path)
        for dataset, path in partition_paths.items()
    }
    results = {key: (path, _read_json(path)) for key, path in result_paths.items()}
    processed_metrics = load_processed_metrics(args.processed_metrics)
    plot_data, tables, provenance = build_plot_data(
        model_id=args.model_id,
        partitions=partitions,
        partition_paths=partition_paths,
        results=results,
        processed_metrics=processed_metrics,
        processed_metrics_path=args.processed_metrics,
        num_position_bins=args.num_position_bins,
    )
    write_bundle(
        output_dir=args.output_dir,
        plot_data=plot_data,
        tables=tables,
        provenance=provenance,
        input_paths=input_paths,
    )


if __name__ == "__main__":
    main()
