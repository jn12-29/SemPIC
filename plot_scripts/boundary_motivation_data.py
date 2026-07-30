"""Data contracts and extraction for boundary-conditioning motivation evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

try:
    from plot_scripts.evidence_config_match import (
        behavioral_projection,
        compare_behavioral_configs,
        differences_json,
        load_completed_result,
        measurement_id,
        sha256_file,
        sha256_value,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from evidence_config_match import (
        behavioral_projection,
        compare_behavioral_configs,
        differences_json,
        load_completed_result,
        measurement_id,
        sha256_file,
        sha256_value,
    )


SCHEMA_VERSION = 1
QUERY_PASS_ID = "shifted_prediction"
ATTENTION_METRIC = "attention_absolute_deviation"
ATTENTION_VIEW = "raw"
ATTENTION_EDGE_RATIO = "0.1"
ATTENTION_REGIONS = ("prefix", "interior")
ATTENTION_METHODS = ("vanilla_pic", "kvpacket")
BEHAVIOR_METHODS = ("full_recompute", "no_recompute", "kvpacket")
TARGET_DATASETS = ("biography", "hotpot_qa", "musique", "niah")
CLAIM_ID = "CLAIM-MOTIVATION-BOUNDARY-LOCALITY"
WORK_ITEM_ID = "EXPORT-01"

PLOT_DATA_FIELDS = (
    "schema_version",
    "model_id",
    "dataset_id",
    "query_pass_id",
    "status",
    "missing_reason",
    "full_recompute_status",
    "no_recompute_status",
    "kvpacket_status",
    "full_measurement_id",
    "no_recompute_measurement_id",
    "kvpacket_measurement_id",
    "full_f1",
    "no_recompute_f1",
    "kvpacket_f1",
    "f1_residual_gap",
    "f1_residual_gap_status",
    "f1_recovery_fraction",
    "f1_recovery_status",
    "prefix_vanilla_attention_id",
    "prefix_kvpacket_attention_id",
    "prefix_vanilla_mean",
    "prefix_vanilla_sem",
    "prefix_vanilla_count",
    "prefix_kvpacket_mean",
    "prefix_kvpacket_sem",
    "prefix_kvpacket_count",
    "prefix_attention_ratio",
    "prefix_attention_ratio_status",
    "interior_vanilla_attention_id",
    "interior_kvpacket_attention_id",
    "interior_vanilla_mean",
    "interior_vanilla_sem",
    "interior_vanilla_count",
    "interior_kvpacket_mean",
    "interior_kvpacket_sem",
    "interior_kvpacket_count",
    "interior_attention_ratio",
    "interior_attention_ratio_status",
    "source_manifest",
    "source_summary",
    "known_uncertainty",
)

BUNDLE_FILES = frozenset(
    {
        ".boundary_motivation_bundle.json",
        "attention_measurements.jsonl",
        "behavior_measurements.jsonl",
        "boundary_motivation_diagnostic.pdf",
        "boundary_motivation_diagnostic.png",
        "boundary_motivation_diagnostic.svg",
        "candidate_inventory.csv",
        "completion_report.md",
        "plot_data.csv",
        "schema.md",
        "statistics.json",
        "resolved_configs",
    }
)


@dataclass(frozen=True)
class AuthorityPoint:
    model_id: str
    dataset_id: str
    manifest_path: Path
    summary_path: Path
    method_configs: Mapping[str, Mapping[str, Any]]
    source_configs: Mapping[str, str]
    attention_rows: Mapping[tuple[str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateResult:
    path: Path
    payload: Mapping[str, Any]
    dataset_id: str
    method_key: str


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _display_path(path: str | Path, repo_root: Path) -> str:
    resolved = Path(path).resolve(strict=False)
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _safe_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not segment:
        raise ValueError(f"Unsafe empty path segment derived from {value!r}")
    return segment


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _parse_float(row: Mapping[str, str], field: str, source: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} in {source}: {row.get(field)!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} in {source}: {value!r}")
    return value


def _parse_count(row: Mapping[str, str], source: Path) -> int:
    value = _parse_float(row, "count", source)
    if not value.is_integer() or value <= 0:
        raise ValueError(f"Invalid positive integer count in {source}: {value!r}")
    return int(value)


def extract_authoritative_attention_rows(
    summary_path: str | Path,
) -> tuple[str, dict[tuple[str, str, str], dict[str, Any]]]:
    """Select reducer-authored global bars without rebuilding their aggregation."""

    path = Path(summary_path).resolve()
    selected: dict[tuple[str, str, str], dict[str, Any]] = {}
    model_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model", "dataset", "query_pass", "metric", "view", "method", "facets", "mean", "sem", "count"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"Summary {path} lacks columns: {sorted(missing_columns)}")
        for row in reader:
            try:
                facets = json.loads(row["facets"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid facets JSON in {path}: {row.get('facets')!r}") from error
            if not isinstance(facets, dict):
                raise ValueError(f"Facets must be an object in {path}: {facets!r}")
            if not (
                row["query_pass"] == QUERY_PASS_ID
                and row["metric"] == ATTENTION_METRIC
                and row["view"] == "global_bar"
                and row["method"] in ATTENTION_METHODS
                and facets.get("attention_view") == ATTENTION_VIEW
                and str(facets.get("edge_ratio")) == ATTENTION_EDGE_RATIO
                and facets.get("region") in ATTENTION_REGIONS
            ):
                continue
            key = (row["dataset"], row["method"], facets["region"])
            if key in selected:
                raise ValueError(f"Ambiguous duplicate authoritative row {key} in {path}")
            model_ids.add(row["model"])
            selected[key] = {
                "model_id": row["model"],
                "dataset_id": row["dataset"],
                "query_pass_id": row["query_pass"],
                "metric": row["metric"],
                "view": row["view"],
                "method_key": row["method"],
                "attention_view": facets["attention_view"],
                "edge_ratio": str(facets["edge_ratio"]),
                "region": facets["region"],
                "global_bar_mean": _parse_float(row, "mean", path),
                "global_bar_sem": _parse_float(row, "sem", path),
                "global_bar_count": _parse_count(row, path),
            }

    expected_keys = {
        (dataset, method, region)
        for dataset in TARGET_DATASETS
        for method in ATTENTION_METHODS
        for region in ATTENTION_REGIONS
    }
    if set(selected) != expected_keys:
        missing = sorted(expected_keys - set(selected))
        extra = sorted(set(selected) - expected_keys)
        raise ValueError(f"Incomplete authoritative matrix in {path}; missing={missing}, extra={extra}")
    if len(model_ids) != 1:
        raise ValueError(f"Expected one model identity in {path}, found {sorted(model_ids)}")
    return next(iter(model_ids)), selected


def load_authority_source(manifest_path: str | Path, summary_path: str | Path) -> list[AuthorityPoint]:
    manifest = Path(manifest_path).resolve()
    summary = Path(summary_path).resolve()
    payload = _load_json(manifest)
    query_passes = payload.get("analysis_config", {}).get("query_passes", [])
    if QUERY_PASS_ID not in {item.get("query_pass_id") for item in query_passes if isinstance(item, dict)}:
        raise ValueError(f"Manifest does not declare {QUERY_PASS_ID}: {manifest}")
    model_id, attention_rows = extract_authoritative_attention_rows(summary)

    method_entries: dict[tuple[str, str], tuple[Mapping[str, Any], str]] = {}
    for entry in payload.get("eval_configs", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict):
            continue
        config = entry["config"]
        dataset_id = config.get("dataset", {}).get("dataset_name")
        method_key = config.get("cache_comb", {}).get("method")
        if dataset_id not in TARGET_DATASETS or method_key not in BEHAVIOR_METHODS:
            continue
        key = (dataset_id, method_key)
        if key in method_entries:
            raise ValueError(f"Ambiguous manifest config {key} in {manifest}")
        source_config = entry.get("source_config")
        if not isinstance(source_config, str) or not source_config:
            raise ValueError(f"Manifest config {key} has no source_config in {manifest}")
        method_entries[key] = (config, source_config)

    expected_entries = {
        (dataset, method) for dataset in TARGET_DATASETS for method in BEHAVIOR_METHODS
    }
    if set(method_entries) != expected_entries:
        missing = sorted(expected_entries - set(method_entries))
        extra = sorted(set(method_entries) - expected_entries)
        raise ValueError(f"Incomplete behavior matrix in {manifest}; missing={missing}, extra={extra}")

    for dataset_id in TARGET_DATASETS:
        shared_projections: dict[str, dict[str, Any]] = {}
        manifest_model_ids: set[str] = set()
        for method in BEHAVIOR_METHODS:
            config = method_entries[(dataset_id, method)][0]
            model_path = config.get("model", {}).get("model_path")
            if not isinstance(model_path, str) or not model_path.rstrip("/"):
                raise ValueError(
                    f"Manifest config {(dataset_id, method)} has no model identity in {manifest}"
                )
            manifest_model_ids.add(Path(model_path.rstrip("/")).name)
            shared_config = {
                key: value
                for key, value in config.items()
                if key not in {"cache_comb", "packet_wrapper", "lora"}
            }
            shared_projections[method] = behavioral_projection(
                shared_config, manifest.parent
            )
        if manifest_model_ids != {model_id}:
            raise ValueError(
                f"Summary model {model_id!r} does not match manifest models "
                f"{sorted(manifest_model_ids)} for {dataset_id} in {manifest}"
            )
        reference = shared_projections["full_recompute"]
        for method in ("no_recompute", "kvpacket"):
            if shared_projections[method] != reference:
                raise ValueError(
                    f"Behavior fields differ across methods for {(model_id, dataset_id)}: "
                    f"full_recompute != {method}"
                )

    points: list[AuthorityPoint] = []
    for dataset_id in TARGET_DATASETS:
        point_attention = {
            (method, region): attention_rows[(dataset_id, method, region)]
            for method in ATTENTION_METHODS
            for region in ATTENTION_REGIONS
        }
        points.append(
            AuthorityPoint(
                model_id=model_id,
                dataset_id=dataset_id,
                manifest_path=manifest,
                summary_path=summary,
                method_configs={
                    method: method_entries[(dataset_id, method)][0]
                    for method in BEHAVIOR_METHODS
                },
                source_configs={
                    method: method_entries[(dataset_id, method)][1]
                    for method in BEHAVIOR_METHODS
                },
                attention_rows=point_attention,
            )
        )
    return points


def load_candidate_results(paths: Iterable[str | Path]) -> list[CandidateResult]:
    candidates: list[CandidateResult] = []
    seen: set[Path] = set()
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            expanded.extend(sorted(path.rglob("*_result.json")))
        elif path.is_file():
            expanded.append(path)
        else:
            raise ValueError(f"Candidate result input does not exist: {path}")
    for path in expanded:
        if path in seen:
            raise ValueError(f"Candidate result was supplied more than once: {path}")
        seen.add(path)
        payload = load_completed_result(path)
        config = payload["config"]
        dataset_id = config.get("dataset", {}).get("dataset_name")
        method_key = config.get("cache_comb", {}).get("method")
        if dataset_id not in TARGET_DATASETS or method_key not in BEHAVIOR_METHODS:
            raise ValueError(
                f"Candidate is outside the declared matrix: {path} "
                f"(dataset={dataset_id!r}, method={method_key!r})"
            )
        candidates.append(CandidateResult(path, payload, dataset_id, method_key))
    return candidates


def _artifact_path(config: Mapping[str, Any]) -> str | None:
    method = config["cache_comb"]["method"]
    if method == "kvpacket":
        return config.get("packet_wrapper", {}).get("path")
    return None


def _attention_measurement(
    point: AuthorityPoint,
    method: str,
    region: str,
    repo_root: Path,
) -> dict[str, Any]:
    row = dict(point.attention_rows[(method, region)])
    summary_hash = sha256_file(point.summary_path)
    identity = {
        "kind": "attention_global_bar",
        "source_summary_sha256": summary_hash,
        **row,
    }
    identifier = f"attention-{sha256_value(identity)[:24]}"
    return {
        "measurement_id": identifier,
        "measurement_kind": "attention_global_bar",
        "claim_id": CLAIM_ID,
        "work_item_id": WORK_ITEM_ID,
        **row,
        "source_summary": _display_path(point.summary_path, repo_root),
        "source_summary_sha256": summary_hash,
        "source_manifest": _display_path(point.manifest_path, repo_root),
        "aggregation_rule": "authoritative reducer global_bar row; not rebuilt",
        "unit": "model-dataset single run",
        "known_uncertainty": "SEM is within-run sample variability, not training-seed uncertainty.",
    }


def _behavior_metrics(
    full_f1: float | None,
    no_recompute_f1: float | None,
    kvpacket_f1: float | None,
) -> dict[str, Any]:
    gap: float | None = None
    gap_status = "missing_input"
    if full_f1 is not None and kvpacket_f1 is not None:
        gap = full_f1 - kvpacket_f1
        gap_status = "defined"

    recovery: float | None = None
    recovery_status = "missing_input"
    if full_f1 is not None and no_recompute_f1 is not None and kvpacket_f1 is not None:
        denominator = full_f1 - no_recompute_f1
        if denominator <= 0:
            recovery_status = "nonpositive_denominator"
        else:
            recovery = (kvpacket_f1 - no_recompute_f1) / denominator
            recovery_status = "defined"
    return {
        "f1_residual_gap": gap,
        "f1_residual_gap_status": gap_status,
        "f1_recovery_fraction": recovery,
        "f1_recovery_status": recovery_status,
    }


def _attention_ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator <= 0:
        return None, "nonpositive_denominator"
    return numerator / denominator, "defined"


def build_motivation_data(
    sources: Sequence[tuple[str | Path, str | Path]],
    candidate_paths: Iterable[str | Path],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build all machine-readable bundle records without writing files."""

    root = Path(repo_root).resolve()
    points = [
        point
        for manifest_path, summary_path in sources
        for point in load_authority_source(manifest_path, summary_path)
    ]
    point_keys = [(point.model_id, point.dataset_id) for point in points]
    if len(points) != 8 or len(set(point_keys)) != 8:
        raise ValueError(f"Expected eight unique model-dataset points, found {point_keys}")
    candidates = load_candidate_results(candidate_paths)

    behavior_measurements: list[dict[str, Any]] = []
    attention_measurements: list[dict[str, Any]] = []
    candidate_inventory: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []

    for point in points:
        method_values: dict[str, float | None] = {}
        method_statuses: dict[str, str] = {}
        method_ids: dict[str, str] = {}
        for method in BEHAVIOR_METHODS:
            expected_config = point.method_configs[method]
            routed = [
                candidate
                for candidate in candidates
                if candidate.dataset_id == point.dataset_id and candidate.method_key == method
            ]
            exact: list[CandidateResult] = []
            for candidate in routed:
                comparison = compare_behavioral_configs(
                    expected_config,
                    candidate.payload["config"],
                    root,
                )
                candidate_inventory.append(
                    {
                        "model_id": point.model_id,
                        "dataset_id": point.dataset_id,
                        "method_key": method,
                        "candidate_result_path": _display_path(candidate.path, root),
                        "status": comparison.status,
                        "differences": differences_json(comparison.differences),
                    }
                )
                if comparison.matched:
                    exact.append(candidate)
            if not exact:
                method_values[method] = None
                method_ids[method] = ""
                method_statuses[method] = "incompatible" if routed else "missing"
                continue

            exact_groups: dict[str, list[CandidateResult]] = {}
            for candidate in exact:
                semantic_payload = {
                    "config": behavioral_projection(candidate.payload["config"], root),
                    "result": candidate.payload["result"],
                }
                exact_groups.setdefault(sha256_value(semantic_payload), []).append(candidate)
            if len(exact_groups) > 1:
                paths = [_display_path(candidate.path, root) for candidate in exact]
                raise ValueError(
                    f"Ambiguous exact results for {(point.model_id, point.dataset_id, method)}: {paths}"
                )

            aliases = next(iter(exact_groups.values()))
            aliases.sort(
                key=lambda item: (
                    "eval_outputs" not in item.path.parts,
                    _display_path(item.path, root),
                )
            )
            candidate = aliases[0]
            config_projection = behavioral_projection(expected_config, root)
            config_hash = sha256_value(config_projection)
            result_hash = next(iter(exact_groups))
            identifier = measurement_id(
                method_key=method,
                config_projection_sha256=config_hash,
                result_payload_sha256=result_hash,
            )
            f1 = float(candidate.payload["result"]["f1"])
            method_values[method] = f1
            method_ids[method] = identifier
            method_statuses[method] = "matched"
            behavior_measurements.append(
                {
                    "measurement_id": identifier,
                    "measurement_kind": "behavior_f1",
                    "claim_id": CLAIM_ID,
                    "work_item_id": WORK_ITEM_ID,
                    "model_id": point.model_id,
                    "dataset_id": point.dataset_id,
                    "method_key": method,
                    "quality_metric_name": "corpus_whitespace_token_micro_f1",
                    "quality_metric_value": f1,
                    "result_path": _display_path(candidate.path, root),
                    "result_path_aliases": [
                        _display_path(alias.path, root) for alias in aliases[1:]
                    ],
                    "result_payload_sha256": result_hash,
                    "result_file_sha256": sha256_file(candidate.path),
                    "hash_observation_time": "export time; not historical run time",
                    "source_config": point.source_configs[method],
                    "config_projection_sha256": config_hash,
                    "checkpoint_or_artifact": _artifact_path(expected_config),
                    "sample_count": expected_config["dataset"].get("num_samples"),
                    "seed": expected_config["dataset"].get("seed"),
                    "metric_definition": "run_eval.py corpus whitespace-token micro-F1",
                    "sample_membership": "config-implied only; historical behavior result has no membership digest",
                    "known_uncertainty": "Single completed run; no training-seed interval.",
                }
            )

        behavior = _behavior_metrics(
            method_values["full_recompute"],
            method_values["no_recompute"],
            method_values["kvpacket"],
        )
        missing_methods = [
            method for method in BEHAVIOR_METHODS if method_statuses[method] != "matched"
        ]
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_id": point.model_id,
            "dataset_id": point.dataset_id,
            "query_pass_id": QUERY_PASS_ID,
            "status": "pass" if not missing_methods else "blocked",
            "missing_reason": (
                "" if not missing_methods else "No exact behavior result for: " + ", ".join(missing_methods)
            ),
            "full_recompute_status": method_statuses["full_recompute"],
            "no_recompute_status": method_statuses["no_recompute"],
            "kvpacket_status": method_statuses["kvpacket"],
            "full_measurement_id": method_ids["full_recompute"],
            "no_recompute_measurement_id": method_ids["no_recompute"],
            "kvpacket_measurement_id": method_ids["kvpacket"],
            "full_f1": method_values["full_recompute"],
            "no_recompute_f1": method_values["no_recompute"],
            "kvpacket_f1": method_values["kvpacket"],
            **behavior,
            "source_manifest": _display_path(point.manifest_path, root),
            "source_summary": _display_path(point.summary_path, root),
            "known_uncertainty": (
                "Single-run descriptive evidence; behavior membership is config-implied because historical "
                "result payloads do not serialize sample IDs."
            ),
        }

        for region in ATTENTION_REGIONS:
            records = {
                method: _attention_measurement(point, method, region, root)
                for method in ATTENTION_METHODS
            }
            attention_measurements.extend(records.values())
            vanilla = records["vanilla_pic"]
            kvpacket = records["kvpacket"]
            ratio, ratio_status = _attention_ratio(
                kvpacket["global_bar_mean"], vanilla["global_bar_mean"]
            )
            row.update(
                {
                    f"{region}_vanilla_attention_id": vanilla["measurement_id"],
                    f"{region}_kvpacket_attention_id": kvpacket["measurement_id"],
                    f"{region}_vanilla_mean": vanilla["global_bar_mean"],
                    f"{region}_vanilla_sem": vanilla["global_bar_sem"],
                    f"{region}_vanilla_count": vanilla["global_bar_count"],
                    f"{region}_kvpacket_mean": kvpacket["global_bar_mean"],
                    f"{region}_kvpacket_sem": kvpacket["global_bar_sem"],
                    f"{region}_kvpacket_count": kvpacket["global_bar_count"],
                    f"{region}_attention_ratio": ratio,
                    f"{region}_attention_ratio_status": ratio_status,
                }
            )
        plot_rows.append(row)

    plot_rows.sort(key=lambda row: (row["dataset_id"], row["model_id"]))
    behavior_measurements.sort(
        key=lambda row: (row["dataset_id"], row["model_id"], row["method_key"])
    )
    attention_measurements.sort(
        key=lambda row: (
            row["dataset_id"], row["model_id"], row["region"], row["method_key"]
        )
    )
    candidate_inventory.sort(
        key=lambda row: (
            row["dataset_id"], row["model_id"], row["method_key"], row["candidate_result_path"]
        )
    )
    return {
        "points": points,
        "plot_rows": plot_rows,
        "behavior_measurements": behavior_measurements,
        "attention_measurements": attention_measurements,
        "candidate_inventory": candidate_inventory,
    }


def materialize_pinned_configs(
    points: Sequence[AuthorityPoint],
    output_dir: str | Path,
) -> dict[tuple[str, str, str], Path]:
    """Write the 16 frozen Full/No-Recompute configs without default inheritance."""

    directory = Path(output_dir).resolve()
    written: dict[tuple[str, str, str], Path] = {}
    for point in points:
        for method in ("full_recompute", "no_recompute"):
            path = (
                directory
                / _safe_segment(point.model_id)
                / _safe_segment(point.dataset_id)
                / f"{method}.json"
            )
            config = point.method_configs[method]
            if path.exists():
                if _load_json(path) != config:
                    raise FileExistsError(f"Refusing to replace a different pinned config: {path}")
            else:
                _atomic_write_json(path, config)
            written[(point.model_id, point.dataset_id, method)] = path
    if len(written) != 16:
        raise ValueError(f"Expected to materialize 16 pinned configs, found {len(written)}")
    return written


def record_bundle_files(
    output_dir: str | Path,
    paths: Iterable[str | Path],
) -> None:
    """Add generated files inside a bundle to its marker inventory."""

    directory = Path(output_dir).resolve()
    marker = directory / ".boundary_motivation_bundle.json"
    payload = _load_json(marker)
    if payload.get("schema_name") != "sempic.boundary_motivation_bundle":
        raise ValueError(f"Unrecognized bundle marker: {marker}")
    generated = set(payload.get("generated_files", []))
    for path in paths:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(directory)
        except ValueError as error:
            raise ValueError(f"Generated file is outside bundle: {resolved}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Generated bundle file does not exist: {resolved}")
        generated.add(relative.as_posix())
    payload["generated_files"] = sorted(generated)
    _atomic_write_json(marker, payload)


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fieldnames})
    return buffer.getvalue()


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def _descriptive_statistics(plot_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "f1_recovery_fraction",
        "f1_residual_gap",
        "prefix_attention_ratio",
        "interior_attention_ratio",
    )
    summaries: dict[str, Any] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in plot_rows if row.get(metric) is not None]
        summaries[metric] = {
            "unweighted_mean": fmean(values) if values else None,
            "valid_point_count": len(values),
            "total_point_count": len(plot_rows),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": "model-dataset configuration",
        "scope": "unweighted descriptive means; not seed-level inference",
        "metrics": summaries,
    }


def _schema_markdown() -> str:
    return """# Boundary Motivation Evidence Schema

`plot_data.csv` contains exactly one row per model--dataset identity. `status=pass`
requires exact matched Full Recompute, No Recompute, and KV Packet behavior
measurements; blocked rows remain present and carry a `missing_reason`.

Behavior values reference method-specific records in `behavior_measurements.jsonl`.
Attention values reference reducer-authored records in
`attention_measurements.jsonl`. The attention extractor uses only
`shifted_prediction`, `attention_absolute_deviation`, `global_bar`, raw attention,
edge ratio 0.1, and the prefix/interior facets; it never rebuilds regions from
heatmap rows.

Recovery is `(KVPacket - NoRecompute) / (Full - NoRecompute)` and is null when
an input is absent or the denominator is nonpositive. Residual gap is
`Full - KVPacket`. Attention ratios are `KV Packet / Vanilla PIC` and are null
when the Vanilla PIC denominator is nonpositive. Values are never clipped.

Hashes are computed at export time. Historical sample-membership and artifact
content hashes are not inferred when the original result did not record them.
"""


def _completion_report(plot_rows: Sequence[Mapping[str, Any]]) -> str:
    passed = sum(row["status"] == "pass" for row in plot_rows)
    overall = "complete" if passed == len(plot_rows) else "partial"
    lines = [
        "# Boundary Motivation Completion Report",
        "",
        f"- Overall status: {overall}",
        f"- EXPORT-01: {'pass' if passed == len(plot_rows) else 'blocked'} ({passed}/{len(plot_rows)} points complete)",
        "- STAT-01: pass for available values; descriptive only",
        f"- VIS-01: {'pass-ready' if passed == len(plot_rows) else 'blocked pending matched behavior results'}",
        "- EXP-01: not-run",
        "- Sample membership: config-implied; historical behavior result payloads do not contain sample IDs",
        "",
        "## Point status",
        "",
    ]
    for row in plot_rows:
        detail = row["missing_reason"] or "all three behavior methods matched"
        lines.append(f"- {row['model_id']} / {row['dataset_id']}: {row['status']} — {detail}")
    return "\n".join(lines) + "\n"


def prepare_bundle_directory(output_dir: str | Path, overwrite: bool = False) -> Path:
    directory = Path(output_dir).resolve()
    if not directory.exists():
        directory.mkdir(parents=True)
        return directory
    entries = {entry.name for entry in directory.iterdir()}
    if not entries:
        return directory
    if not overwrite:
        raise FileExistsError(f"Refusing to write into nonempty bundle directory: {directory}")
    marker = directory / ".boundary_motivation_bundle.json"
    if not marker.is_file():
        raise FileExistsError(f"Refusing overwrite without bundle marker: {directory}")
    marker_payload = _load_json(marker)
    if marker_payload.get("schema_name") != "sempic.boundary_motivation_bundle":
        raise FileExistsError(f"Unrecognized bundle marker: {marker}")
    unexpected = entries - BUNDLE_FILES
    if unexpected:
        raise FileExistsError(f"Refusing overwrite with unexpected files: {sorted(unexpected)}")
    return directory


def write_bundle(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    directory = prepare_bundle_directory(output_dir, overwrite=overwrite)
    plot_rows = data["plot_rows"]
    candidate_fields = (
        "model_id",
        "dataset_id",
        "method_key",
        "candidate_result_path",
        "status",
        "differences",
    )
    outputs = {
        "plot_data": directory / "plot_data.csv",
        "behavior_measurements": directory / "behavior_measurements.jsonl",
        "attention_measurements": directory / "attention_measurements.jsonl",
        "candidate_inventory": directory / "candidate_inventory.csv",
        "statistics": directory / "statistics.json",
        "schema": directory / "schema.md",
        "completion_report": directory / "completion_report.md",
        "marker": directory / ".boundary_motivation_bundle.json",
    }
    _atomic_write_text(outputs["plot_data"], _csv_text(plot_rows, PLOT_DATA_FIELDS))
    _atomic_write_text(
        outputs["behavior_measurements"], _jsonl_text(data["behavior_measurements"])
    )
    _atomic_write_text(
        outputs["attention_measurements"], _jsonl_text(data["attention_measurements"])
    )
    _atomic_write_text(
        outputs["candidate_inventory"],
        _csv_text(data["candidate_inventory"], candidate_fields),
    )
    _atomic_write_json(outputs["statistics"], _descriptive_statistics(plot_rows))
    _atomic_write_text(outputs["schema"], _schema_markdown())
    _atomic_write_text(outputs["completion_report"], _completion_report(plot_rows))
    _atomic_write_json(
        outputs["marker"],
        {
            "schema_name": "sempic.boundary_motivation_bundle",
            "schema_version": SCHEMA_VERSION,
            "generated_files": sorted(BUNDLE_FILES - {"resolved_configs"}),
        },
    )
    return outputs
