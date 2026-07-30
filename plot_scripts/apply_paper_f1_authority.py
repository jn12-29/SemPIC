"""Apply the explicit paper F1 authority to the two paper evidence bundles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


AUTHORITY_FIELDS = (
    "schema_version",
    "model_id",
    "bundle_model_id",
    "dataset_id",
    "full_f1",
    "no_cache_f1",
    "no_recompute_f1",
    "kvpacket_f1",
    "sempic_f1",
    "joint_f1",
)
F1_FIELDS = AUTHORITY_FIELDS[4:]
MODEL_IDENTITIES = {
    "Qwen3-4B": "Qwen3-4B-Instruct-2507",
    "Qwen3-8B": "Qwen3-8B",
    "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
}
DATASET_IDENTITIES = {
    "Biography": "biography",
    "HotpotQA": "hotpot_qa",
    "MuSiQue": "musique",
    "NIAH": "niah",
}
BOUNDARY_METHOD_FIELDS = {
    "full_recompute": "full_f1",
    "no_recompute": "no_recompute_f1",
    "kvpacket": "kvpacket_f1",
}
ATTENTION_METHOD_FIELDS = {
    "full_recompute": "full_f1",
    "vanilla_pic": "no_recompute_f1",
    "sempic": "sempic_f1",
}
AUTHORITY_ROLE = "sole_paper_f1_authority"
HISTORICAL_ROLE = "historical_eval_config_match_only"


@dataclass(frozen=True)
class AuthorityEntry:
    schema_version: int
    model_id: str
    bundle_model_id: str
    dataset_id: str
    bundle_dataset_id: str
    values: Mapping[str, float]

    def value_for(self, method_key: str, mapping: Mapping[str, str]) -> float:
        try:
            return self.values[mapping[method_key]]
        except KeyError as error:
            raise ValueError(f"Unsupported F1 method: {method_key}") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSON {path}: {error}") from error


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load JSONL {path}: {error}") from error
    return records


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or ())
            return fieldnames, [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ValueError(f"Cannot load CSV {path}: {error}") from error


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        for record in records
    )


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _with_fields(fieldnames: Sequence[str], *extra: str) -> list[str]:
    result = list(fieldnames)
    result.extend(field for field in extra if field not in result)
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_authority(path: str | Path) -> dict[tuple[str, str], AuthorityEntry]:
    source = Path(path)
    fieldnames, rows = _load_csv(source)
    if tuple(fieldnames) != AUTHORITY_FIELDS:
        raise ValueError(
            f"Authority columns must exactly equal {list(AUTHORITY_FIELDS)}, found {fieldnames}"
        )
    expected_row_count = len(MODEL_IDENTITIES) * len(DATASET_IDENTITIES)
    if len(rows) != expected_row_count:
        raise ValueError(
            f"Authority must contain exactly {expected_row_count} rows, found {len(rows)}"
        )

    entries: dict[tuple[str, str], AuthorityEntry] = {}
    for line_number, row in enumerate(rows, start=2):
        try:
            schema_version = int(row["schema_version"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Authority line {line_number} has invalid schema_version") from error
        if schema_version != 1 or str(schema_version) != row["schema_version"]:
            raise ValueError(f"Authority line {line_number} requires schema_version=1")

        model_id = row["model_id"]
        bundle_model_id = row["bundle_model_id"]
        dataset_id = row["dataset_id"]
        if model_id not in MODEL_IDENTITIES:
            raise ValueError(f"Authority line {line_number} has unknown model_id={model_id!r}")
        if MODEL_IDENTITIES[model_id] != bundle_model_id:
            raise ValueError(
                f"Authority line {line_number} has invalid bundle_model_id={bundle_model_id!r}"
            )
        if dataset_id not in DATASET_IDENTITIES:
            raise ValueError(f"Authority line {line_number} has unknown dataset_id={dataset_id!r}")
        key = (bundle_model_id, DATASET_IDENTITIES[dataset_id])
        if key in entries:
            raise ValueError(f"Duplicate authority identity: {key}")

        values: dict[str, float] = {}
        for field in F1_FIELDS:
            try:
                value = float(row[field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Authority line {line_number} has invalid {field}={row.get(field)!r}"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Authority line {line_number} requires finite {field} in [0,1]"
                )
            values[field] = value
        entries[key] = AuthorityEntry(
            schema_version=schema_version,
            model_id=model_id,
            bundle_model_id=bundle_model_id,
            dataset_id=dataset_id,
            bundle_dataset_id=DATASET_IDENTITIES[dataset_id],
            values=values,
        )

    expected = {
        (bundle_model_id, bundle_dataset_id)
        for bundle_model_id in MODEL_IDENTITIES.values()
        for bundle_dataset_id in DATASET_IDENTITIES.values()
    }
    missing = expected - set(entries)
    extra = set(entries) - expected
    if missing or extra:
        raise ValueError(f"Authority matrix mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return entries


def _authority_metadata(path: Path, digest: str, entry: AuthorityEntry, field: str) -> dict[str, Any]:
    return {
        "f1_authority_path": str(path.resolve()),
        "f1_authority_sha256": digest,
        "f1_authority_schema_version": entry.schema_version,
        "f1_authority_role": AUTHORITY_ROLE,
        "f1_authority_row": (
            f"{entry.model_id}/{entry.bundle_model_id}/{entry.dataset_id}/{field}"
        ),
    }


def _measurement_id(
    *, digest: str, entry: AuthorityEntry, method_key: str, value: float
) -> str:
    identity = json.dumps(
        {
            "authority_sha256": digest,
            "model_id": entry.bundle_model_id,
            "dataset_id": entry.bundle_dataset_id,
            "method_key": method_key,
            "value": value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "behavior-authority-" + hashlib.sha256(identity.encode()).hexdigest()[:24]


def _ratio(numerator: float, denominator: float, *, undefined: str) -> tuple[float | None, str]:
    if denominator <= 0:
        return None, undefined
    return numerator / denominator, "defined"


def _require_files(directory: Path, relative_paths: Sequence[str]) -> dict[str, Path]:
    paths = {relative: directory / relative for relative in relative_paths}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Bundle is missing required files: {missing}")
    return paths


def _boundary_updates(
    *,
    bundle: Path,
    authority_path: Path,
    authority_digest: str,
    authority: Mapping[tuple[str, str], AuthorityEntry],
) -> dict[Path, str]:
    paths = _require_files(
        bundle,
        (
            "plot_data.csv",
            "behavior_measurements.jsonl",
            "statistics.json",
            ".boundary_motivation_bundle.json",
        ),
    )
    plot_fields, plot_rows = _load_csv(paths["plot_data.csv"])
    if len(plot_rows) != 8:
        raise ValueError(f"Boundary plot_data.csv must contain 8 rows, found {len(plot_rows)}")
    plot_identities = [(row.get("model_id"), row.get("dataset_id")) for row in plot_rows]
    if len(set(plot_identities)) != len(plot_identities) or set(plot_identities) != set(authority):
        raise ValueError("Boundary plot identity matrix does not match the authority matrix")
    immutable_plot = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
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
                "known_uncertainty",
            }
        }
        for row in plot_rows
    ]

    behavior_records = _load_jsonl(paths["behavior_measurements.jsonl"])
    if len(behavior_records) != 24:
        raise ValueError(
            f"Boundary behavior_measurements.jsonl must contain 24 rows, found {len(behavior_records)}"
        )
    behavior_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in behavior_records:
        key = (record.get("model_id"), record.get("dataset_id"), record.get("method_key"))
        if key in behavior_index:
            raise ValueError(f"Duplicate boundary behavior measurement: {key}")
        behavior_index[key] = record

    expected_behavior = {
        (model_id, dataset_id, method)
        for model_id, dataset_id in authority
        for method in BOUNDARY_METHOD_FIELDS
    }
    if set(behavior_index) != expected_behavior:
        raise ValueError("Boundary behavior matrix does not match the authority matrix")

    for row in plot_rows:
        key = (row.get("model_id"), row.get("dataset_id"))
        if key not in authority:
            raise ValueError(f"Unknown boundary plot identity: {key}")
        entry = authority[key]
        values = {
            method: entry.value_for(method, BOUNDARY_METHOD_FIELDS)
            for method in BOUNDARY_METHOD_FIELDS
        }
        ids: dict[str, str] = {}
        for method, value in values.items():
            record = behavior_index[(key[0], key[1], method)]
            record.setdefault("historical_quality_metric_value", record["quality_metric_value"])
            record.setdefault("historical_sample_count", record.get("sample_count"))
            record.setdefault("historical_seed", record.get("seed"))
            record.setdefault(
                "historical_known_uncertainty", record.get("known_uncertainty")
            )
            record["quality_metric_value"] = value
            record["sample_count"] = None
            record["seed"] = None
            record["sample_scope_role"] = HISTORICAL_ROLE
            record["source_config_role"] = HISTORICAL_ROLE
            record["known_uncertainty"] = (
                "User-supplied final table; corrected F1 raw-run linkage is not recorded."
            )
            record["metric_definition"] = (
                "Paper F1 from explicit corrected authority; historical eval JSON is "
                "retained for configuration matching only"
            )
            record["result_path_role"] = HISTORICAL_ROLE
            record["result_payload_sha256_role"] = HISTORICAL_ROLE
            record["result_file_sha256_role"] = HISTORICAL_ROLE
            field = BOUNDARY_METHOD_FIELDS[method]
            record.update(_authority_metadata(authority_path, authority_digest, entry, field))
            identifier = _measurement_id(
                digest=authority_digest, entry=entry, method_key=method, value=value
            )
            record["measurement_id"] = identifier
            ids[method] = identifier

        full = values["full_recompute"]
        baseline = values["no_recompute"]
        kvpacket = values["kvpacket"]
        recovery, recovery_status = _ratio(
            kvpacket - baseline,
            full - baseline,
            undefined="nonpositive_denominator",
        )
        row.update(
            {
                "full_measurement_id": ids["full_recompute"],
                "no_recompute_measurement_id": ids["no_recompute"],
                "kvpacket_measurement_id": ids["kvpacket"],
                "full_f1": full,
                "no_recompute_f1": baseline,
                "kvpacket_f1": kvpacket,
                "f1_residual_gap": full - kvpacket,
                "f1_residual_gap_status": "defined",
                "f1_recovery_fraction": recovery,
                "f1_recovery_status": recovery_status,
                "known_uncertainty": (
                    "Attention sample scope is recorded by the attention sources; corrected "
                    "F1 raw-run linkage is not recorded."
                ),
            }
        )

    if immutable_plot != [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
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
                "known_uncertainty",
            }
        }
        for row in plot_rows
    ]:
        raise AssertionError("Boundary non-F1 plot fields changed")

    statistics = _load_json(paths["statistics.json"])
    if not isinstance(statistics, dict) or not isinstance(statistics.get("metrics"), dict):
        raise ValueError("Boundary statistics.json has an invalid shape")
    for metric in ("f1_recovery_fraction", "f1_residual_gap"):
        values = [float(row[metric]) for row in plot_rows if row.get(metric) not in (None, "")]
        statistics["metrics"][metric] = {
            "unweighted_mean": fmean(values) if values else None,
            "valid_point_count": len(values),
            "total_point_count": len(plot_rows),
        }

    marker = _load_json(paths[".boundary_motivation_bundle.json"])
    if not isinstance(marker, dict):
        raise ValueError("Boundary bundle marker must be an object")
    marker["f1_authority_input"] = {
        "path": str(authority_path.resolve()),
        "sha256": authority_digest,
        "schema_version": 1,
        "role": AUTHORITY_ROLE,
    }
    return {
        paths["plot_data.csv"]: _csv_text(plot_fields, plot_rows),
        paths["behavior_measurements.jsonl"]: _jsonl_text(behavior_records),
        paths["statistics.json"]: _json_text(statistics),
        paths[".boundary_motivation_bundle.json"]: _json_text(marker),
    }


def _attention_immutable(point: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(
        {
            "profiles": point.get("profiles"),
            "regions": point.get("regions"),
            "interior_attention_errors": point.get("interior_attention_errors"),
            "relative_interior_attention_errors": point.get(
                "relative_interior_attention_errors"
            ),
            "sink_ratio": point.get("sink_ratio"),
            "sink_ratio_status": point.get("sink_ratio_status"),
            "sink_ratio_measurement_id": point.get("sink_ratio_measurement_id"),
            "region_rule": point.get("region_rule"),
        }
    )


def _update_attention_csv(
    *,
    path: Path,
    authority_path: Path,
    authority_digest: str,
    entries: Mapping[tuple[str, str], AuthorityEntry],
    value_field: str,
    result_path_field: str,
) -> str:
    fieldnames, rows = _load_csv(path)
    result_role_field = f"{result_path_field}_role"
    result_sha_field = (
        "result_sha256" if result_path_field == "result_path" else "quality_result_sha256"
    )
    result_sha_role_field = f"{result_sha_field}_role"
    extras = (
        "historical_quality_metric_value",
        result_role_field,
        result_sha_role_field,
        "f1_authority_path",
        "f1_authority_sha256",
        "f1_authority_schema_version",
        "f1_authority_role",
        "f1_authority_row",
    )
    if path.name == "behavior_measurements.csv":
        extras += (
            "historical_sample_count",
            "historical_eval_seed",
            "historical_dataset_seed",
            "sample_scope_role",
            "seed_scope_role",
            "known_uncertainty",
        )
    for row in rows:
        key = (row.get("model_id"), row.get("dataset_id"))
        method = row.get("method_key")
        if key not in entries or method not in ATTENTION_METHOD_FIELDS:
            raise ValueError(f"Unknown attention quality identity: {key + (method,)}")
        entry = entries[key]
        field = ATTENTION_METHOD_FIELDS[method]
        value = entry.value_for(method, ATTENTION_METHOD_FIELDS)
        row.setdefault("historical_quality_metric_value", row[value_field])
        row[value_field] = value
        row[result_role_field] = HISTORICAL_ROLE
        row[result_sha_role_field] = HISTORICAL_ROLE
        row.update(_authority_metadata(authority_path, authority_digest, entry, field))
        if path.name == "behavior_measurements.csv":
            row.setdefault("historical_sample_count", row.get("sample_count"))
            row.setdefault("historical_eval_seed", row.get("eval_seed"))
            row.setdefault("historical_dataset_seed", row.get("dataset_seed"))
            if "sample_count" in row:
                row["sample_count"] = ""
            if "eval_seed" in row:
                row["eval_seed"] = ""
            if "dataset_seed" in row:
                row["dataset_seed"] = ""
            row["sample_scope_role"] = HISTORICAL_ROLE
            row["seed_scope_role"] = HISTORICAL_ROLE
            row["known_uncertainty"] = (
                "User-supplied final table; corrected F1 raw-run linkage is not recorded."
            )
        if not row.get(result_path_field):
            raise ValueError(f"Attention quality row lacks {result_path_field}: {key + (method,)}")
        if not row.get(result_sha_field):
            raise ValueError(f"Attention quality row lacks {result_sha_field}: {key + (method,)}")
    return _csv_text(_with_fields(fieldnames, *extras), rows)


def _provenance_identity(record: Mapping[str, Any]) -> tuple[str, str] | None:
    identifier = record.get("measurement_id")
    if not isinstance(identifier, str):
        return None
    parts = identifier.split(":")
    if parts[0] in {"behavior", "position", "region"} and len(parts) >= 3:
        return parts[1], parts[2]
    return None


def _attention_updates(
    *,
    bundle: Path,
    authority_path: Path,
    authority_digest: str,
    authority: Mapping[tuple[str, str], AuthorityEntry],
) -> dict[Path, str]:
    paths = _require_files(
        bundle,
        (
            "data/plot_data.json",
            "data/behavior_measurements.csv",
            "data/position_measurements.csv",
            "data/region_measurements.csv",
            "data/provenance.jsonl",
            "data/statistics.json",
            "data/resolved_configs.json",
            "bundle_manifest.json",
        ),
    )
    model_id = "Qwen3-4B-Instruct-2507"
    entries = {key: value for key, value in authority.items() if key[0] == model_id}
    if len(entries) != 4:
        raise ValueError("Attention authority must contain four Qwen3-4B rows")

    plot_data = _load_json(paths["data/plot_data.json"])
    if not isinstance(plot_data, dict) or not isinstance(plot_data.get("points"), list):
        raise ValueError("Attention plot_data.json has an invalid shape")
    if plot_data.get("model_id") != model_id or len(plot_data["points"]) != 4:
        raise ValueError("Attention plot data must contain the four Qwen3-4B identities")
    immutable = {
        point.get("dataset_id"): _attention_immutable(point)
        for point in plot_data["points"]
    }
    for point in plot_data["points"]:
        dataset_id = point.get("dataset_id")
        key = (model_id, dataset_id)
        if key not in entries:
            raise ValueError(f"Unknown attention plot identity: {key}")
        entry = entries[key]
        behavior = point.get("behavior")
        if not isinstance(behavior, list) or {
            row.get("method_key") for row in behavior if isinstance(row, dict)
        } != set(ATTENTION_METHOD_FIELDS):
            raise ValueError(f"Attention behavior matrix is incomplete for {dataset_id}")
        values: dict[str, float] = {}
        for row in behavior:
            method = row["method_key"]
            field = ATTENTION_METHOD_FIELDS[method]
            value = entry.value_for(method, ATTENTION_METHOD_FIELDS)
            row.setdefault("historical_f1", row["f1"])
            row["f1"] = value
            row["result_path_role"] = HISTORICAL_ROLE
            row.update(_authority_metadata(authority_path, authority_digest, entry, field))
            values[method] = value

        recovery, recovery_status = _ratio(
            values["sempic"] - values["vanilla_pic"],
            values["full_recompute"] - values["vanilla_pic"],
            undefined="undefined_nonpositive_denominator",
        )
        point["recovery_fraction"] = recovery
        point["recovery_fraction_status"] = recovery_status
        point["f1_change"] = values["sempic"] - values["vanilla_pic"]
        point["f1_change_status"] = "defined"
        point["f1_change_measurement_id"] = f"estimand:{dataset_id}:f1_change"
        conditions = {
            "f1_sempic_greater_than_vanilla": values["sempic"] > values["vanilla_pic"],
            "sink_ratio_greater_than_one": (
                point["sink_ratio"] > 1 if point.get("sink_ratio") is not None else None
            ),
        }
        point["coexistence_conditions"] = conditions
        point["interpretation_status"] = (
            "supports_coexistence"
            if all(value is True for value in conditions.values())
            else "does_not_support_coexistence"
        )
        point["status_reason"] = (
            "Complete matched raw-profile evidence with explicit paper F1 authority."
        )

    if immutable != {
        point.get("dataset_id"): _attention_immutable(point)
        for point in plot_data["points"]
    }:
        raise AssertionError("Attention profiles, regions, interior errors, R, or S changed")

    behavior_csv = _update_attention_csv(
        path=paths["data/behavior_measurements.csv"],
        authority_path=authority_path,
        authority_digest=authority_digest,
        entries=entries,
        value_field="quality_metric_value",
        result_path_field="result_path",
    )
    position_csv = _update_attention_csv(
        path=paths["data/position_measurements.csv"],
        authority_path=authority_path,
        authority_digest=authority_digest,
        entries=entries,
        value_field="quality_metric_value",
        result_path_field="quality_result_path",
    )
    region_csv = _update_attention_csv(
        path=paths["data/region_measurements.csv"],
        authority_path=authority_path,
        authority_digest=authority_digest,
        entries=entries,
        value_field="quality_metric_value",
        result_path_field="quality_result_path",
    )

    provenance = [
        record
        for record in _load_jsonl(paths["data/provenance.jsonl"])
        if not str(record.get("measurement_id", "")).endswith(":f1_change")
    ]
    recovery_provenance: dict[str, dict[str, Any]] = {}
    for record in provenance:
        identifier = record.get("measurement_id", "")
        identity = _provenance_identity(record)
        if identity is not None:
            dataset_id, method = identity
            key = (model_id, dataset_id)
            if key not in entries or method not in ATTENTION_METHOD_FIELDS:
                continue
            entry = entries[key]
            field = ATTENTION_METHOD_FIELDS[method]
            value = entry.value_for(method, ATTENTION_METHOD_FIELDS)
            record.update(_authority_metadata(authority_path, authority_digest, entry, field))
            if identifier.startswith("behavior:"):
                record.setdefault(
                    "historical_sample_or_repeat_count",
                    record.get("sample_or_repeat_count"),
                )
                record.setdefault(
                    "historical_seed_or_seed_list", record.get("seed_or_seed_list")
                )
                record["sample_or_repeat_count"] = None
                record["seed_or_seed_list"] = None
                record["sample_scope_role"] = HISTORICAL_ROLE
                record["seed_scope_role"] = HISTORICAL_ROLE
                record["known_uncertainty"] = (
                    "User-supplied final table; corrected F1 raw-run linkage is not recorded."
                )
                record["value"] = value
                record["metric_definition"] = (
                    "Paper F1 from explicit corrected authority; historical eval JSON is "
                    "retained for configuration matching only"
                )
                record["result_path_role"] = HISTORICAL_ROLE
                record["source_artifact_sha256_role"] = HISTORICAL_ROLE
                record["paired_quality_result_path_role"] = HISTORICAL_ROLE
                record["paired_quality_result_sha256_role"] = HISTORICAL_ROLE
            else:
                record["paired_quality_metric_value"] = value
                record["paired_quality_result_path_role"] = HISTORICAL_ROLE
                record["paired_quality_result_sha256_role"] = HISTORICAL_ROLE
            continue

        parts = identifier.split(":") if isinstance(identifier, str) else []
        if len(parts) == 3 and parts[0] == "estimand" and parts[1] in DATASET_IDENTITIES.values():
            key = (model_id, parts[1])
            entry = entries[key]
            if parts[2] == "recovery_fraction":
                full = entry.values["full_f1"]
                baseline = entry.values["no_recompute_f1"]
                sempic = entry.values["sempic_f1"]
                value, status = _ratio(
                    sempic - baseline,
                    full - baseline,
                    undefined="undefined_nonpositive_denominator",
                )
                record.update(
                    _authority_metadata(
                        authority_path,
                        authority_digest,
                        entry,
                        "full_f1,no_recompute_f1,sempic_f1",
                    )
                )
                record["result_path"] = str(authority_path.resolve())
                record["source_artifact_sha256"] = authority_digest
                record["paired_quality_result_path"] = str(authority_path.resolve())
                record["paired_quality_result_sha256"] = authority_digest
                for field in (
                    "checkpoint_or_artifact",
                    "sample_membership",
                    "sample_or_repeat_count",
                    "seed_or_seed_list",
                    "source_config",
                    "training_run",
                ):
                    record.setdefault(f"historical_{field}", record.get(field))
                    record[field] = None
                record["sample_scope_role"] = HISTORICAL_ROLE
                record["seed_scope_role"] = HISTORICAL_ROLE
                record["source_config_role"] = HISTORICAL_ROLE
                record["known_uncertainty"] = (
                    "User-supplied final table; corrected F1 raw-run linkage is not recorded."
                )
                record["value"] = value
                record["status"] = status
                record["metric_definition"] = (
                    "Recovery=(F1_SemPIC-F1_NoRecompute)/(F1_Full-F1_NoRecompute); "
                    f"status={status}; value={value}"
                )
                recovery_provenance[parts[1]] = record
            elif parts[2] == "sink_ratio":
                record.setdefault(
                    "historical_paired_quality_result_path",
                    record.get("paired_quality_result_path"),
                )
                record.setdefault(
                    "historical_paired_quality_result_sha256",
                    record.get("paired_quality_result_sha256"),
                )
                record["paired_quality_result_path"] = str(authority_path.resolve())
                record["paired_quality_result_sha256"] = authority_digest
                record["paired_quality_result_path_role"] = AUTHORITY_ROLE

    statistics = _load_json(paths["data/statistics.json"])
    if not isinstance(statistics, dict) or not isinstance(statistics.get("estimands"), list):
        raise ValueError("Attention statistics.json has an invalid shape")
    point_by_dataset = {point["dataset_id"]: point for point in plot_data["points"]}
    for row in statistics["estimands"]:
        point = point_by_dataset.get(row.get("dataset_id"))
        if point is None:
            raise ValueError(f"Unknown attention statistics identity: {row.get('dataset_id')}")
        row["recovery_fraction"] = point["recovery_fraction"]
        row["recovery_fraction_status"] = point["recovery_fraction_status"]
        row["f1_change"] = point["f1_change"]
        row["f1_change_status"] = point["f1_change_status"]
        row["interpretation_status"] = point["interpretation_status"]

    for dataset_id, point in point_by_dataset.items():
        if dataset_id not in recovery_provenance:
            raise ValueError(f"Missing recovery provenance for {dataset_id}")
        entry = entries[(model_id, dataset_id)]
        record = deepcopy(recovery_provenance[dataset_id])
        record["measurement_id"] = point["f1_change_measurement_id"]
        record.update(
            _authority_metadata(
                authority_path,
                authority_digest,
                entry,
                "no_recompute_f1,sempic_f1",
            )
        )
        record["metric_definition"] = (
            "DeltaF1=F1_SemPIC-F1_NoRecompute; status=defined; "
            f"value={point['f1_change']}"
        )
        record["value"] = point["f1_change"]
        record["status"] = point["f1_change_status"]
        provenance.append(record)

    resolved_configs = _load_json(paths["data/resolved_configs.json"])
    if not isinstance(resolved_configs, list):
        raise ValueError("Attention resolved_configs.json must be a list")
    for record in resolved_configs:
        identifier = record.get("measurement_id", "")
        parts = identifier.split(":") if isinstance(identifier, str) else []
        if len(parts) != 3 or parts[0] != "behavior":
            raise ValueError(f"Invalid resolved config measurement_id: {identifier!r}")
        dataset_id, method = parts[1], parts[2]
        entry = entries[(model_id, dataset_id)]
        field = ATTENTION_METHOD_FIELDS[method]
        record["result_path_role"] = HISTORICAL_ROLE
        record.update(_authority_metadata(authority_path, authority_digest, entry, field))

    manifest = _load_json(paths["bundle_manifest.json"])
    if not isinstance(manifest, dict) or not isinstance(manifest.get("inputs"), list):
        raise ValueError("Attention bundle_manifest.json has an invalid shape")
    authority_resolved = str(authority_path.resolve())
    manifest["inputs"] = [
        item
        for item in manifest["inputs"]
        if item.get("path") != authority_resolved and item.get("role") != AUTHORITY_ROLE
    ]
    for item in manifest["inputs"]:
        path = item.get("path")
        if isinstance(path, str) and path.endswith("_result.json"):
            item["role"] = HISTORICAL_ROLE
        else:
            item.setdefault("role", "attention_or_processed_metrics_source")
    authority_input = {
        "path": authority_resolved,
        "sha256": authority_digest,
        "schema_version": 1,
        "role": AUTHORITY_ROLE,
    }
    manifest["inputs"].append(authority_input)
    manifest["f1_authority_input"] = authority_input

    return {
        paths["data/plot_data.json"]: _json_text(plot_data),
        paths["data/behavior_measurements.csv"]: behavior_csv,
        paths["data/position_measurements.csv"]: position_csv,
        paths["data/region_measurements.csv"]: region_csv,
        paths["data/provenance.jsonl"]: _jsonl_text(provenance),
        paths["data/statistics.json"]: _json_text(statistics),
        paths["data/resolved_configs.json"]: _json_text(resolved_configs),
        paths["bundle_manifest.json"]: _json_text(manifest),
    }


def apply_authority(
    *, authority_path: str | Path, boundary_bundle: str | Path, attention_bundle: str | Path
) -> dict[str, Any]:
    authority_source = Path(authority_path)
    authority = load_authority(authority_source)
    digest = _sha256(authority_source)
    boundary = Path(boundary_bundle)
    attention = Path(attention_bundle)
    if not boundary.is_dir() or not attention.is_dir():
        raise ValueError("Both bundle paths must be existing directories")

    _, boundary_rows = _load_csv(boundary / "plot_data.csv")
    boundary_keys = {
        (row.get("model_id"), row.get("dataset_id")) for row in boundary_rows
    }
    boundary_authority = {
        key: entry for key, entry in authority.items() if key in boundary_keys
    }
    if set(boundary_authority) != boundary_keys:
        missing = boundary_keys - set(boundary_authority)
        raise ValueError(f"Boundary authority rows are missing: {sorted(missing)}")

    attention_plot = _load_json(attention / "data/plot_data.json")
    if not isinstance(attention_plot, dict) or not isinstance(
        attention_plot.get("points"), list
    ):
        raise ValueError("Attention plot_data.json has an invalid shape")
    attention_keys = {
        (point.get("model_id"), point.get("dataset_id"))
        for point in attention_plot["points"]
    }
    attention_authority = {
        key: entry for key, entry in authority.items() if key in attention_keys
    }
    if set(attention_authority) != attention_keys:
        missing = attention_keys - set(attention_authority)
        raise ValueError(f"Attention authority rows are missing: {sorted(missing)}")

    updates = {
        **_boundary_updates(
            bundle=boundary,
            authority_path=authority_source,
            authority_digest=digest,
            authority=boundary_authority,
        ),
        **_attention_updates(
            bundle=attention,
            authority_path=authority_source,
            authority_digest=digest,
            authority=attention_authority,
        ),
    }
    for path, text in updates.items():
        _atomic_write_text(path, text)
    return {
        "authority_path": str(authority_source.resolve()),
        "authority_sha256": digest,
        "updated_files": [str(path.resolve()) for path in updates],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--boundary-bundle", type=Path, required=True)
    parser.add_argument("--attention-bundle", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = apply_authority(
        authority_path=args.authority,
        boundary_bundle=args.boundary_bundle,
        attention_bundle=args.attention_bundle,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
