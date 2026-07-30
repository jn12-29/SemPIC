"""Canonical identities for attention profile partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch


CANONICAL_METHODS = (
    "full_recompute",
    "vanilla_pic",
    "kvpacket",
    "sempic",
    "sempic_kvpacket",
)

_IDENTITY_FIELDS = {
    "model_config",
    "dataset_config",
    "eval_seed",
    "artifact_snapshots",
    "model_id",
    "dataset_id",
    "query_pass_id",
    "query_spec",
    "methods",
    "max_samples",
    "dataset_iteration",
}
_LEGACY_IDENTITY_FIELDS = _IDENTITY_FIELDS - {"eval_seed"}
_METHOD_FIELDS = {
    "method_key",
    "runtime_fingerprint",
    "resolved_method_config",
    "source_config",
}
_RESOLVED_METHOD_FIELDS = {
    "cache_comb",
    "packet_wrapper",
    "lora",
    "compress",
    "quantization",
}
_SNAPSHOT_FIELDS = {"artifact_key", "canonical_path", "files"}
_SNAPSHOT_FILE_FIELDS = {"relative_path", "size", "mtime_ns"}


def fingerprint(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_method_key(method: str) -> str:
    normalized = "vanilla_pic" if method == "no_recompute" else method
    if normalized not in CANONICAL_METHODS:
        raise ValueError(f"Unsupported attention method: {method!r}.")
    return normalized


def runtime_fingerprint(
    model_config: dict[str, object], resolved_method_config: dict[str, object]
) -> str:
    return fingerprint({
        "model_config": model_config,
        "resolved_method_config": resolved_method_config,
    })


def sanitized_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not normalized:
        raise ValueError("Resolved identity cannot produce an empty path ID.")
    return normalized


def strict_dict(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has missing or unknown fields.")
    return value


def non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def validate_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError(f"{name} must contain only finite numbers.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_json_value(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings.")
            validate_json_value(item, f"{name}.{key}")
        return
    raise ValueError(f"{name} contains unsupported type {type(value).__name__}.")


def validate_methods(
    value: object, *, model_config: dict[str, object] | None = None
) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("methods must be a non-empty list.")
    methods: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        method = strict_dict(raw, _METHOD_FIELDS, f"methods[{index}]")
        key = normalize_method_key(method["method_key"])
        if key != method["method_key"]:
            raise ValueError("Serialized method keys must use canonical names.")
        if key in seen:
            raise ValueError("methods must not contain duplicate method keys.")
        seen.add(key)
        non_empty_string(method["runtime_fingerprint"], "runtime_fingerprint")
        resolved = strict_dict(
            method["resolved_method_config"],
            _RESOLVED_METHOD_FIELDS,
            "resolved_method_config",
        )
        cache_comb = strict_dict(
            resolved["cache_comb"], {"method", "kwargs"}, "cache_comb"
        )
        if cache_comb["method"] != key or not isinstance(cache_comb["kwargs"], dict):
            raise ValueError("cache_comb must use the canonical method key and kwargs.")
        for artifact_name in ("packet_wrapper", "lora"):
            artifact_config = strict_dict(
                resolved[artifact_name], {"path"}, artifact_name
            )
            path = artifact_config["path"]
            if path is not None and (not isinstance(path, str) or not path):
                raise ValueError(f"{artifact_name}.path must be non-empty or null.")
        expected_artifacts = {
            "full_recompute": (False, False),
            "vanilla_pic": (False, False),
            "kvpacket": (True, False),
            "sempic": (False, True),
            "sempic_kvpacket": (True, True),
        }[key]
        has_packet = resolved["packet_wrapper"]["path"] is not None
        has_lora = resolved["lora"]["path"] is not None
        if (has_packet, has_lora) != expected_artifacts:
            raise ValueError(f"Method {key} has inconsistent PacketWrapper or LoRA paths.")
        for optional_config in ("compress", "quantization"):
            if resolved[optional_config] is not None and not isinstance(
                resolved[optional_config], dict
            ):
                raise ValueError(f"{optional_config} must be an object or null.")
        if model_config is not None and method["runtime_fingerprint"] != runtime_fingerprint(
            model_config, resolved
        ):
            raise ValueError("Method runtime_fingerprint mismatch.")
        non_empty_string(method["source_config"], "source_config")
        validate_json_value(method, f"methods[{index}]")
        methods.append(method)
    if methods[0]["method_key"] != "full_recompute":
        raise ValueError("full_recompute must be the first method.")
    return methods


def validate_query_pass_identity(value: object) -> dict[str, object]:
    if isinstance(value, dict) and set(value) == _LEGACY_IDENTITY_FIELDS:
        identity = value
    else:
        identity = strict_dict(value, _IDENTITY_FIELDS, "partition_identity")
    if not isinstance(identity["model_config"], dict) or not identity["model_config"]:
        raise ValueError("model_config must be a non-empty object.")
    if not isinstance(identity["dataset_config"], dict) or not identity["dataset_config"]:
        raise ValueError("dataset_config must be a non-empty object.")
    snapshot_paths, snapshot_keys = _validate_artifact_snapshots(
        identity["artifact_snapshots"]
    )
    for field in ("model_id", "dataset_id", "query_pass_id"):
        non_empty_string(identity[field], f"partition_identity.{field}")
    if not isinstance(identity["query_spec"], dict) or not identity["query_spec"]:
        raise ValueError("partition_identity.query_spec must be a non-empty object.")
    from .spec import QueryPassSpec
    query_spec = QueryPassSpec.from_dict(identity["query_spec"])
    if query_spec.query_pass_id != identity["query_pass_id"]:
        raise ValueError("query_spec.query_pass_id must match query_pass_id.")
    methods = validate_methods(
        identity["methods"], model_config=identity["model_config"]
    )
    missing_artifacts = []
    model_path = _canonical_artifact_path(
        identity["model_config"].get("model_path"), "model"
    )
    if model_path not in snapshot_paths and "model" not in snapshot_keys:
        missing_artifacts.append("model")
    for method in methods:
        resolved = method["resolved_method_config"]
        for artifact_name in ("packet_wrapper", "lora"):
            configured_path = resolved[artifact_name]["path"]
            if configured_path is not None:
                path = _canonical_artifact_path(configured_path, artifact_name)
                artifact_key = f"{artifact_name}:{method['method_key']}"
                if path not in snapshot_paths and artifact_key not in snapshot_keys:
                    missing_artifacts.append(artifact_key)
    if missing_artifacts:
        raise ValueError("artifact_snapshots do not cover all configured artifact paths.")
    max_samples = identity["max_samples"]
    if max_samples is not None:
        non_negative_int(max_samples, "partition_identity.max_samples")
        if max_samples == 0:
            raise ValueError("partition_identity.max_samples must be positive.")
    if "eval_seed" in identity:
        non_negative_int(identity["eval_seed"], "partition_identity.eval_seed")
    if not isinstance(identity["dataset_iteration"], dict):
        raise ValueError("dataset_iteration must be an object.")
    expected_ids = {
        "model_id": sanitized_id(Path(model_path).name),
        "dataset_id": sanitized_id(identity["dataset_config"].get("dataset_name", "")),
    }
    for field, expected in expected_ids.items():
        if identity[field] != expected:
            raise ValueError(f"partition_identity.{field} mismatch.")
    validate_json_value(identity, "partition_identity")
    return identity


def _canonical_artifact_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} artifact path must be a non-empty string.")
    return str(Path(value).expanduser().resolve())


def _validate_artifact_snapshots(value: object) -> tuple[set[str], set[str]]:
    if not isinstance(value, list):
        raise ValueError("artifact_snapshots must be a list.")
    artifact_keys: set[str] = set()
    canonical_paths: set[str] = set()
    for snapshot_index, raw_snapshot in enumerate(value):
        snapshot = strict_dict(
            raw_snapshot, _SNAPSHOT_FIELDS, f"artifact_snapshots[{snapshot_index}]"
        )
        artifact_key = non_empty_string(snapshot["artifact_key"], "artifact_key")
        if artifact_key in artifact_keys:
            raise ValueError("artifact_snapshots must have unique artifact keys.")
        artifact_keys.add(artifact_key)
        canonical_path = Path(non_empty_string(snapshot["canonical_path"], "canonical_path"))
        if not canonical_path.is_absolute():
            raise ValueError("artifact snapshot canonical_path must be absolute.")
        normalized_path = str(canonical_path.resolve())
        if str(canonical_path) != normalized_path or normalized_path in canonical_paths:
            raise ValueError("artifact snapshot canonical paths must be normalized and unique.")
        canonical_paths.add(normalized_path)
        files = snapshot["files"]
        if not isinstance(files, list):
            raise ValueError("artifact snapshot files must be a list.")
        relative_paths: list[str] = []
        for file_index, raw_file in enumerate(files):
            file_record = strict_dict(
                raw_file,
                _SNAPSHOT_FILE_FIELDS,
                f"artifact_snapshots[{snapshot_index}].files[{file_index}]",
            )
            relative_path = non_empty_string(file_record["relative_path"], "relative_path")
            parsed_path = Path(relative_path)
            if parsed_path.is_absolute() or ".." in parsed_path.parts:
                raise ValueError("artifact snapshot file paths must be relative descendants.")
            non_negative_int(file_record["size"], "artifact file size")
            non_negative_int(file_record["mtime_ns"], "artifact file mtime_ns")
            relative_paths.append(relative_path)
        if relative_paths != sorted(set(relative_paths)):
            raise ValueError("artifact snapshot files must be unique and sorted by path.")
    return canonical_paths, artifact_keys


__all__ = [
    "CANONICAL_METHODS",
    "fingerprint",
    "non_empty_string",
    "non_negative_int",
    "normalize_method_key",
    "runtime_fingerprint",
    "sanitized_id",
    "strict_dict",
    "validate_json_value",
    "validate_methods",
    "validate_query_pass_identity",
]
