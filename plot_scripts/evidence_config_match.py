"""Strict behavioral-config matching for paper evidence exports."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


NON_BEHAVIORAL_TOP_LEVEL_FIELDS = frozenset({"logging", "debug_dump", "run_suffix"})
PATH_FIELD_NAMES = frozenset({"model_path", "path"})


@dataclass(frozen=True)
class ConfigDifference:
    """One exact difference between two behavioral projections."""

    field: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class ConfigMatch:
    """Result of comparing a candidate result config to frozen authority."""

    status: str
    expected_projection: dict[str, Any]
    candidate_projection: dict[str, Any]
    differences: tuple[ConfigDifference, ...]

    @property
    def matched(self) -> bool:
        return self.status == "matched"


def _normalize_path(value: str, repo_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve(strict=False))


def _project_value(value: Any, repo_root: Path, field_name: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _project_value(child, repo_root, key)
            for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        return [_project_value(child, repo_root, field_name) for child in value]
    if isinstance(value, str) and field_name in PATH_FIELD_NAMES:
        return _normalize_path(value, repo_root)
    return value


def behavioral_projection(config: Mapping[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Return the complete config minus the contract's non-behavioral fields.

    Unknown fields remain in the projection. This makes the matcher fail closed:
    only logging, debug output, and run suffix are ignored, while relative path
    spellings are normalized against ``repo_root``.
    """

    root = Path(repo_root).resolve()
    return {
        key: _project_value(value, root, key)
        for key, value in sorted(config.items())
        if key not in NON_BEHAVIORAL_TOP_LEVEL_FIELDS
    }


def _compare_values(expected: Any, actual: Any, field: str = "") -> list[ConfigDifference]:
    if type(expected) is not type(actual):
        return [ConfigDifference(field, expected, actual)]

    if isinstance(expected, dict):
        differences: list[ConfigDifference] = []
        for key in sorted(set(expected) | set(actual)):
            child_field = f"{field}.{key}" if field else key
            if key not in expected:
                differences.append(ConfigDifference(child_field, "<missing>", actual[key]))
            elif key not in actual:
                differences.append(ConfigDifference(child_field, expected[key], "<missing>"))
            else:
                differences.extend(_compare_values(expected[key], actual[key], child_field))
        return differences

    if isinstance(expected, list):
        if expected != actual:
            return [ConfigDifference(field, expected, actual)]
        return []

    if expected != actual:
        return [ConfigDifference(field, expected, actual)]
    return []


def compare_behavioral_configs(
    expected: Mapping[str, Any],
    candidate: Mapping[str, Any],
    repo_root: str | Path,
) -> ConfigMatch:
    """Compare two configs under the frozen behavioral-equivalence contract."""

    expected_projection = behavioral_projection(expected, repo_root)
    candidate_projection = behavioral_projection(candidate, repo_root)
    differences = tuple(_compare_values(expected_projection, candidate_projection))
    return ConfigMatch(
        status="matched" if not differences else "incompatible",
        expected_projection=expected_projection,
        candidate_projection=candidate_projection,
        differences=differences,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_completed_result(path: str | Path) -> dict[str, Any]:
    """Load the aggregate evaluation payload required by the evidence builder."""

    result_path = Path(path)
    with result_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Result payload has no config object: {result_path}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Result payload has no result object: {result_path}")
    f1 = result.get("f1")
    if isinstance(f1, bool) or not isinstance(f1, (int, float)) or not math.isfinite(f1):
        raise ValueError(f"Result payload has no finite numeric F1: {result_path}")
    return payload


def measurement_id(
    *,
    method_key: str,
    config_projection_sha256: str,
    result_payload_sha256: str,
) -> str:
    """Build a method-normalized ID without conflating distinct result files."""

    identity = {
        "kind": "behavior",
        "method_key": method_key,
        "config_projection_sha256": config_projection_sha256,
        "result_payload_sha256": result_payload_sha256,
    }
    return f"behavior-{sha256_value(identity)[:24]}"


def differences_json(differences: Iterable[ConfigDifference]) -> str:
    return json.dumps(
        [difference.as_dict() for difference in differences],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
