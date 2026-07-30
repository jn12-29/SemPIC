from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import derive_checkpoint_metadata


METRIC_PREFIX = "metric."


class InvalidResultError(ValueError):
    """Raised when a result file cannot produce a normalized record."""


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    source_path: Path
    result_dir: Path
    run_label: str
    modified_at: float
    config: Mapping[str, Any]
    result: Mapping[str, Any]
    model_path: str
    model_name: str
    dataset_name: str
    benchmark_label: str
    method_raw: str
    method: str
    method_label: str
    dataset_seed: Any
    run_seed: Any
    comparison_id: str
    comparison_label: str
    checkpoint_id: str | None
    checkpoint_label: str | None
    checkpoint_source_dataset: str | None
    checkpoint_scope: str
    algorithm_variant_id: str
    algorithm_variant_label: str
    series_id: str
    series_label: str
    metrics: Mapping[str, float]


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidResultError(f"config is not canonical JSON: {exc}") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_method(method: str) -> str:
    return "kvpacket" if method in {"kv_packet", "kvpacket"} else method


def _artifact(value: Any) -> Any:
    if value is None:
        return {"path": None}
    if isinstance(value, str):
        return {"path": value}
    return value


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidResultError(f"{name} must be an object")
    return value


def _require_string(container: Mapping[str, Any], key: str, name: str) -> str:
    value = container.get(key)
    if not isinstance(value, str):
        raise InvalidResultError(f"{name} must be a string")
    return value


def normalize_result(
    source_path: str | Path,
    payload: Any,
    *,
    modified_at: float,
) -> tuple[NormalizedRecord, tuple[str, ...]]:
    source = Path(source_path).resolve()
    root = _require_object(payload, "document")
    config = _require_object(root.get("config"), "config")
    result = _require_object(root.get("result"), "result")
    model = _require_object(config.get("model"), "config.model")
    dataset = _require_object(config.get("dataset"), "config.dataset")
    cache_comb = _require_object(config.get("cache_comb"), "config.cache_comb")

    model_path = _require_string(model, "model_path", "config.model.model_path")
    dataset_name = _require_string(dataset, "dataset_name", "config.dataset.dataset_name")
    method_raw = _require_string(cache_comb, "method", "config.cache_comb.method")

    # Validate the complete config before selecting identity fields. This also
    # rejects Python's permissive NaN/Infinity JSON extensions.
    canonical_json(config)

    comparison_model = {
        key: model[key]
        for key in ("model_path", "dtype", "generation_kwargs")
        if key in model
    }
    comparison_dataset = {
        key: value for key, value in dataset.items() if key != "seed"
    }
    comparison_id = canonical_digest(
        {"model": comparison_model, "dataset": comparison_dataset}
    )

    method = normalize_method(method_raw)
    run_label = source.name.removesuffix("_result.json")
    method_label = method.replace("_", " ").title()
    checkpoint = derive_checkpoint_metadata(
        config,
        dataset_name,
        method,
        method_label,
        run_label,
    )
    series_value: dict[str, Any] = {
        "cache_comb": {"method": method},
        "packet_wrapper": _artifact(config.get("packet_wrapper")),
        "lora": _artifact(config.get("lora")),
        "run_label": run_label,
    }
    if "kwargs" in cache_comb:
        series_value["cache_comb"]["kwargs"] = cache_comb["kwargs"]
    for key in ("compress", "quantization"):
        if key in config:
            series_value[key] = config[key]
    series_id = canonical_digest(series_value)

    warnings: list[str] = []
    metrics: dict[str, float] = {}
    for key, value in result.items():
        metric = math.nan
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                metric = float(value)
            except (OverflowError, ValueError):
                pass
        if math.isfinite(metric):
            metrics[key] = metric
        else:
            warnings.append(f"metric {key!r} is not a finite JSON number and was omitted")
    if not metrics:
        raise InvalidResultError("result has no valid numeric metrics")

    model_name = Path(model_path.rstrip("/")).name or model_path
    subset = dataset.get("subset")
    benchmark_label = dataset_name
    if subset not in (None, ""):
        benchmark_label = f"{dataset_name} ({subset})"
    comparison_label = f"{model_name} / {benchmark_label}"
    series_label = f"{method_label} / {run_label}"

    return NormalizedRecord(
        source_path=source,
        result_dir=source.parent,
        run_label=run_label,
        modified_at=modified_at,
        config=config,
        result=result,
        model_path=model_path,
        model_name=model_name,
        dataset_name=dataset_name,
        benchmark_label=benchmark_label,
        method_raw=method_raw,
        method=method,
        method_label=method_label,
        dataset_seed=dataset.get("seed"),
        run_seed=config.get("seed"),
        comparison_id=comparison_id,
        comparison_label=comparison_label,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_label=checkpoint.checkpoint_label,
        checkpoint_source_dataset=checkpoint.checkpoint_source_dataset,
        checkpoint_scope=checkpoint.checkpoint_scope,
        algorithm_variant_id=checkpoint.algorithm_variant_id,
        algorithm_variant_label=checkpoint.algorithm_variant_label,
        series_id=series_id,
        series_label=series_label,
        metrics=metrics,
    ), tuple(warnings)
