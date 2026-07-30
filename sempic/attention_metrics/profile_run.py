"""Run real attention collection once per model, dataset, and query pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .config_loading import load_analysis_configs
from ..evaluation import create_eval_resource_cache, release_eval_resource_cache
from .profile_collection import collect_query_pass_partition
from .profile_identity import normalize_method_key, runtime_fingerprint, sanitized_id
from .real_profile_provider import QueryPassForwardProvider
from .spec import AttentionAnalysisConfig, QueryPassSpec


METHOD_ORDER = (
    "full_recompute", "vanilla_pic", "kvpacket", "sempic", "sempic_kvpacket"
)


def collect_profile_run(
    config_files: Sequence[str], *, analysis_config: str | Path,
    run_dir: str | Path, max_samples: int | None, overwrite: bool = False,
) -> list[Path]:
    return collect_profile_run_loaded(
        load_analysis_configs(config_files),
        analysis=AttentionAnalysisConfig.from_file(analysis_config),
        run_dir=run_dir,
        max_samples=max_samples,
        overwrite=overwrite,
    )


def collect_profile_run_loaded(
    loaded_configs: Sequence[tuple[str, dict]], *, analysis: AttentionAnalysisConfig,
    run_dir: str | Path, max_samples: int | None, overwrite: bool = False,
) -> list[Path]:
    root = Path(run_dir)
    outputs: list[Path] = []
    groups_by_model: dict[str, list[list[tuple[str, dict]]]] = {}
    for group in group_profile_configs(loaded_configs):
        ordered = _ordered_group(group)
        groups_by_model.setdefault(_model_resource_key(ordered[0][1]["model"]), []).append(ordered)

    for model_groups in groups_by_model.values():
        resource_cache = create_eval_resource_cache()
        try:
            for group in model_groups:
                provider = QueryPassForwardProvider(
                    configs=group, resource_cache=resource_cache
                )
                try:
                    model_id = sanitized_id(Path(group[0][1]["model"]["model_path"]).name)
                    dataset_id = sanitized_id(group[0][1]["dataset"]["dataset_name"])
                    for query_pass in analysis.query_passes:
                        identity = build_partition_identity(
                            group, query_pass=query_pass, model_id=model_id,
                            dataset_id=dataset_id, max_samples=max_samples,
                        )
                        partition_path = root / "statistics" / model_id / dataset_id / f"{query_pass.query_pass_id}.pt"
                        outputs.append(collect_query_pass_partition(
                            partition_identity=identity,
                            query_pass=query_pass,
                            entries=provider.entries(max_samples),
                            identify_sample=provider.identify_sample,
                            stream_method=provider.stream_method,
                            layer_count=provider.layer_count,
                            query_head_count=provider.query_head_count,
                            partition_path=partition_path,
                            work_dir=root / ".work" / model_id / dataset_id / query_pass.query_pass_id,
                            overwrite=overwrite,
                        ))
                finally:
                    provider.close()
        finally:
            release_eval_resource_cache(resource_cache)
    return outputs


def group_profile_configs(loaded: Sequence[tuple[str, dict]]) -> list[list[tuple[str, dict]]]:
    groups: dict[str, list[tuple[str, dict]]] = {}
    for item in loaded:
        config = item[1]
        key = json.dumps(
            {
                "model": config["model"],
                "dataset": config["dataset"],
                "eval_seed": config["seed"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        methods = [normalize_method_key(config["cache_comb"]["method"]) for _, config in group]
        if len(set(methods)) != len(methods) or "full_recompute" not in methods:
            raise ValueError("Each model/dataset needs Full and at most one config per method.")
    return list(groups.values())


def _ordered_group(group: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    by_method = {normalize_method_key(config["cache_comb"]["method"]): (source, config) for source, config in group}
    return [by_method[method] for method in METHOD_ORDER if method in by_method]


def _model_resource_key(model_config: dict) -> str:
    return json.dumps(
        {field: model_config[field] for field in ("model_path", "dtype", "device")},
        sort_keys=True, separators=(",", ":"),
    )


def build_partition_identity(
    group: list[tuple[str, dict]], *, query_pass: QueryPassSpec,
    model_id: str, dataset_id: str, max_samples: int | None,
) -> dict[str, object]:
    group = _ordered_group(group)
    model_config = dict(group[0][1]["model"])
    methods = []
    artifact_paths = [("model", model_config["model_path"])]
    for source_config, config in group:
        key = normalize_method_key(config["cache_comb"]["method"])
        resolved = {
            "cache_comb": {"method": key, "kwargs": dict(config["cache_comb"]["kwargs"])},
            "packet_wrapper": dict(config["packet_wrapper"]),
            "lora": dict(config["lora"]),
            "compress": config["compress"],
            "quantization": config["quantization"],
        }
        methods.append({
            "method_key": key,
            "runtime_fingerprint": runtime_fingerprint(model_config, resolved),
            "resolved_method_config": resolved,
            "source_config": str(Path(source_config).resolve()),
        })
        for artifact_name in ("packet_wrapper", "lora"):
            if resolved[artifact_name]["path"] is not None:
                artifact_paths.append((f"{artifact_name}:{key}", resolved[artifact_name]["path"]))
    snapshots = []
    seen = set()
    for artifact_key, path in artifact_paths:
        canonical = str(Path(path).expanduser().resolve())
        if canonical not in seen:
            snapshots.append(snapshot_artifact(artifact_key, canonical))
            seen.add(canonical)
    return {
        "model_config": model_config,
        "dataset_config": dict(group[0][1]["dataset"]),
        "eval_seed": group[0][1]["seed"],
        "artifact_snapshots": snapshots,
        "model_id": model_id,
        "dataset_id": dataset_id,
        "query_pass_id": query_pass.query_pass_id,
        "query_spec": query_pass.to_dict(),
        "methods": methods,
        "max_samples": max_samples,
        "dataset_iteration": dict(group[0][1]["dataset"]),
    }


def snapshot_artifact(artifact_key: str, path: str | Path) -> dict[str, object]:
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Attention artifact does not exist: {artifact_path}")
    return {"artifact_key": artifact_key, "canonical_path": str(artifact_path), "files": []}


__all__ = [
    "METHOD_ORDER", "build_partition_identity", "collect_profile_run",
    "collect_profile_run_loaded", "group_profile_configs", "snapshot_artifact",
]
