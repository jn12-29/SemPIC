"""Create and resume complete attention-analysis runs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

from ..utils.run_storage import allocate_run_dir, atomic_write_json, compose_run_suffix
from .analysis_pipeline import process_attention_run
from .config_loading import load_analysis_configs
from .processing import normalize_processing_config
from .profile_identity import sanitized_id
from .profile_run import (
    build_partition_identity, collect_profile_run_loaded, group_profile_configs,
)
from .spec import AttentionAnalysisConfig


RUN_SCHEMA_NAME = "sempic.attention_analysis_run"
RUN_SCHEMA_VERSION = 2
_RUN_FIELDS = {
    "schema_name", "schema_version", "analysis_config_source",
    "processing_config_source", "analysis_config", "eval_configs",
    "max_samples", "processing_config", "figure_formats", "tasks",
}


def create_attention_run(
    config_files: Sequence[str], *, analysis_config_path: str | Path,
    processing_config_path: str | Path, processing_config: object,
    output_dir: str | Path, run_name: str | None, max_samples: int | None,
) -> Path:
    analysis = AttentionAnalysisConfig.from_file(analysis_config_path)
    loaded = load_analysis_configs(config_files)
    record = _make_run_record(
        loaded, analysis=analysis,
        analysis_config_source=analysis_config_path,
        processing_config_source=processing_config_path,
        processing_config=normalize_processing_config(processing_config),
        max_samples=max_samples,
    )
    run_dir = allocate_run_dir(
        output_dir,
        compose_run_suffix(Path(analysis_config_path).stem, user_suffix=run_name),
    )
    atomic_write_json(run_dir / "config.json", record)
    _execute_run(run_dir, record)
    return run_dir


def resume_attention_run(run_dir: str | Path) -> Path:
    root = Path(run_dir).resolve()
    record = load_run_record(root / "config.json")
    loaded, analysis = _restore_inputs(record)
    current = _make_run_record(
        loaded, analysis=analysis,
        analysis_config_source=record["analysis_config_source"],
        processing_config_source=record["processing_config_source"],
        processing_config=record["processing_config"],
        max_samples=record["max_samples"],
    )
    # Preserve the historical declaration when resuming an older run. It
    # describes that run rather than controlling the current renderer.
    current["figure_formats"] = record["figure_formats"]
    if current != record:
        raise ValueError("Run inputs changed since config.json was created.")
    _execute_run(root, record)
    return root


def load_run_record(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or set(value) != _RUN_FIELDS:
        raise ValueError("Attention run config has missing or unknown fields.")
    if value["schema_name"] != RUN_SCHEMA_NAME or value["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError("Unsupported attention run config schema.")
    if value["figure_formats"] not in (["pdf"], ["png", "pdf"]):
        raise ValueError("Attention run figure_formats is unsupported.")
    normalize_processing_config(value["processing_config"])
    _restore_inputs(value)
    if not isinstance(value["tasks"], list) or not value["tasks"]:
        raise ValueError("Attention run config must contain tasks.")
    return value


def _execute_run(run_dir: Path, record: dict[str, object]) -> None:
    loaded, analysis = _restore_inputs(record)
    collect_profile_run_loaded(
        loaded, analysis=analysis, run_dir=run_dir,
        max_samples=record["max_samples"],
    )
    process_attention_run(run_dir, processing_config=record["processing_config"])


def _restore_inputs(record: dict[str, object]):
    analysis = AttentionAnalysisConfig.from_dict(record["analysis_config"])
    configs = record["eval_configs"]
    if not isinstance(configs, list) or not configs:
        raise ValueError("Run eval_configs must be non-empty.")
    loaded = []
    for item in configs:
        if not isinstance(item, dict) or set(item) != {"source_config", "config"}:
            raise ValueError("Run eval config record is invalid.")
        loaded.append((item["source_config"], item["config"]))
    return loaded, analysis


def _make_run_record(
    loaded: Sequence[tuple[str, dict]], *, analysis: AttentionAnalysisConfig,
    analysis_config_source: str | Path, processing_config_source: str | Path,
    processing_config: dict[str, object], max_samples: int | None,
) -> dict[str, object]:
    loaded = _freeze_artifact_paths(loaded)
    eval_configs = [
        {"source_config": str(Path(source).resolve()), "config": config}
        for source, config in loaded
    ]
    canonical = [(item["source_config"], item["config"]) for item in eval_configs]
    tasks = []
    for group in group_profile_configs(canonical):
        model_id = sanitized_id(Path(group[0][1]["model"]["model_path"]).name)
        dataset_id = sanitized_id(group[0][1]["dataset"]["dataset_name"])
        for query_pass in analysis.query_passes:
            tasks.append({
                "partition_identity": build_partition_identity(
                    group, query_pass=query_pass, model_id=model_id,
                    dataset_id=dataset_id, max_samples=max_samples,
                ),
                "partition_path": f"statistics/{model_id}/{dataset_id}/{query_pass.query_pass_id}.pt",
                "work_dir": f".work/{model_id}/{dataset_id}/{query_pass.query_pass_id}",
            })
    return {
        "schema_name": RUN_SCHEMA_NAME,
        "schema_version": RUN_SCHEMA_VERSION,
        "analysis_config_source": str(Path(analysis_config_source).resolve()),
        "processing_config_source": str(Path(processing_config_source).resolve()),
        "analysis_config": analysis.to_dict(),
        "eval_configs": eval_configs,
        "max_samples": max_samples,
        "processing_config": processing_config,
        "figure_formats": ["pdf"],
        "tasks": tasks,
    }


def _freeze_artifact_paths(loaded: Sequence[tuple[str, dict]]) -> list[tuple[str, dict]]:
    frozen = copy.deepcopy(loaded)
    for _, config in frozen:
        config["model"]["model_path"] = str(Path(config["model"]["model_path"]).expanduser().resolve())
        for artifact in ("packet_wrapper", "lora"):
            path = config[artifact]["path"]
            if path is not None:
                config[artifact]["path"] = str(Path(path).expanduser().resolve())
    return frozen


__all__ = [
    "RUN_SCHEMA_NAME", "RUN_SCHEMA_VERSION", "create_attention_run",
    "load_run_record", "resume_attention_run",
]
