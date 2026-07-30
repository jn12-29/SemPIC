"""Model-free processing and visualization for attention profile partitions."""

from __future__ import annotations

import fcntl
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

from sempic.attention_visualization.processed import plot_processed_metrics
from sempic.utils.run_storage import atomic_write_json, validate_run_suffix

from .processed_storage import save_processed_metrics
from .processing import normalize_processing_config, process_partitions
from .profile_storage import load_partition


_AUTO_VARIANT = re.compile(r"^\d{8}_\d{6}-(\d+)$")


def load_processing_config(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("Processing config must be a JSON object.")
    return value


def find_profile_partitions(run_dir: str | Path) -> list[Path]:
    paths = sorted((Path(run_dir) / "statistics").glob("*/*/*.pt"))
    if not paths:
        raise FileNotFoundError("No attention profile partitions were found.")
    return paths


def process_profile_partitions(
    partition_paths: Sequence[str | Path],
    *,
    processing_config: object,
    metrics_path: str | Path,
) -> Path:
    partitions = [load_partition(path) for path in partition_paths]
    artifact = process_partitions(partitions, processing_config)
    return save_processed_metrics(metrics_path, artifact)


def allocate_analysis_variant(
    run_dir: str | Path,
    *,
    suffix: str | None = None,
    now: datetime | None = None,
) -> Path:
    root = Path(run_dir)
    if not root.is_dir():
        raise ValueError(f"Attention run directory does not exist: {root}")
    try:
        validate_run_suffix(suffix)
    except ValueError as error:
        raise ValueError(
            "suffix must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores, and hyphens."
        ) from error
    if suffix is not None and suffix.isdigit():
        raise ValueError("suffix must contain at least one non-digit character.")

    analysis_root = root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    with (analysis_root / ".allocation.lock").open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if suffix is None:
            indices = [
                int(match.group(1))
                for path in analysis_root.iterdir()
                if path.is_dir() and (match := _AUTO_VARIANT.fullmatch(path.name))
            ]
            name = f"{timestamp}-{max(indices, default=-1) + 1}"
        else:
            base = f"{timestamp}-{suffix}"
            name = base
            collision = 0
            while (analysis_root / name).exists():
                name = f"{base}-{collision}"
                collision += 1
        variant_dir = analysis_root / name
        variant_dir.mkdir()
        return variant_dir


def process_attention_run(
    run_dir: str | Path,
    *,
    processing_config: object,
    suffix: str | None = None,
    now: datetime | None = None,
) -> Path:
    normalized = normalize_processing_config(processing_config)
    variant_dir = allocate_analysis_variant(run_dir, suffix=suffix, now=now)
    atomic_write_json(variant_dir / "processing_config.json", normalized)
    metrics_path = process_profile_partitions(
        find_profile_partitions(run_dir),
        processing_config=normalized,
        metrics_path=variant_dir / "metrics.pt",
    )
    plot_processed_metrics(metrics_path, variant_dir / "figures")
    return variant_dir


def plot_attention_variant(variant_dir: str | Path) -> Path:
    root = Path(variant_dir)
    metrics_path = root / "metrics.pt"
    if not metrics_path.is_file():
        raise ValueError(f"Analysis variant has no metrics.pt: {root}")
    plot_processed_metrics(metrics_path, root / "figures")
    return root


__all__ = [
    "allocate_analysis_variant",
    "find_profile_partitions",
    "load_processing_config",
    "plot_attention_variant",
    "process_attention_run",
    "process_profile_partitions",
]
