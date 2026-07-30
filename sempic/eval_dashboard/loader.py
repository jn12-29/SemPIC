from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .schema import InvalidResultError, NormalizedRecord, normalize_result


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    directories: tuple[Path, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LoadReport:
    records: tuple[NormalizedRecord, ...]
    warnings: tuple[str, ...]


def discover_result_directories(
    roots: Sequence[str | Path],
) -> DiscoveryReport:
    directories: set[Path] = set()
    warnings: list[str] = []
    resolved_root_set: set[Path] = set()
    for raw_root in roots:
        try:
            resolved_root_set.add(Path(raw_root).resolve())
        except (OSError, RuntimeError) as exc:
            warnings.append(f"{raw_root}: cannot resolve scan root: {exc}")
    resolved_roots = sorted(resolved_root_set, key=str)
    for root in resolved_roots:
        if not root.exists():
            warnings.append(f"{root}: root does not exist")
            continue
        if not root.is_dir():
            warnings.append(f"{root}: root is not a directory")
            continue

        def onerror(error: OSError) -> None:
            warnings.append(f"{error.filename or root}: cannot scan: {error}")

        for current, _, files in os.walk(root, onerror=onerror):
            if any(name.endswith("_result.json") for name in files):
                try:
                    directories.add(Path(current).resolve())
                except (OSError, RuntimeError) as exc:
                    warnings.append(f"{current}: cannot resolve result directory: {exc}")
    return DiscoveryReport(
        directories=tuple(sorted(directories, key=str)),
        warnings=tuple(warnings),
    )


def load_result_directories(
    directories: Sequence[str | Path],
) -> LoadReport:
    records: list[NormalizedRecord] = []
    warnings: list[str] = []
    seen_sources: set[Path] = set()
    resolved_directory_set: set[Path] = set()
    for raw_directory in directories:
        try:
            resolved_directory_set.add(Path(raw_directory).resolve())
        except (OSError, RuntimeError) as exc:
            warnings.append(f"{raw_directory}: cannot resolve result directory: {exc}")
    resolved_directories = sorted(resolved_directory_set, key=str)
    for directory in resolved_directories:
        if not directory.exists() or not directory.is_dir():
            warnings.append(f"{directory}: result directory is missing or unreadable")
            continue
        try:
            sources = sorted(directory.glob("*_result.json"), key=str)
        except OSError as exc:
            warnings.append(f"{directory}: cannot list result files: {exc}")
            continue
        for raw_source in sources:
            try:
                source = raw_source.resolve()
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                modified_at = source.stat().st_mtime
                with source.open(encoding="utf-8") as stream:
                    payload = json.load(stream)
                record, metric_warnings = normalize_result(
                    source, payload, modified_at=modified_at
                )
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                InvalidResultError,
                RecursionError,
                RuntimeError,
            ) as exc:
                warnings.append(f"{raw_source}: {exc}")
                continue
            records.append(record)
            warnings.extend(f"{source}: {warning}" for warning in metric_warnings)
    return LoadReport(records=tuple(records), warnings=tuple(warnings))
