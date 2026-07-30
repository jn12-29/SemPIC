from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class QuerySpec:
    models: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    benchmarks: tuple[str, ...] = ()
    run_labels: tuple[str, ...] = ()
    run_label_pattern: str | None = None
    source_path_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    frame: pd.DataFrame
    errors: tuple[str, ...] = ()


_SELECTOR_COLUMNS = (
    ("models", "model_name"),
    ("methods", "method"),
    ("datasets", "dataset_name"),
    ("benchmarks", "benchmark_label"),
    ("run_labels", "run_label"),
)


def apply_query(frame: pd.DataFrame, spec: QuerySpec) -> QueryResult:
    """Apply exact selectors and optional regular expressions without mutation."""
    requested_patterns = (
        ("Run label", "run_label", spec.run_label_pattern),
        ("Result path", "source_path", spec.source_path_pattern),
    )
    compiled: list[tuple[str, re.Pattern[str]]] = []
    errors: list[str] = []
    for label, column, pattern in requested_patterns:
        if pattern is None or pattern == "":
            continue
        try:
            compiled.append((column, re.compile(pattern)))
        except (re.error, OverflowError) as exc:
            errors.append(f"{label} regex is invalid: {exc}")

    if errors:
        return QueryResult(frame=frame.iloc[0:0].copy(), errors=tuple(errors))

    filtered = frame
    for field, column in _SELECTOR_COLUMNS:
        selected = getattr(spec, field)
        if not selected:
            continue
        if column not in filtered.columns:
            filtered = filtered.iloc[0:0]
            continue
        filtered = filtered.loc[filtered[column].isin(selected)]

    for column, pattern in compiled:
        if column not in filtered.columns:
            filtered = filtered.iloc[0:0]
            continue
        mask = filtered[column].map(
            lambda value: isinstance(value, str) and pattern.search(value) is not None
        )
        filtered = filtered.loc[mask]
    return QueryResult(frame=filtered.copy())
