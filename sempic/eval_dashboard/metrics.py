from __future__ import annotations

import pandas as pd


METRIC_LABELS = {
    "metric.f1": "F1",
    "metric.precision": "Precision",
    "metric.recall": "Recall",
    "metric.ttft": "TTFT mean (s)",
    "metric.ttft_mean": "TTFT mean (s)",
    "metric.ttft_p50": "TTFT P50 (s)",
    "metric.ttft_p90": "TTFT P90 (s)",
    "metric.ttft_p99": "TTFT P99 (s)",
    "metric.flops": "FLOPs",
    "metric.ttft_count": "TTFT count",
    "metric.ttft_max": "TTFT max (s)",
    "metric.ttft_min": "TTFT min (s)",
    "metric.ttft_std": "TTFT std (s)",
}

_METRIC_UNITS = {
    "metric.ttft": "s",
    "metric.ttft_mean": "s",
    "metric.ttft_p50": "s",
    "metric.ttft_p90": "s",
    "metric.ttft_p99": "s",
    "metric.ttft_min": "s",
    "metric.ttft_max": "s",
    "metric.ttft_std": "s",
}


def _canonical_metric(metric: str) -> str:
    return "metric.ttft_mean" if metric == "metric.ttft" else metric


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(
        metric, metric.removeprefix("metric.").replace("_", " ").title()
    )


def metric_unit(metric: str) -> str | None:
    return _METRIC_UNITS.get(metric)


def metric_is_lower_better(metric: str) -> bool:
    return metric.startswith("metric.ttft") or metric == "metric.flops"


def metric_numeric(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Return numeric values aligned to frame, preserving missing values."""

    def numeric(column: str) -> pd.Series:
        values = frame.get(column, pd.Series(index=frame.index, dtype=float))
        return pd.to_numeric(values, errors="coerce")

    if metric == "metric.ttft_mean":
        return numeric("metric.ttft_mean").combine_first(numeric("metric.ttft"))
    return numeric(metric)


def metric_options(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return display metrics with legacy TTFT canonicalized to TTFT mean."""
    present = {
        str(column) for column in frame.columns if str(column).startswith("metric.")
    }
    canonical_present = {_canonical_metric(metric) for metric in present}
    preferred: list[str] = []
    for metric in METRIC_LABELS:
        canonical = _canonical_metric(metric)
        if canonical in canonical_present and canonical not in preferred:
            preferred.append(canonical)
    extras = sorted(canonical_present.difference(preferred))
    return tuple((*preferred, *extras))
