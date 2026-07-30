from __future__ import annotations

import pandas as pd
import streamlit as st

from sempic.eval_dashboard.metrics import (
    metric_is_lower_better,
    metric_label,
    metric_numeric,
    metric_options,
)
from sempic.eval_dashboard.query import QuerySpec, apply_query

from .common import render_multiselect, render_run_detail, select_id, status_badge


_SORT_METRICS = (
    "metric.f1",
    "metric.ttft_mean",
    "metric.ttft_p99",
    "metric.flops",
)


def _available_sort_options(frame: pd.DataFrame) -> dict[str, tuple[str, bool]]:
    present = set(metric_options(frame))
    available = {
        f"{metric_label(metric)} · "
        f"{'low to high' if metric_is_lower_better(metric) else 'high to low'}": (
            metric,
            metric_is_lower_better(metric),
        )
        for metric in _SORT_METRICS
        if metric in present
    }
    available["Run label · A to Z"] = ("run_label", True)
    return available


def _exact_runs_table(frame: pd.DataFrame) -> pd.DataFrame:
    def values(column: str) -> pd.Series:
        series = frame.get(column, pd.Series(index=frame.index, dtype=object))
        return series.fillna("")

    return pd.DataFrame(
        {
            "Dataset": values("benchmark_label"),
            "Method": values("method_label"),
            "Run label": values("run_label"),
            "Checkpoint": values("checkpoint_label"),
            "F1": metric_numeric(frame, "metric.f1"),
            "TTFT Mean (s)": metric_numeric(frame, "metric.ttft_mean"),
            "TTFT P99 (s)": metric_numeric(frame, "metric.ttft_p99"),
            "FLOPs": metric_numeric(frame, "metric.flops"),
            "Result path": values("source_path"),
        },
        index=frame.index,
    ).reset_index(drop=True)


def render_runs(frame: pd.DataFrame) -> None:
    st.subheader("Runs")
    status_badge("Exact · no aggregation")
    st.caption(
        "Browse one observation per result file. Dataset and regex controls apply only here."
    )

    datasets = tuple(sorted(frame["dataset_name"].dropna().astype(str).unique()))
    dataset_column, run_regex_column, path_regex_column = st.columns([1, 1, 1])
    with dataset_column:
        selected_datasets = render_multiselect(
            "Datasets", datasets, key="eval_dashboard.runs.datasets"
        )
    with run_regex_column:
        run_pattern = st.text_input(
            "Run label regex", key="eval_dashboard.runs.run_regex"
        )
    with path_regex_column:
        path_pattern = st.text_input(
            "Result path regex", key="eval_dashboard.runs.path_regex"
        )

    if not selected_datasets:
        st.info("No datasets are selected for Runs.")
        return
    query = apply_query(
        frame,
        QuerySpec(
            datasets=selected_datasets,
            run_label_pattern=run_pattern,
            source_path_pattern=path_pattern,
        ),
    )
    for error in query.errors:
        st.error(error)
    if query.frame.empty:
        st.info("No exact runs match the local controls.")
        return

    sort_options = _available_sort_options(query.frame)
    sort_label = st.selectbox(
        "Sort exact runs",
        tuple(sort_options),
        key="eval_dashboard.runs.sort",
    )
    sort_column, ascending = sort_options[sort_label]
    selected = query.frame.copy()
    if sort_column.startswith("metric."):
        selected["_run_sort"] = metric_numeric(selected, sort_column)
        sort_column = "_run_sort"
    if sort_column in selected.columns:
        selected = selected.sort_values(
            [sort_column, "source_path"],
            ascending=[ascending, True],
            na_position="last",
            kind="stable",
        )
    st.caption(f"{len(selected):,} exact observations")
    st.dataframe(
        _exact_runs_table(selected),
        width="stretch",
        hide_index=True,
        column_order=(
            "Dataset",
            "Method",
            "Run label",
            "Checkpoint",
            "F1",
            "TTFT Mean (s)",
            "TTFT P99 (s)",
            "FLOPs",
        ),
    )

    paths = selected["source_path"].astype(str).tolist()
    labels = dict(zip(paths, selected["run_label"].astype(str).tolist()))
    selected_path = select_id(
        "Inspect exact run",
        paths,
        key="eval_dashboard.runs.selected_source",
        labels={path: f"{labels[path]} · {path}" for path in paths},
    )
    render_run_detail(
        selected.loc[selected["source_path"].astype(str) == selected_path].iloc[0]
    )
