from __future__ import annotations

import pandas as pd
import streamlit as st

from sempic.eval_dashboard.aggregate import build_run_leaderboard
from sempic.eval_dashboard.charts import (
    ExactChartTooLarge,
    build_run_metric_figure,
    build_run_tradeoff_figure,
)
from sempic.eval_dashboard.metrics import metric_label, metric_options

from .common import (
    metric_sort_label,
    select_id,
    status_badge,
)


def _comparison_labels(frame: pd.DataFrame) -> dict[str, str]:
    pairs = frame[["comparison_id", "comparison_label"]].drop_duplicates()
    labels: dict[str, str] = {}
    for row in pairs.itertuples(index=False):
        identifier = str(row.comparison_id)
        label = str(row.comparison_label)
        labels[identifier] = label
    duplicate_labels = {
        label for label in labels.values() if list(labels.values()).count(label) > 1
    }
    return {
        identifier: (
            f"{label} [{identifier[:8]}]" if label in duplicate_labels else label
        )
        for identifier, label in labels.items()
    }


def render_experiment(frame: pd.DataFrame) -> None:
    st.subheader("Experiment")
    status_badge("Exact runs · shared comparison snapshot")
    st.caption(
        "Select one evaluation context. The leaderboard and chart use the same exact observations."
    )

    labels = _comparison_labels(frame)
    comparison_ids = tuple(sorted(labels, key=lambda value: (labels[value], value)))
    if not comparison_ids:
        st.info("No experiment comparisons are available in the shared scope.")
        return
    comparison_id = select_id(
        "Comparison",
        comparison_ids,
        key="eval_dashboard.experiment.comparison",
        labels=labels,
    )
    comparison = frame.loc[frame["comparison_id"].astype(str) == comparison_id].copy()
    st.caption(
        f"{len(comparison):,} observations · "
        f"{comparison['checkpoint_id'].dropna().astype(str).nunique():,} checkpoints"
    )

    st.markdown("#### Exact leaderboard")
    sort_metrics = tuple(
        metric
        for metric in (
            "metric.f1",
            "metric.ttft_mean",
            "metric.ttft_p50",
            "metric.ttft_p99",
            "metric.flops",
        )
        if metric in comparison.columns
        or (metric == "metric.ttft_mean" and "metric.ttft" in comparison.columns)
    )
    if sort_metrics:
        sort_metric = select_id(
            "Sort leaderboard by",
            sort_metrics,
            key="eval_dashboard.experiment.leaderboard_sort",
            labels={metric: metric_sort_label(metric) for metric in sort_metrics},
        )
        leaderboard = build_run_leaderboard(
            comparison, comparison_id=comparison_id, sort_metric=sort_metric
        )
        st.dataframe(
            leaderboard,
            width="stretch",
            hide_index=True,
            column_order=(
                "Method",
                "Run label",
                "Checkpoint",
                "F1",
                "TTFT Mean",
                "TTFT P50",
                "TTFT P99",
                "FLOPs",
            ),
            column_config={
                "TTFT Mean": "TTFT Mean (s)",
                "TTFT P50": "TTFT P50 (s)",
                "TTFT P99": "TTFT P99 (s)",
            },
        )
    else:
        st.info("This comparison has no standard leaderboard metrics.")

    chart_kinds = (
        "Ranked metric",
        "F1 vs TTFT",
        "F1 vs FLOPs",
    )
    chart_kind = st.selectbox(
        "Chart view",
        chart_kinds,
        key="eval_dashboard.experiment.chart_kind",
    )
    try:
        if chart_kind == "F1 vs TTFT":
            figure = build_run_tradeoff_figure(
                comparison,
                "metric.ttft_mean",
                title="F1 vs TTFT mean · exact runs",
            )
        elif chart_kind == "F1 vs FLOPs":
            figure = build_run_tradeoff_figure(
                comparison,
                "metric.flops",
                title="F1 vs FLOPs · exact runs",
            )
        else:
            metrics = metric_options(comparison)
            if not metrics:
                st.info("This comparison has no numeric metrics to chart.")
                return
            metric = select_id(
                "Chart metric",
                metrics,
                key="eval_dashboard.experiment.metric",
                labels={value: metric_label(value) for value in metrics},
            )
            figure = build_run_metric_figure(
                comparison,
                metric,
                title=f"{metric_label(metric)} by exact run",
            )
    except ExactChartTooLarge as exc:
        st.warning(
            f"This exact chart contains {exc.count:,} observations, above the "
            f"{exc.limit:,}-observation rendering limit. Narrow the shared scope "
            "or select a smaller comparison. No observations were sampled."
        )
        return
    if figure is None:
        st.info("No finite values are available for the selected metric.")
    else:
        st.plotly_chart(figure, width="stretch")
