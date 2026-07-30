from __future__ import annotations

import streamlit as st

from sempic.eval_dashboard.aggregate import build_metric_table, checkpoint_options
from sempic.eval_dashboard.charts import ExactChartTooLarge, build_cross_dataset_figure
from sempic.eval_dashboard.metrics import metric_label, metric_options

from .common import render_exports, select_id, status_badge


_MODES = {
    "Exact runs": "exact_runs",
    "Dataset-matched algorithm rollup": "dataset_matched",
    "Shared checkpoint": "shared_checkpoint",
}


def render_cross_dataset(frame) -> None:
    st.subheader("Cross-dataset")
    st.caption(
        "Compare datasets without hiding checkpoint membership. Exact observations are the default."
    )
    metrics = metric_options(frame)
    if not metrics:
        st.info("No numeric metrics are available for cross-dataset analysis.")
        return

    metric_column, mode_column = st.columns(2)
    with metric_column:
        metric = select_id(
            "Metric",
            metrics,
            key="eval_dashboard.cross_dataset.metric",
            labels={value: metric_label(value) for value in metrics},
        )
    with mode_column:
        mode_label = st.selectbox(
            "Analysis mode",
            tuple(_MODES),
            key="eval_dashboard.cross_dataset.mode",
        )
    mode = _MODES[mode_label]

    checkpoint_id = None
    if mode == "shared_checkpoint":
        options = checkpoint_options(frame)
        if options.empty:
            st.info("No checkpoint-backed observations are available in this scope.")
            return
        checkpoint_ids = options["checkpoint_id"].astype(str).tolist()
        checkpoint_labels = dict(
            zip(checkpoint_ids, options["checkpoint_label"].astype(str).tolist())
        )
        checkpoint_id = select_id(
            "Exact checkpoint",
            checkpoint_ids,
            key="eval_dashboard.cross_dataset.checkpoint",
            labels=checkpoint_labels,
        )

    result = build_metric_table(
        frame,
        metric,
        mode=mode,
        checkpoint_id=checkpoint_id,
    )
    aggregated = mode != "exact_runs"
    status_badge(result.aggregation_label, aggregated=aggregated)
    st.caption(
        f"Estimator: {result.estimator}. {result.grouping_description}"
    )
    if mode == "dataset_matched":
        st.warning(
            "This explicit rollup can combine different checkpoints that share an inferred "
            "algorithm variant. Use provenance before drawing checkpoint-level conclusions."
        )

    diagnostics = (
        ("Included observations", result.included_rows),
        ("Distinct checkpoints", len(result.included_checkpoints)),
        ("Cross-dataset excluded", result.excluded_cross_dataset),
        ("Unresolved excluded", result.excluded_unresolved),
    )
    for column, (label, value) in zip(st.columns(4), diagnostics):
        column.metric(label, int(value))

    if result.table.empty:
        st.info("No rows are available for this metric and analysis mode.")
    else:
        st.dataframe(result.table, width="stretch", hide_index=True)

        included_provenance = result.provenance.loc[
            result.provenance["included"].astype(bool)
        ]
        eligible_paths = set(included_provenance.get("source_path", ()))
        chart_frame = (
            frame.loc[frame["source_path"].isin(eligible_paths)]
            if eligible_paths
            else frame.iloc[0:0]
        )
        try:
            figure = build_cross_dataset_figure(
                chart_frame,
                metric,
                title=f"{metric_label(metric)} across datasets",
            )
        except ExactChartTooLarge as exc:
            st.warning(
                f"This exact chart contains {exc.count:,} observations, above the "
                f"{exc.limit:,}-observation rendering limit. Narrow the shared scope "
                "or choose a stricter analysis mode. No observations were sampled."
            )
            figure = None
        if figure is not None:
            st.caption(
                "The chart shows included exact observations; the table applies the selected mode."
            )
            st.plotly_chart(figure, width="stretch")

    with st.expander("Provenance", expanded=aggregated):
        st.caption(
            "Source membership for every displayed exact row or aggregate contribution."
        )
        st.dataframe(result.provenance, width="stretch", hide_index=True)
    if not result.table.empty:
        with st.expander("Download displayed table"):
            render_exports(
                lambda: result.table,
                stem=f"eval_cross_dataset_{mode}",
                key_prefix="eval_dashboard.cross_dataset.download",
            )
