from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from sempic.eval_dashboard.aggregate import summarize_results_with_provenance

from .common import render_exports, render_run_detail, select_id


def render_audit(
    frame: pd.DataFrame,
    *,
    warnings: Sequence[str],
    roots: Sequence[Path],
    directories: Sequence[str],
) -> None:
    st.subheader("Audit")
    st.caption(
        "Inspect source coverage, parser diagnostics, raw result provenance, and snapshot exports."
    )

    source_column, warning_column = st.columns(2)
    with source_column:
        st.markdown("#### Source directories")
        st.caption(f"{len(directories):,} selected directories")
        with st.expander("Server-local paths", expanded=False):
            for root in roots:
                st.code(f"Root: {root}", language=None)
            for directory in directories:
                st.code(directory, language=None)
    with warning_column:
        st.markdown("#### Diagnostics")
        if warnings:
            st.caption(f"{len(warnings):,} discovery or parsing warnings")
            with st.expander("Warning details", expanded=True):
                for warning in warnings:
                    st.warning(warning)
        else:
            st.success("No discovery or parsing warnings.")

    st.markdown("#### Raw observation")
    ordered = frame.sort_values("source_path", kind="stable")
    paths = ordered["source_path"].astype(str).tolist()
    if not paths:
        st.info("No observations are available in the shared scope.")
        return
    run_labels = dict(zip(paths, ordered["run_label"].astype(str).tolist()))
    selected_path = select_id(
        "Result source",
        paths,
        key="eval_dashboard.audit.source",
        labels={path: f"{run_labels[path]} · {path}" for path in paths},
    )
    render_run_detail(
        ordered.loc[ordered["source_path"].astype(str) == selected_path].iloc[0],
        heading="Source config and result",
    )

    st.markdown("#### Snapshot exports")
    st.caption(
        "Exact exports preserve source rows; metric summaries include estimator, grouping, "
        "and source/checkpoint membership."
    )
    exact_column, summary_column = st.columns(2)
    with exact_column:
        st.markdown("**Exact observations**")
        render_exports(
            lambda: frame,
            stem="eval_results_audit",
            key_prefix="eval_dashboard.audit.results",
        )
    with summary_column:
        st.markdown("**Auditable metric summary**")
        render_exports(
            lambda: summarize_results_with_provenance(frame),
            stem="eval_summary_audit",
            key_prefix="eval_dashboard.audit.summary",
        )
