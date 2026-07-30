from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st

from sempic.eval_dashboard.aggregate import records_to_frame
from sempic.eval_dashboard.loader import (
    discover_result_directories,
    load_result_directories,
)
from sempic.eval_dashboard.query import QuerySpec, apply_query
from sempic.eval_dashboard.state import (
    DirectorySelection,
    transition_directory_selection,
)
from sempic.eval_dashboard.views import (
    render_audit,
    render_cross_dataset,
    render_experiment,
    render_runs,
)
from sempic.eval_dashboard.views.common import render_multiselect


_DIRECTORY_STATE_KEY = "eval_dashboard.directory_state"
_DIRECTORY_WIDGET_KEY = "eval_dashboard.directory_widget"
_DISCOVERED_KEY = "eval_dashboard.discovered"
_PENDING_DIRECTORY_ACTION_KEY = "eval_dashboard.pending_directory_action"

_DASHBOARD_STYLES = """
<style>
:root {
  --sempic-ink: #182230;
  --sempic-muted: #667085;
  --sempic-line: #e4e7ec;
  --sempic-panel: #ffffff;
  --sempic-accent: #2457c5;
}
.stApp { background: #f8fafc; color: var(--sempic-ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1500px; padding-top: 1.7rem; padding-bottom: 3rem; }
h1 { color: #101828; letter-spacing: -0.035em; margin-bottom: 0.15rem !important; }
h2, h3, h4 { color: #1d2939; letter-spacing: -0.018em; }
[data-testid="stCaptionContainer"] { color: var(--sempic-muted); }
[data-testid="stMetric"] {
  min-height: 5.6rem; padding: 0.8rem 0.95rem; background: var(--sempic-panel);
  border: 1px solid var(--sempic-line); border-radius: 0.7rem;
}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
  overflow: hidden; background: var(--sempic-panel);
  border: 1px solid var(--sempic-line); border-radius: 0.7rem;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  max-height: 7rem; overflow-y: auto; align-items: flex-start;
}
.sempic-status {
  display: inline-flex; align-items: center; padding: 0.22rem 0.55rem;
  border-radius: 999px; background: #eef4ff; color: #1849a9;
  font-size: 0.78rem; font-weight: 650; border: 1px solid #d1e0ff;
}
.sempic-status.aggregate { background: #fff6ed; color: #b54708; border-color: #ffead5; }
@media (max-width: 760px) {
  .block-container { padding-top: 1rem; }
  [data-testid="stMetric"] { min-height: 4.8rem; }
}
</style>
"""


def _store_directory_selection(selection: DirectorySelection) -> None:
    st.session_state[_DIRECTORY_STATE_KEY] = selection
    st.session_state[_DIRECTORY_WIDGET_KEY] = list(selection.selected)


def _on_directory_change() -> None:
    discovered = st.session_state.get(_DISCOVERED_KEY, ())
    previous = st.session_state.get(_DIRECTORY_STATE_KEY)
    selection = transition_directory_selection(
        discovered,
        previous,
        action="customize",
        user_selection=st.session_state.get(_DIRECTORY_WIDGET_KEY, ()),
    )
    _store_directory_selection(selection)


def _on_directory_action(action: Literal["select_all", "clear"]) -> None:
    st.session_state[_PENDING_DIRECTORY_ACTION_KEY] = action


def _source_kind(roots: Sequence[Path]) -> str:
    parts = [set(root.parts) for root in roots]
    has_stable = any(
        "eval_config" in value or (root / "eval_config").is_dir()
        for root, value in zip(roots, parts)
    )
    has_history = any(
        "eval_outputs" in value or (root / "eval_outputs").is_dir()
        for root, value in zip(roots, parts)
    )
    if has_stable and has_history:
        return "Mixed sources"
    if parts and all("eval_config" in value for value in parts):
        return "Stable results"
    if parts and all("eval_outputs" in value for value in parts):
        return "Run history"
    return "Custom roots"


def _format_directory(path: str, roots: Sequence[Path]) -> str:
    resolved = Path(path)
    for root in roots:
        try:
            return f"{resolved.relative_to(root)} — {resolved}"
        except ValueError:
            continue
    return str(resolved)


def _render_directory_controls(
    roots: tuple[Path, ...],
    discovered: tuple[str, ...],
    selection: DirectorySelection,
) -> None:
    source_kind = _source_kind(roots)
    with st.sidebar:
        st.markdown("### Data source")
        st.caption(source_kind)
        st.caption(
            f"{len(selection.selected):,} selected / {len(discovered):,} discovered directories"
        )
        st.button("Refresh now", width="stretch", key="eval_dashboard.refresh_now")

        with st.expander("Directory details", expanded=False):
            st.multiselect(
                "Selected result directories",
                discovered,
                format_func=lambda path: _format_directory(path, roots),
                key=_DIRECTORY_WIDGET_KEY,
                on_change=_on_directory_change,
            )
            first, second = st.columns(2)
            first.button(
                "Select all",
                key="eval_dashboard.directories.all",
                on_click=_on_directory_action,
                args=("select_all",),
                width="stretch",
            )
            second.button(
                "Clear",
                key="eval_dashboard.directories.clear",
                on_click=_on_directory_action,
                args=("clear",),
                width="stretch",
            )
            for root in roots:
                st.code(str(root), language=None)


def _render_shared_scope(frame: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:

        st.markdown("### Shared scope")
        models = tuple(sorted(frame["model_name"].dropna().astype(str).unique()))
        methods = tuple(sorted(frame["method"].dropna().astype(str).unique()))
        with st.expander("Models", expanded=False):
            selected_models = render_multiselect(
                "Models", models, key="eval_dashboard.scope.models"
            )
            st.caption(f"{len(selected_models):,} of {len(models):,} selected")
        with st.expander("Methods", expanded=False):
            selected_methods = render_multiselect(
                "Methods", methods, key="eval_dashboard.scope.methods"
            )
            st.caption(f"{len(selected_methods):,} of {len(methods):,} selected")

    if not selected_models or not selected_methods:
        return frame.iloc[0:0].copy()
    return apply_query(
        frame,
        QuerySpec(models=selected_models, methods=selected_methods),
    ).frame


def _render_header(
    frame: pd.DataFrame,
    *,
    source_kind: str,
    selected_directories: int,
    warning_count: int,
) -> None:
    title_column, status_column = st.columns([4, 1])
    with title_column:
        st.title("Evaluation dashboard")
        st.caption(
            "Exact runs, experiment comparison, cross-dataset analysis, and source audit."
        )
    with status_column:
        st.markdown(
            f'<span class="sempic-status">{source_kind} · {selected_directories} dirs</span>',
            unsafe_allow_html=True,
        )

    checkpoints = frame.get("checkpoint_id", pd.Series(dtype=object)).dropna()
    metrics = (
        ("Observations", len(frame)),
        ("Checkpoints", checkpoints.astype(str).loc[lambda values: values != ""].nunique()),
        ("Datasets", frame.get("dataset_name", pd.Series(dtype=object)).nunique()),
        ("Warnings", warning_count),
    )
    for column, (label, value) in zip(st.columns(4), metrics):
        column.metric(label, int(value))


def _render_empty_source(message: str, warnings: Sequence[str]) -> None:
    st.title("Evaluation dashboard")
    st.info(message)
    if warnings:
        with st.expander(f"Diagnostics ({len(warnings)})"):
            for warning in warnings:
                st.warning(warning)


def _render_refresh_cycle(roots: tuple[Path, ...]) -> None:
    discovery = discover_result_directories(roots)
    discovered = tuple(str(path) for path in discovery.directories)
    discovered_parts = [set(Path(path).parts) for path in discovered]
    if any("eval_config" in parts for parts in discovered_parts) and any(
        "eval_outputs" in parts for parts in discovered_parts
    ):
        _render_empty_source(
            "Stable results and run history were discovered under the same roots. "
            "Start separate dashboard processes for eval_config and eval_outputs.",
            discovery.warnings,
        )
        return
    st.session_state[_DISCOVERED_KEY] = discovered

    previous = st.session_state.get(_DIRECTORY_STATE_KEY)
    pending_action = st.session_state.pop(_PENDING_DIRECTORY_ACTION_KEY, "refresh")
    selection = transition_directory_selection(
        discovered,
        previous,
        action=pending_action,
    )
    _store_directory_selection(selection)
    _render_directory_controls(roots, discovered, selection)

    if not discovered:
        _render_empty_source(
            "No result directories were discovered. Check the server-local roots.",
            discovery.warnings,
        )
        return
    if not selection.selected:
        _render_empty_source(
            "No result directories are selected. Use the sidebar to select a source.",
            discovery.warnings,
        )
        return

    loaded = load_result_directories(selection.selected)
    warnings = (*discovery.warnings, *loaded.warnings)
    frame = records_to_frame(loaded.records)
    if frame.empty:
        _render_empty_source(
            "The selected directories contain no valid result records.", warnings
        )
        return

    scoped_frame = _render_shared_scope(frame)
    _render_header(
        scoped_frame,
        source_kind=_source_kind(roots),
        selected_directories=len(selection.selected),
        warning_count=len(warnings),
    )
    workflow = st.segmented_control(
        "Workflow",
        ("Runs", "Experiment", "Cross-dataset", "Audit"),
        default="Runs",
        required=True,
        width="stretch",
        key="eval_dashboard.workflow",
        label_visibility="collapsed",
    )
    if scoped_frame.empty:
        st.info("The shared model and method scope contains no observations.")
        if workflow == "Audit":
            render_audit(
                scoped_frame,
                warnings=warnings,
                roots=roots,
                directories=selection.selected,
            )
        return
    if workflow == "Runs":
        render_runs(scoped_frame)
    elif workflow == "Experiment":
        render_experiment(scoped_frame)
    elif workflow == "Cross-dataset":
        render_cross_dataset(scoped_frame)
    else:
        render_audit(
            scoped_frame,
            warnings=warnings,
            roots=roots,
            directories=selection.selected,
        )


def render_dashboard(roots: Sequence[str | Path]) -> None:
    """Render the read-only Streamlit dashboard for server-local result roots."""
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    st.set_page_config(page_title="SemPIC Eval Dashboard", layout="wide")
    st.markdown(_DASHBOARD_STYLES, unsafe_allow_html=True)
    if _source_kind(resolved_roots) == "Mixed sources":
        st.title("Evaluation dashboard")
        st.error(
            "Stable results and run history cannot be scanned together because they "
            "contain duplicate copies of completed evaluations. Start one dashboard "
            "process for eval_config or eval_outputs."
        )
        return

    refresh_options: dict[str, float | None] = {
        "Off": None,
        "5 seconds": 5.0,
        "15 seconds": 15.0,
        "30 seconds": 30.0,
        "60 seconds": 60.0,
    }
    with st.sidebar:
        refresh_label = st.selectbox(
            "Automatic refresh",
            tuple(refresh_options),
            index=0,
            key="eval_dashboard.refresh_interval",
        )

    @st.fragment(run_every=refresh_options[refresh_label])
    def refresh_fragment() -> None:
        _render_refresh_cycle(resolved_roots)

    refresh_fragment()


__all__ = [
    "DirectorySelection",
    "render_dashboard",
    "transition_directory_selection",
]
