from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pandas as pd
import streamlit as st

from sempic.eval_dashboard.aggregate import (
    frame_to_csv,
    frame_to_latex,
    frame_to_markdown,
)
from sempic.eval_dashboard.metrics import (
    metric_is_lower_better,
    metric_label,
)
from sempic.eval_dashboard.state import SelectionState, reconcile_multiselect


def metric_sort_label(metric: str) -> str:
    direction = "low to high" if metric_is_lower_better(metric) else "high to low"
    return f"{metric_label(metric)} · {direction}"


def select_id(
    label: str,
    options: Sequence[str],
    *,
    key: str,
    labels: dict[str, str] | None = None,
) -> str:
    if not options:
        raise ValueError(f"{label} requires at least one option")
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return st.selectbox(
        label,
        options,
        key=key,
        format_func=None if labels is None else labels.__getitem__,
    )


def _on_multiselect_change(key: str) -> None:
    state_key = f"{key}.selection"
    options_key = f"{key}.available"
    previous = st.session_state.get(state_key)
    st.session_state[state_key] = reconcile_multiselect(
        st.session_state.get(options_key, ()),
        previous,
        action="customize",
        user_selection=st.session_state.get(key, ()),
    )


def _on_multiselect_action(key: str, action: str) -> None:
    state_key = f"{key}.selection"
    options_key = f"{key}.available"
    result = reconcile_multiselect(
        st.session_state.get(options_key, ()),
        st.session_state.get(state_key),
        action=action,
    )
    st.session_state[state_key] = result
    st.session_state[key] = list(result.selected)


def render_multiselect(
    label: str,
    options: Sequence[str],
    *,
    key: str,
) -> tuple[str, ...]:
    """Render a selection whose follow-all intent is stored explicitly."""
    available = tuple(dict.fromkeys(options))
    state_key = f"{key}.selection"
    options_key = f"{key}.available"
    previous = st.session_state.get(state_key)
    selection = reconcile_multiselect(available, previous)
    st.session_state[state_key] = selection
    st.session_state[options_key] = available
    if st.session_state.get(key) != list(selection.selected):
        st.session_state[key] = list(selection.selected)

    st.multiselect(
        label,
        available,
        key=key,
        on_change=_on_multiselect_change,
        args=(key,),
    )
    first, second = st.columns(2)
    first.button(
        "All",
        key=f"{key}.all",
        on_click=_on_multiselect_action,
        args=(key, "select_all"),
        width="stretch",
    )
    second.button(
        "Clear",
        key=f"{key}.clear",
        on_click=_on_multiselect_action,
        args=(key, "clear"),
        width="stretch",
    )
    current = st.session_state.get(state_key)
    if not isinstance(current, SelectionState):
        return ()
    return current.selected


def status_badge(label: str, *, aggregated: bool = False) -> None:
    css_class = "sempic-status aggregate" if aggregated else "sempic-status"
    st.markdown(
        f'<span class="{css_class}">{label}</span>', unsafe_allow_html=True
    )


def json_value(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def render_run_detail(row: pd.Series, *, heading: str = "Run detail") -> None:
    st.markdown(f"#### {heading}")
    source_path = str(row.get("source_path", ""))
    st.code(source_path, language=None)
    config_column, result_column = st.columns(2)
    with config_column:
        st.markdown("**Resolved config**")
        st.json(json_value(row.get("config_json", row.get("config", {}))), expanded=False)
    with result_column:
        st.markdown("**Result**")
        st.json(json_value(row.get("result_json", row.get("result", {}))), expanded=False)


def render_exports(
    frame_factory: Callable[[], pd.DataFrame],
    *,
    stem: str,
    key_prefix: str,
) -> None:
    exports = {
        "CSV": (frame_to_csv, "csv", "text/csv"),
        "Markdown": (frame_to_markdown, "md", "text/markdown"),
        "LaTeX": (frame_to_latex, "tex", "application/x-tex"),
    }
    format_column, prepare_column = st.columns(2)
    with format_column:
        format_name = st.selectbox(
            "Format",
            tuple(exports),
            key=f"{key_prefix}.format",
        )
    with prepare_column:
        prepare = st.button(
            "Prepare download",
            key=f"{key_prefix}.prepare",
            width="stretch",
        )
    if not prepare:
        return

    serializer, suffix, mime = exports[format_name]
    content = serializer(frame_factory())
    st.download_button(
        f"Download {format_name}",
        data=content,
        file_name=f"{stem}.{suffix}",
        mime=mime,
        key=f"{key_prefix}.download.{suffix}",
        on_click="ignore",
        width="stretch",
    )
