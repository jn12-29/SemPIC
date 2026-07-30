from __future__ import annotations

import colorsys
import hashlib

import pandas as pd
import plotly.graph_objects as go

from sempic.eval_dashboard.metrics import (
    metric_is_lower_better,
    metric_label,
    metric_numeric,
    metric_unit,
)

_ALGORITHM_COLORS = {
    "full_recompute": "#0072B2",
    "no_cache": "#D55E00",
    "no_recompute": "#009E73",
    "kvpacket": "#CC79A7",
    "sempic": "#E69F00",
    "sempic_kvpacket": "#6A3D9A",
    "a3": "#88CCEE",
    "cache_blend": "#8C564B",
    "epic": "#DADA6C",
    "rand_recompute": "#4D4D4D",
    "sam_kv": "#17A2A4",
}
_GRID_COLOR = "#E2E8F0"
_TEXT_COLOR = "#1E293B"
_MUTED_TEXT_COLOR = "#64748B"
_MARKER_LINE_COLOR = "#FFFFFF"
_WEBGL_THRESHOLD = 500
_MAX_EXACT_OBSERVATIONS = 5_000
_MAX_RANK_TICKS = 30


class ExactChartTooLarge(ValueError):
    """Signal that an exact chart would exceed its honest rendering limit."""

    def __init__(self, count: int, limit: int = _MAX_EXACT_OBSERVATIONS) -> None:
        self.count = count
        self.limit = limit
        super().__init__(
            f"Exact chart has {count:,} observations; the limit is {limit:,}."
        )


def _normalized_method(method: object) -> str:
    normalized = str(method)
    return "kvpacket" if normalized == "kv_packet" else normalized


def _hex_color(red: float, green: float, blue: float) -> str:
    channels = (red, green, blue)
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in channels)


def algorithm_color(method: str) -> str:
    """Return the stable color family for a normalized algorithm method."""
    normalized = _normalized_method(method)
    curated = _ALGORITHM_COLORS.get(normalized)
    if curated is not None:
        return curated

    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535
    red, green, blue = colorsys.hls_to_rgb(hue, 0.42, 0.62)
    return _hex_color(red, green, blue)


def series_color(method: str, series_id: str) -> str:
    """Return a deterministic shade within an algorithm's color family."""
    base = algorithm_color(method)
    red, green, blue = (int(base[index : index + 2], 16) / 255 for index in (1, 3, 5))
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    digest = hashlib.sha256(str(series_id).encode("utf-8")).digest()
    saturation_shift = (digest[0] / 255 - 0.5) * 0.18
    lightness_shift = (digest[1] / 255 - 0.5) * 0.16
    saturation = (
        0.0
        if saturation < 0.08
        else min(0.82, max(0.42, saturation + saturation_shift))
    )
    lightness = min(0.62, max(0.30, lightness + lightness_shift))
    return _hex_color(*colorsys.hls_to_rgb(hue, lightness, saturation))


def _value_format(metric: str, axis: str) -> str:
    if metric == "metric.flops":
        return f"%{{{axis}:.3s}}"
    unit = metric_unit(metric)
    if unit:
        return f"%{{{axis}:.3f}} {unit}"
    return f"%{{{axis}:.4f}}"


def _axis_tick_format(metric: str) -> str | None:
    if metric == "metric.flops":
        return "~s"
    if metric_unit(metric):
        return ".3f"
    return None


def _text_values(frame: pd.DataFrame, column: str, fallback: str = "") -> pd.Series:
    values = frame.get(column, pd.Series(fallback, index=frame.index, dtype=object))
    return values.map(lambda value: value if isinstance(value, str) else fallback)


def _exact_metric_rows(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if frame.empty:
        return frame.iloc[0:0].copy()
    numeric = metric_numeric(frame, metric)
    selected = frame.loc[numeric.notna()].copy()
    selected[metric] = numeric.loc[selected.index]
    return selected


def _guard_exact_chart_size(rows: pd.DataFrame) -> None:
    if len(rows) > _MAX_EXACT_OBSERVATIONS:
        raise ExactChartTooLarge(len(rows))


def _add_exact_scatter(
    figure: go.Figure,
    observation_count: int,
    **kwargs: object,
) -> None:
    if observation_count > _WEBGL_THRESHOLD:
        figure.add_scattergl(**kwargs)
    else:
        figure.add_scatter(**kwargs)


def _rank_ticks(count: int) -> tuple[int, ...]:
    if count <= _MAX_RANK_TICKS:
        return tuple(range(1, count + 1))
    return tuple(
        1 + round(index * (count - 1) / (_MAX_RANK_TICKS - 1))
        for index in range(_MAX_RANK_TICKS)
    )


def _provenance_data(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": _text_values(rows, "method_label").where(
                _text_values(rows, "method_label").ne(""),
                _text_values(rows, "method"),
            ),
            "run": _text_values(rows, "run_label"),
            "checkpoint": _text_values(rows, "checkpoint_label"),
            "checkpoint_id": _text_values(rows, "checkpoint_id"),
            "source": _text_values(rows, "source_path"),
        },
        index=rows.index,
    )


def _method_groups(rows: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    methods = _text_values(rows, "method")
    grouped = rows.assign(_chart_method=methods)
    return [
        (str(method), method_rows)
        for method, method_rows in grouped.groupby("_chart_method", sort=True)
    ]


def _apply_theme(figure: go.Figure, *, title: str, show_legend: bool) -> None:
    figure.update_layout(
        template="plotly_white",
        title={
            "text": title,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 18, "color": "#0F172A"},
        },
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={"family": "Arial, sans-serif", "color": _TEXT_COLOR, "size": 13},
        hoverlabel={
            "bgcolor": "#0F172A",
            "bordercolor": "#0F172A",
            "font": {"color": "#FFFFFF", "family": "Arial, sans-serif"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255,255,255,0)",
            "title": {"text": ""},
        },
        margin={"l": 72, "r": 28, "t": 104 if show_legend else 74, "b": 68},
        hovermode="closest",
        bargap=0.22,
        bargroupgap=0.08,
        showlegend=show_legend,
    )
    figure.update_xaxes(
        showline=True,
        linecolor="#CBD5E1",
        gridcolor=_GRID_COLOR,
        griddash="dot",
        tickfont={"color": _MUTED_TEXT_COLOR},
        title_font={"color": "#334155"},
        automargin=True,
    )
    figure.update_yaxes(
        showline=False,
        gridcolor=_GRID_COLOR,
        griddash="dot",
        zerolinecolor="#CBD5E1",
        tickfont={"color": _MUTED_TEXT_COLOR},
        title_font={"color": "#334155"},
        automargin=True,
    )


def build_run_metric_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
) -> go.Figure | None:
    """Plot one mark per exact result observation, sorted by the selected metric."""
    rows = _exact_metric_rows(frame, metric)
    if rows.empty:
        return None
    _guard_exact_chart_size(rows)

    source = _text_values(rows, "source_path")
    rows = rows.assign(_chart_source=source).sort_values(
        [metric, "_chart_source"],
        ascending=[metric_is_lower_better(metric), True],
        kind="stable",
    )
    rows = rows.assign(_chart_rank=range(1, len(rows) + 1))
    figure = go.Figure()
    for method, method_rows in _method_groups(rows):
        provenance = _provenance_data(method_rows)
        series_ids = _text_values(method_rows, "series_id", fallback=method)
        colors = [series_color(method, series_id) for series_id in series_ids]
        label = provenance["method"].replace("", method).iloc[0]
        _add_exact_scatter(
            figure,
            len(rows),
            name=label,
            x=method_rows[metric],
            y=method_rows["_chart_rank"],
            mode="markers",
            marker={
                "color": colors,
                "size": 11,
                "line": {"color": _MARKER_LINE_COLOR, "width": 1.4},
                "opacity": 0.92,
            },
            customdata=provenance,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{metric_label(metric)}={_value_format(metric, 'x')}<br>"
                + "Run=%{customdata[1]}<br>"
                + "Checkpoint=%{customdata[2]}<br>"
                + "Checkpoint ID=%{customdata[3]}<br>"
                + "Source=%{customdata[4]}<extra></extra>"
            ),
        )

    figure.update_xaxes(
        title_text=metric_label(metric),
        tickformat=_axis_tick_format(metric),
        rangemode="tozero" if (rows[metric] >= 0).all() else "normal",
    )
    ticks = _rank_ticks(len(rows))
    figure.update_yaxes(
        title_text="Exact observations",
        tickmode="array",
        tickvals=ticks,
        ticktext=[f"#{rank}" for rank in ticks],
        autorange="reversed",
        showgrid=False,
    )
    figure.update_layout(height=max(360, min(920, 180 + 26 * len(rows))))
    _apply_theme(
        figure,
        title=title or f"{metric_label(metric)} · exact runs (n={len(rows)})",
        show_legend=True,
    )
    return figure


def build_cross_dataset_figure(
    frame: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
) -> go.Figure | None:
    """Plot exact observations across datasets without connecting or averaging them."""
    rows = _exact_metric_rows(frame, metric)
    if rows.empty:
        return None
    _guard_exact_chart_size(rows)

    dataset_labels = _text_values(rows, "benchmark_label").where(
        _text_values(rows, "benchmark_label").ne(""),
        _text_values(rows, "dataset_name", fallback="Unknown dataset"),
    )
    rows = rows.assign(_chart_dataset=dataset_labels)
    datasets = sorted(rows["_chart_dataset"].unique())
    dataset_positions = {dataset: index for index, dataset in enumerate(datasets)}
    methods = sorted(_text_values(rows, "method").unique())
    method_offsets = {
        method: (index - (len(methods) - 1) / 2)
        * min(0.12, 0.5 / max(len(methods), 1))
        for index, method in enumerate(methods)
    }

    figure = go.Figure()
    for method, method_rows in _method_groups(rows):
        provenance = _provenance_data(method_rows)
        x_values = []
        for dataset, source in zip(
            method_rows["_chart_dataset"],
            _text_values(method_rows, "source_path"),
            strict=True,
        ):
            digest = hashlib.sha256(source.encode("utf-8")).digest()
            observation_offset = (digest[0] / 255 - 0.5) * 0.04
            x_values.append(
                dataset_positions[dataset]
                + method_offsets.get(method, 0.0)
                + observation_offset
            )
        label = provenance["method"].replace("", method).iloc[0]
        customdata = provenance.assign(dataset=method_rows["_chart_dataset"])
        _add_exact_scatter(
            figure,
            len(rows),
            name=label,
            x=x_values,
            y=method_rows[metric],
            mode="markers",
            marker={
                "color": algorithm_color(method),
                "size": 10,
                "line": {"color": _MARKER_LINE_COLOR, "width": 1.2},
                "opacity": 0.82,
            },
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Dataset=%{customdata[5]}<br>"
                + f"{metric_label(metric)}={_value_format(metric, 'y')}<br>"
                + "Run=%{customdata[1]}<br>"
                + "Checkpoint=%{customdata[2]}<br>"
                + "Checkpoint ID=%{customdata[3]}<br>"
                + "Source=%{customdata[4]}<extra></extra>"
            ),
        )

    figure.update_xaxes(
        title_text="Evaluation dataset",
        tickmode="array",
        tickvals=list(range(len(datasets))),
        ticktext=datasets,
        range=[-0.5, len(datasets) - 0.5],
        showgrid=False,
    )
    figure.update_yaxes(
        title_text=metric_label(metric),
        tickformat=_axis_tick_format(metric),
        rangemode="tozero" if (rows[metric] >= 0).all() else "normal",
    )
    _apply_theme(
        figure,
        title=title or f"{metric_label(metric)} across datasets · exact (n={len(rows)})",
        show_legend=True,
    )
    return figure


def build_run_tradeoff_figure(
    frame: pd.DataFrame,
    efficiency_metric: str,
    *,
    title: str | None = None,
) -> go.Figure | None:
    """Plot one exact observation per mark for F1 versus an efficiency metric."""
    if frame.empty:
        return None
    efficiency = metric_numeric(frame, efficiency_metric)
    quality = metric_numeric(frame, "metric.f1")
    rows = frame.loc[efficiency.notna() & quality.notna()].copy()
    if rows.empty:
        return None
    _guard_exact_chart_size(rows)
    rows[efficiency_metric] = efficiency.loc[rows.index]
    rows["metric.f1"] = quality.loc[rows.index]

    figure = go.Figure()
    for method, method_rows in _method_groups(rows):
        provenance = _provenance_data(method_rows)
        series_ids = _text_values(method_rows, "series_id", fallback=method)
        label = provenance["method"].replace("", method).iloc[0]
        _add_exact_scatter(
            figure,
            len(rows),
            name=label,
            x=method_rows[efficiency_metric],
            y=method_rows["metric.f1"],
            mode="markers",
            marker={
                "color": [series_color(method, value) for value in series_ids],
                "size": 11,
                "line": {"color": _MARKER_LINE_COLOR, "width": 1.3},
                "opacity": 0.88,
            },
            customdata=provenance,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{metric_label(efficiency_metric)}="
                + _value_format(efficiency_metric, "x")
                + "<br>F1=%{y:.4f}<br>"
                + "Run=%{customdata[1]}<br>"
                + "Checkpoint=%{customdata[2]}<br>"
                + "Checkpoint ID=%{customdata[3]}<br>"
                + "Source=%{customdata[4]}<extra></extra>"
            ),
        )
    figure.update_xaxes(
        title_text=metric_label(efficiency_metric),
        tickformat=_axis_tick_format(efficiency_metric),
    )
    figure.update_yaxes(title_text="F1", rangemode="tozero")
    _apply_theme(
        figure,
        title=title or f"F1 vs {metric_label(efficiency_metric)} · exact runs",
        show_legend=True,
    )
    return figure
