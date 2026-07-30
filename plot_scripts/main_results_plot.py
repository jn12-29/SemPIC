from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt


LEGEND_ORDER = (
    "sempic",
    "sempic_kvpacket",
    "kvpacket",
    "full_recompute",
    "no_recompute",
    "no_cache",
    "rand_recompute",
    "a3",
    "sam_kv",
    "cache_blend",
    "epic",
)
BOLD_LEGEND_SERIES = {"sempic", "sempic_kvpacket"}
LEGEND_NAME_ALIASES = {"kv_packet": "kvpacket"}

SERIES_DISPLAY_NAMES = {
    "a3": "A3",
    "cache_blend": "Cache Blend",
    "epic": "EPIC",
    "full_recompute": "Full Recompute",
    "kv_packet": "KVPacket",
    "kvpacket": "KVPacket",
    "no_cache": "No Cache",
    "no_recompute": "No Recompute",
    "rand_recompute": "Random Recompute",
    "sam_kv": "SAM-KV",
    "sempic": "SemPIC",
    "sempic_kvpacket": "Joint",
}

DATASET_DISPLAY_NAMES = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MusiQue",
    "niah": "Needle-in-a-Haystack",
}

STYLE_CONFIG: dict[str, dict[str, Any]] = {
    "no_recompute": {"color": "gray", "marker": "X", "s": 100, "zorder": 2},
    "full_recompute": {"color": "black", "marker": "s", "s": 100, "zorder": 2},
    "no_cache": {"color": "red", "marker": "d", "s": 100, "zorder": 2},
    "a3": {"color": "brown", "marker": "v", "s": 100, "zorder": 2},
    "cache_blend": {"color": "blue", "marker": "o", "s": 80, "zorder": 2},
    "epic": {"color": "green", "marker": "^", "s": 80, "zorder": 2},
    "kv_packet": {
        "color": "#6F42C1", "marker": "D", "s": 125, "zorder": 8,
        "edgecolors": "black", "linewidths": 0.5,
    },
    "kvpacket": {
        "color": "#6F42C1", "marker": "D", "s": 125, "zorder": 8,
        "edgecolors": "black", "linewidths": 0.5,
    },
    "sempic": {
        "color": "#0072B2", "marker": "*", "s": 180, "zorder": 10,
        "edgecolors": "black", "linewidths": 0.5,
    },
    "sempic_kvpacket": {
        "color": "#009E73", "marker": "P", "s": 165, "zorder": 9,
        "edgecolors": "black", "linewidths": 0.5,
    },
    "rand_recompute": {"color": "orange", "marker": ".", "s": 80, "zorder": 2},
    "sam_kv": {"color": "gold", "marker": "P", "s": 80, "zorder": 2},
    "default": {"color": "orange", "marker": ".", "s": 50, "zorder": 1},
}


def display_series_name(name: str) -> str:
    return SERIES_DISPLAY_NAMES.get(name, name)


def display_dataset_name(name: str) -> str:
    return DATASET_DISPLAY_NAMES.get(name, name)


def canonical_legend_name(name: str) -> str:
    return LEGEND_NAME_ALIASES.get(name, name)


def arrange_legend_names(names: list[str]) -> tuple[list[str], int]:
    canonical_names = list(dict.fromkeys(canonical_legend_name(name) for name in names))
    priorities = {name: index for index, name in enumerate(LEGEND_ORDER)}
    ordered = sorted(
        canonical_names, key=lambda name: priorities.get(name, len(priorities))
    )
    column_count = (len(ordered) + 1) // 2
    if len(ordered) < 2:
        return ordered, column_count

    top_row = ordered[:column_count]
    bottom_row = ordered[column_count:]
    column_major: list[str] = []
    for column_index, top_name in enumerate(top_row):
        column_major.append(top_name)
        if column_index < len(bottom_row):
            column_major.append(bottom_row[column_index])
    return column_major, column_count


def _series_style(series: dict[str, Any]) -> dict[str, Any]:
    style = STYLE_CONFIG.get(series["name"], STYLE_CONFIG["default"]).copy()
    for key in ("color", "marker"):
        if key in series:
            style[key] = series[key]
    return style


def _draw_dataset_axis(
    axis: Any,
    dataset: dict[str, Any],
    x_key: str,
    x_label: str,
    *,
    show_title: bool,
    show_ylabel: bool,
    legend_handles: dict[str, Any],
) -> None:
    for series in dataset["series"]:
        points = series["points"]
        style = _series_style(series)
        if len(points) > 1:
            sorted_points = sorted(points, key=lambda point: point[x_key])
            axis.plot(
                [point[x_key] for point in sorted_points],
                [point["f1"] for point in sorted_points],
                color=style["color"], linestyle="--", linewidth=1.6,
                alpha=0.5, zorder=style["zorder"] - 1,
            )

        for point_index, point in enumerate(points):
            handle = axis.scatter(
                point[x_key], point["f1"],
                label=display_series_name(series["name"]) if point_index == 0 else None,
                alpha=0.7, **style,
            )
            if point_index == 0:
                legend_handles.setdefault(canonical_legend_name(series["name"]), handle)
            if point.get("annotate") and point.get("label"):
                axis.annotate(
                    point["label"], (point[x_key], point["f1"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, alpha=0.7,
                )

    if show_title:
        axis.set_title(display_dataset_name(dataset["name"]), fontsize=14, pad=15)
    if show_ylabel:
        axis.set_ylabel("F1 Score", fontsize=12)
    axis.set_xlabel(x_label, fontsize=11)
    axis.grid(True, linestyle="--", alpha=0.4)


def _add_global_legend(figure: Any, legend_handles: dict[str, Any]) -> int:
    if not legend_handles:
        return 0
    legend_names, legend_columns = arrange_legend_names(list(legend_handles))
    width, height = figure.get_size_inches()
    figure.set_size_inches(max(width, 2.1 * legend_columns), height)
    legend = figure.legend(
        [legend_handles[name] for name in legend_names],
        [display_series_name(name) for name in legend_names],
        loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=legend_columns,
        frameon=True, edgecolor="black",
    )
    for name, text in zip(legend_names, legend.get_texts()):
        if name in BOLD_LEGEND_SERIES:
            text.set_fontweight("bold")
    return 2 if len(legend_handles) > 1 else 1


def _save_figure(
    figure: Any, legend_handles: dict[str, Any], output_file: str | Path
) -> None:
    legend_rows = _add_global_legend(figure, legend_handles)
    legend_height = 1.35 if legend_rows == 2 else 0.85
    bottom = legend_height / figure.get_size_inches()[1]
    figure.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=bottom)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")


def _plot_v1(data: dict[str, Any], output_file: str | Path) -> None:
    datasets = data["datasets"]
    figure, axes = plt.subplots(
        2, len(datasets), figsize=(3 * len(datasets), 6), squeeze=False
    )
    legend_handles: dict[str, Any] = {}
    try:
        for column_index, dataset in enumerate(datasets):
            _draw_dataset_axis(
                axes[0, column_index], dataset, "flops", "FLOPs",
                show_title=True, show_ylabel=column_index == 0,
                legend_handles=legend_handles,
            )
            _draw_dataset_axis(
                axes[1, column_index], dataset, "ttft", "TTFT",
                show_title=False, show_ylabel=column_index == 0,
                legend_handles=legend_handles,
            )
        _save_figure(figure, legend_handles, output_file)
    finally:
        plt.close(figure)


def _plot_v2(data: dict[str, Any], output_file: str | Path) -> None:
    models = data["models"]
    column_count = max(len(model["datasets"]) for model in models)
    figure = plt.figure(figsize=(3 * column_count, 5.2 * len(models) + 0.8))
    height_ratios = [ratio for _ in models for ratio in (0.16, 1.0, 1.0)]
    grid = figure.add_gridspec(
        3 * len(models), column_count, height_ratios=height_ratios, hspace=0.45
    )
    legend_handles: dict[str, Any] = {}
    try:
        for model_index, model in enumerate(models):
            base_row = 3 * model_index
            header_axis = figure.add_subplot(grid[base_row, :])
            header_axis.axis("off")
            header_axis.text(
                0.5, 0.5, model["display_name"], ha="center", va="center",
                fontsize=16, fontweight="bold",
            )
            for column_index in range(column_count):
                flops_axis = figure.add_subplot(grid[base_row + 1, column_index])
                ttft_axis = figure.add_subplot(grid[base_row + 2, column_index])
                if column_index >= len(model["datasets"]):
                    flops_axis.set_visible(False)
                    ttft_axis.set_visible(False)
                    continue
                dataset = model["datasets"][column_index]
                _draw_dataset_axis(
                    flops_axis, dataset, "flops", "FLOPs", show_title=True,
                    show_ylabel=column_index == 0, legend_handles=legend_handles,
                )
                _draw_dataset_axis(
                    ttft_axis, dataset, "ttft", "TTFT", show_title=False,
                    show_ylabel=column_index == 0, legend_handles=legend_handles,
                )
        _save_figure(figure, legend_handles, output_file)
    finally:
        plt.close(figure)


def plot_results(data: dict[str, Any], output_file: str | Path) -> None:
    with plt.rc_context({"svg.fonttype": "path"}):
        if data["schema_version"] == 1:
            _plot_v1(data, output_file)
        elif data["schema_version"] == 2:
            _plot_v2(data, output_file)
        else:
            raise ValueError("schema_version must be 1 or 2")
