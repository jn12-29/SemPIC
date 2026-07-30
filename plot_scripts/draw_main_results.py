import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.ticker import EngFormatter, FuncFormatter, MaxNLocator

try:
    from plot_scripts.main_results_data import (
        load_plot_document_files,
        suffix_output_path,
    )
    from plot_scripts.main_results_plot import (
        BOLD_LEGEND_SERIES,
        STYLE_CONFIG,
        _draw_dataset_axis,
        arrange_legend_names,
        canonical_legend_name,
        display_dataset_name,
        display_series_name,
        plot_results,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from main_results_data import load_plot_document_files, suffix_output_path
    from main_results_plot import (
        BOLD_LEGEND_SERIES,
        STYLE_CONFIG,
        _draw_dataset_axis,
        arrange_legend_names,
        canonical_legend_name,
        display_dataset_name,
        display_series_name,
        plot_results,
    )
from sempic.utils.run_storage import allocate_run_dir
from sempic.utils.runtime import RuntimeContext


LOGGER = logging.getLogger("sempic.plot_scripts.draw_main_results")
DEFAULT_OUTPUT_FILE = "plot_figs/gathered_results_plot.pdf"
LAYOUT_STANDARD = "standard"
LAYOUT_PAIRED_WIDE = "paired-wide"
LAYOUT_COMPACT_4X4 = "compact-4x4"
COMPACT_MARKER_SCALE_BY_SERIES = {"kvpacket": 0.60}


def _compact_axis(
    axis: Any,
    x_key: str,
    *,
    show_y_ticks: bool,
    marker_scale: float = 0.5,
    font_size: float = 7,
    x_tick_count: int | None = None,
    dataset: dict[str, Any] | None = None,
) -> None:
    collection_series_names = []
    if dataset is not None:
        collection_series_names = [
            canonical_legend_name(series["name"])
            for series in dataset["series"]
            for _ in series["points"]
        ]
    for index, collection in enumerate(axis.collections):
        series_name = (
            collection_series_names[index]
            if index < len(collection_series_names)
            else ""
        )
        series_scale = COMPACT_MARKER_SCALE_BY_SERIES.get(series_name, 1.0)
        collection.set_sizes(
            collection.get_sizes() * marker_scale * series_scale
        )
        collection.set_linewidths(collection.get_linewidths() * 0.75)
    for line in axis.lines:
        line.set_linewidth(0.9)

    axis.tick_params(
        axis="both", labelsize=font_size, pad=1.5, width=0.6, length=2.5
    )
    axis.xaxis.label.set_fontsize(font_size)
    axis.yaxis.label.set_fontsize(font_size)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2g}"))
    if x_key == "flops":
        axis.xaxis.set_major_locator(
            MaxNLocator(nbins=x_tick_count or 3, min_n_ticks=2)
        )
        axis.xaxis.set_major_formatter(EngFormatter(unit="", places=0, sep=""))
    else:
        axis.xaxis.set_major_locator(
            MaxNLocator(nbins=x_tick_count or 2, min_n_ticks=2)
        )
        axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")
            )
        )
    axis.xaxis.get_offset_text().set_visible(False)
    if not show_y_ticks:
        axis.tick_params(axis="y", labelleft=False)
    for spine in axis.spines.values():
        spine.set_linewidth(0.6)


def _add_compact_legend(
    figure: Any,
    legend_handles: dict[str, Any],
    *,
    placement: str = "bottom",
) -> None:
    if not legend_handles:
        return
    legend_names, legend_columns = arrange_legend_names(list(legend_handles))
    if placement == "right":
        legend_names = legend_names[::2] + legend_names[1::2]
        figure_width, _ = figure.get_size_inches()
        legend_axis = figure.add_axes(
            [1.0 - 0.94 / figure_width, 0.13, 0.90 / figure_width, 0.74]
        )
        legend_axis.set_xlim(0.0, 1.0)
        legend_axis.set_ylim(0.0, 1.0)
        legend_axis.set_xticks([])
        legend_axis.set_yticks([])
        legend_axis.set_facecolor("white")
        for spine in legend_axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.6)

        top = 0.92
        step = 0.84 / max(len(legend_names) - 1, 1)
        for index, name in enumerate(legend_names):
            y_position = top - index * step
            handle = legend_handles[name]
            facecolors = handle.get_facecolors()
            edgecolors = handle.get_edgecolors()
            linewidths = handle.get_linewidths()
            legend_axis.scatter(
                0.17,
                y_position,
                marker=handle.get_paths()[0],
                s=float(handle.get_sizes()[0]) * 1.05**2,
                facecolors=facecolors[0] if len(facecolors) else "none",
                edgecolors=edgecolors[0] if len(edgecolors) else "none",
                linewidths=float(linewidths[0]) if len(linewidths) else 0.0,
                alpha=handle.get_alpha(),
                clip_on=False,
            )
            label = display_series_name(name)
            if len(label) > 10 and " " in label:
                label = label.replace(" ", "\n")
            legend_axis.text(
                0.34,
                y_position,
                label,
                ha="left",
                va="center",
                fontsize=7,
                fontweight="bold" if name in BOLD_LEGEND_SERIES else "normal",
                linespacing=0.9,
            )
        return

    legend = figure.legend(
        [legend_handles[name] for name in legend_names],
        [display_series_name(name) for name in legend_names],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=legend_columns,
        fontsize=7,
        frameon=True,
        edgecolor="black",
        borderpad=0.35,
        columnspacing=0.8,
        handletextpad=0.35,
        labelspacing=0.25,
        markerscale=1.05,
    )
    for name, text in zip(legend_names, legend.get_texts()):
        if name in BOLD_LEGEND_SERIES:
            text.set_fontweight("bold")


def plot_results_paired_wide(
    data: dict[str, Any], output_file: str | Path
) -> None:
    if data["schema_version"] != 2:
        raise ValueError("paired-wide layout requires schema_version 2 plot data")

    models = data["models"]
    dataset_names = list(
        dict.fromkeys(
            dataset["name"]
            for model in models
            for dataset in model["datasets"]
        )
    )
    model_count = len(models)
    dataset_count = len(dataset_names)
    figure_width = max(7.16, 1.7 * dataset_count)
    figure_height = max(2.25, 1.18 * model_count + 1.0)
    left = 0.58 / figure_width
    right = 1.0 - 0.06 / figure_width
    bottom = 0.68 / figure_height
    top = 1.0 - 0.30 / figure_height

    figure = plt.figure(figsize=(figure_width, figure_height))
    outer_grid = figure.add_gridspec(
        model_count,
        dataset_count,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.3,
        hspace=0.42,
    )
    legend_handles: dict[str, Any] = {}
    row_axes: list[list[Any]] = []
    top_pairs: list[tuple[Any, Any]] = []

    try:
        for model_index, model in enumerate(models):
            datasets_by_name = {
                dataset["name"]: dataset for dataset in model["datasets"]
            }
            first_visible_dataset = next(
                name for name in dataset_names if name in datasets_by_name
            )
            axes_for_row: list[Any] = []
            for dataset_index, dataset_name in enumerate(dataset_names):
                pair_grid = outer_grid[model_index, dataset_index].subgridspec(
                    1, 2, wspace=0.18
                )
                flops_axis = figure.add_subplot(pair_grid[0, 0])
                ttft_axis = figure.add_subplot(pair_grid[0, 1], sharey=flops_axis)
                axes_for_row.extend((flops_axis, ttft_axis))
                if model_index == 0:
                    top_pairs.append((flops_axis, ttft_axis))

                dataset = datasets_by_name.get(dataset_name)
                if dataset is None:
                    flops_axis.set_visible(False)
                    ttft_axis.set_visible(False)
                    continue

                _draw_dataset_axis(
                    flops_axis,
                    dataset,
                    "flops",
                    "FLOPs",
                    show_title=False,
                    show_ylabel=dataset_name == first_visible_dataset,
                    legend_handles=legend_handles,
                )
                _draw_dataset_axis(
                    ttft_axis,
                    dataset,
                    "ttft",
                    "TTFT",
                    show_title=False,
                    show_ylabel=False,
                    legend_handles=legend_handles,
                )
                _compact_axis(flops_axis, "flops", show_y_ticks=True)
                _compact_axis(ttft_axis, "ttft", show_y_ticks=False)
            row_axes.append(axes_for_row)

        for dataset_name, (flops_axis, ttft_axis) in zip(
            dataset_names, top_pairs
        ):
            left_box = flops_axis.get_position()
            right_box = ttft_axis.get_position()
            figure.text(
                (left_box.x0 + right_box.x1) / 2,
                top + 0.055 / figure_height,
                display_dataset_name(dataset_name),
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

        for model, axes_for_row in zip(models, row_axes):
            visible_axes = [axis for axis in axes_for_row if axis.get_visible()]
            if not visible_axes:
                continue
            row_bottom = min(axis.get_position().y0 for axis in visible_axes)
            row_top = max(axis.get_position().y1 for axis in visible_axes)
            figure.text(
                0.012,
                (row_bottom + row_top) / 2,
                model["display_name"],
                ha="center",
                va="center",
                rotation=90,
                fontsize=7.5,
                fontweight="bold",
            )

        _add_compact_legend(figure, legend_handles)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def plot_results_compact_4x4(
    data: dict[str, Any], output_file: str | Path
) -> None:
    if data["schema_version"] != 2:
        raise ValueError("compact-4x4 layout requires schema_version 2 plot data")

    models = data["models"]
    dataset_names = list(
        dict.fromkeys(
            dataset["name"]
            for model in models
            for dataset in model["datasets"]
        )
    )
    model_count = len(models)
    dataset_count = len(dataset_names)
    figure_width = max(7.16, 1.7 * dataset_count)
    figure_height = max(3.4, 1.75 * model_count + 1.0)
    left = 0.62 / figure_width
    right = 1.0 - 1.00 / figure_width
    bottom = 0.32 / figure_height
    top = 1.0 - 0.30 / figure_height

    figure = plt.figure(figsize=(figure_width, figure_height))
    outer_grid = figure.add_gridspec(
        2,
        dataset_count,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        wspace=0.28,
        hspace=0.22,
    )
    legend_handles: dict[str, Any] = {}
    metric_axes: dict[str, list[list[Any]]] = {}
    top_axes: list[Any] = []

    try:
        for metric_index, x_key in enumerate(("flops", "ttft")):
            dataset_grids = [
                outer_grid[metric_index, dataset_index].subgridspec(
                    model_count, 1, hspace=0.06
                )
                for dataset_index in range(dataset_count)
            ]
            axes_for_metric: list[list[Any]] = []
            for model_index, model in enumerate(models):
                datasets_by_name = {
                    dataset["name"]: dataset for dataset in model["datasets"]
                }
                first_visible_dataset = next(
                    name for name in dataset_names if name in datasets_by_name
                )
                axes_for_model: list[Any] = []
                for dataset_index, dataset_name in enumerate(dataset_names):
                    share_x_axis = None
                    if model_index > 0:
                        share_x_axis = axes_for_metric[0][dataset_index]
                    share_axis = None
                    if x_key == "ttft":
                        share_axis = metric_axes["flops"][model_index][
                            dataset_index
                        ]
                    axis = figure.add_subplot(
                        dataset_grids[dataset_index][model_index, 0],
                        sharex=share_x_axis,
                        sharey=share_axis,
                    )
                    axes_for_model.append(axis)
                    if metric_index == 0 and model_index == 0:
                        top_axes.append(axis)

                    dataset = datasets_by_name.get(dataset_name)
                    if dataset is None:
                        axis.set_visible(False)
                        continue

                    _draw_dataset_axis(
                        axis,
                        dataset,
                        x_key,
                        "",
                        show_title=False,
                        show_ylabel=dataset_name == first_visible_dataset,
                        legend_handles=legend_handles,
                    )
                    _compact_axis(
                        axis,
                        x_key,
                        show_y_ticks=True,
                        marker_scale=0.48,
                        font_size=7.5,
                        x_tick_count=3,
                        dataset=dataset,
                    )
                    if model_index < model_count - 1:
                        axis.tick_params(axis="x", labelbottom=False)
                axes_for_metric.append(axes_for_model)
            metric_axes[x_key] = axes_for_metric

        for dataset_name, axis in zip(dataset_names, top_axes):
            box = axis.get_position()
            figure.text(
                (box.x0 + box.x1) / 2,
                top + 0.055 / figure_height,
                display_dataset_name(dataset_name),
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        for x_key, metric_label in (("flops", "FLOPs"), ("ttft", "TTFT")):
            visible_metric_axes = [
                axis
                for axes_for_model in metric_axes[x_key]
                for axis in axes_for_model
                if axis.get_visible()
            ]
            if visible_metric_axes:
                row_left = min(
                    axis.get_position().x0 for axis in visible_metric_axes
                )
                row_right = max(
                    axis.get_position().x1 for axis in visible_metric_axes
                )
                row_bottom = min(
                    axis.get_position().y0 for axis in visible_metric_axes
                )
                figure.text(
                    (row_left + row_right) / 2,
                    row_bottom - 0.15 / figure_height,
                    metric_label,
                    ha="center",
                    va="top",
                    fontsize=7.5,
                )

            for model, axes_for_model in zip(models, metric_axes[x_key]):
                visible_axes = [
                    axis for axis in axes_for_model if axis.get_visible()
                ]
                if not visible_axes:
                    continue
                row_bottom = min(axis.get_position().y0 for axis in visible_axes)
                row_top = max(axis.get_position().y1 for axis in visible_axes)
                figure.text(
                    0.014,
                    (row_bottom + row_top) / 2,
                    model["display_name"],
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=7,
                    fontweight="bold",
                )

        _add_compact_legend(figure, legend_handles, placement="right")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot F1/FLOPs and F1/TTFT tradeoffs from editable plot-data JSON."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="All-v1 or all-v2 plot-data JSON files applied left to right.",
    )
    parser.add_argument(
        "--output-file", default=DEFAULT_OUTPUT_FILE, help="Output image file."
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional suffix inserted before the output file extension.",
    )
    parser.add_argument(
        "--layout",
        choices=(LAYOUT_STANDARD, LAYOUT_PAIRED_WIDE, LAYOUT_COMPACT_4X4),
        default=LAYOUT_STANDARD,
        help=(
            "Plot layout; paired-wide groups metrics by dataset, while "
            "compact-4x4 places FLOPs and TTFT on separate rows per model."
        ),
    )
    args = parser.parse_args()

    output_file = suffix_output_path(args.output_file, args.run_suffix)
    run_dir = allocate_run_dir("./logs/draw_main_results", args.run_suffix)
    with RuntimeContext(
        entrypoint="draw_main_results",
        run_dir=run_dir,
        config_file=None,
        resolved_config=None,
        config_snapshot_name=None,
        cli_args=vars(args).copy(),
    ):
        data = load_plot_document_files(args.input_files)
        if data["schema_version"] == 1:
            LOGGER.info(
                "Loaded %d datasets from %d v1 plot-data files",
                len(data["datasets"]), len(args.input_files),
            )
        else:
            dataset_count = sum(len(model["datasets"]) for model in data["models"])
            LOGGER.info(
                "Loaded %d models and %d datasets from %d v2 plot-data files",
                len(data["models"]), dataset_count, len(args.input_files),
            )
        if args.layout == LAYOUT_PAIRED_WIDE:
            plot_results_paired_wide(data, output_file)
        elif args.layout == LAYOUT_COMPACT_4X4:
            plot_results_compact_4x4(data, output_file)
        else:
            plot_results(data, output_file)
        LOGGER.info("Saved plot to %s", output_file)


if __name__ == "__main__":
    main()
