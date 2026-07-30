"""Draw a compact block-local attention profile and token-detail figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, NullLocator

try:
    from plot_scripts.draw_block_token_sink_curve import (
        MODEL_STYLES,
        _geometric_mean,
        validate_plot_data as validate_token_data,
    )
    from plot_scripts.draw_multimodel_sink import (
        DATASET_ORDER,
        F1_COLOR,
        MODEL_LABEL_COLORS,
        MODEL_ORDER,
        _combined_rows,
        validate_plot_data as validate_profile_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from draw_block_token_sink_curve import (
        MODEL_STYLES,
        _geometric_mean,
        validate_plot_data as validate_token_data,
    )
    from draw_multimodel_sink import (
        DATASET_ORDER,
        F1_COLOR,
        MODEL_LABEL_COLORS,
        MODEL_ORDER,
        _combined_rows,
        validate_plot_data as validate_profile_data,
    )


DATASET_SHORT_LABELS = {
    "biography": "Bio",
    "hotpot_qa": "HQA",
    "musique": "MuS",
    "niah": "NIAH",
}
MODEL_SHORT_LABELS = {
    "Qwen3-4B-Instruct-2507": "Q4",
    "Qwen3-8B": "Q8",
    "Llama-3.1-8B-Instruct": "L8",
}
PROFILE_BIN_COUNT = 20


def _model_identity(models: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(model["model_id"]), str(model["display_name"])) for model in models
    )


def draw_combined_block_sink(
    profile_data: Any,
    token_data: Any,
    output_prefix: str | Path,
) -> dict[str, Path]:
    """Validate both inputs and write SVG, PDF, and PNG figure variants."""

    validated_profile = validate_profile_data(profile_data)
    token_models = validate_token_data(token_data)
    profile_models = list(validated_profile["models"])
    raw_token_models = list(token_data["models"])
    expected_identity = tuple(
        (model_id, str(MODEL_STYLES[model_id]["display_name"]))
        for model_id in MODEL_ORDER
    )
    if (
        _model_identity(profile_models) != expected_identity
        or _model_identity(raw_token_models) != expected_identity
    ):
        raise ValueError(
            "Profile and token data must contain the fixed models in the same order"
        )

    bin_edges, heatmap_rows, _sink_ratios, f1_changes, _display_names = (
        _combined_rows(validated_profile)
    )
    expected_bin_width = 1.0 / PROFILE_BIN_COUNT
    if (
        len(bin_edges) != PROFILE_BIN_COUNT + 1
        or any(len(row) != PROFILE_BIN_COUNT for row in heatmap_rows)
        or any(
            not math.isclose(
                end - start,
                expected_bin_width,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for start, end in zip(bin_edges[:-1], bin_edges[1:], strict=True)
        )
    ):
        raise ValueError(
            f"Combined figure requires {PROFILE_BIN_COUNT} equal-width profile bins"
        )
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.65,
            "lines.solid_capstyle": "round",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    row_count = len(MODEL_ORDER) * len(DATASET_ORDER)
    figure = plt.figure(figsize=(3.33, 2.35))
    heatmap_axis = figure.add_axes((0.17, 0.20, 0.39, 0.60))
    f1_axis = figure.add_axes((0.56, 0.20, 0.075, 0.60))
    token_axis = figure.add_axes((0.73, 0.20, 0.255, 0.53))

    values = [value for row in heatmap_rows for value in row]
    color_norm = LogNorm(vmin=min(0.5, min(values)), vmax=max(100.0, max(values)))
    mesh = heatmap_axis.pcolormesh(
        bin_edges,
        list(range(row_count + 1)),
        heatmap_rows,
        cmap="Blues",
        norm=color_norm,
        edgecolors="white",
        linewidth=0.25,
        shading="flat",
        rasterized=False,
    )
    heatmap_axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            row_count,
            fill=False,
            edgecolor="#666666",
            linewidth=0.65,
        )
    )
    for boundary in (0.1, 0.9):
        heatmap_axis.axvline(
            boundary,
            color="#333333",
            linestyle=(0, (2, 1.5)),
            linewidth=0.8,
            zorder=3,
        )
    for boundary in (len(DATASET_ORDER), 2 * len(DATASET_ORDER)):
        heatmap_axis.axhline(
            boundary,
            color="#4C4C4C",
            linewidth=0.8,
            zorder=4,
        )
        f1_axis.axhline(
            boundary,
            color="#A0A0A0",
            linewidth=0.65,
            zorder=1,
        )

    heatmap_axis.set_xlim(0.0, 1.0)
    heatmap_axis.set_ylim(0.0, row_count)
    heatmap_axis.invert_yaxis()
    row_datasets = list(DATASET_ORDER) * len(MODEL_ORDER)
    heatmap_axis.set_yticks(
        [index + 0.5 for index in range(row_count)],
        [DATASET_SHORT_LABELS[dataset] for dataset in row_datasets],
    )
    heatmap_axis.set_xticks(
        (0.0, 0.1, 0.5, 0.9, 1.0),
        ("0", ".1", ".5", ".9", "1"),
    )
    heatmap_axis.set_xlabel("Normalized block position", labelpad=2.5)
    heatmap_axis.tick_params(axis="x", length=2.2, width=0.6, pad=1.5)
    heatmap_axis.tick_params(axis="y", length=0, pad=2)
    heatmap_axis.spines[:].set_visible(False)
    for model_index, model_id in enumerate(MODEL_ORDER):
        heatmap_axis.text(
            -0.32,
            model_index * len(DATASET_ORDER) + len(DATASET_ORDER) / 2,
            MODEL_SHORT_LABELS[model_id],
            transform=heatmap_axis.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=MODEL_LABEL_COLORS[model_id],
            clip_on=False,
        )

    f1_axis.set_xlim(0.0, 1.0)
    f1_axis.set_ylim(0.0, row_count)
    f1_axis.invert_yaxis()
    for index, f1_change in enumerate(f1_changes):
        f1_axis.text(
            0.5,
            index + 0.5,
            f"{f1_change:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
            ha="center",
            va="center",
            fontsize=8.0,
            color=F1_COLOR,
        )
    f1_axis.set_axis_off()

    x_values = list(range(1, 9))
    for model in token_models:
        model_id = str(model["model_id"])
        style = MODEL_STYLES[model_id]
        ratios_by_offset = model["ratios_by_offset"]
        centers = [_geometric_mean(values) for values in ratios_by_offset]
        lower = [min(values) for values in ratios_by_offset]
        upper = [max(values) for values in ratios_by_offset]
        plotted_centers = [value if value > 0.0 else math.nan for value in centers]
        plotted_lower = [value if value > 0.0 else math.nan for value in lower]
        plotted_upper = [value if value > 0.0 else math.nan for value in upper]
        token_axis.fill_between(
            x_values,
            plotted_lower,
            plotted_upper,
            color=style["color"],
            alpha=0.055,
            linewidth=0,
            zorder=1,
        )
        token_axis.plot(
            x_values,
            plotted_centers,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.75,
            marker=style["marker"],
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.8,
            label=MODEL_SHORT_LABELS[model_id],
            zorder=3,
        )

    token_axis.axhline(
        1.0,
        color="#777777",
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    token_axis.set_yscale("log")
    token_axis.set_xlim(0.75, 8.18)
    token_axis.set_xticks((1, 2, 4, 6, 8))
    token_axis.set_xlabel("Token offset", labelpad=2.5)
    token_axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    token_axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2.0, 5.0), numticks=10)
    )
    token_axis.yaxis.set_minor_formatter(NullFormatter())
    token_axis.grid(
        axis="y", which="major", color="#D8D8D8", linewidth=0.5, alpha=0.75
    )
    token_axis.spines["top"].set_visible(False)
    token_axis.spines["right"].set_visible(False)
    token_axis.tick_params(which="major", length=2.4, width=0.6, pad=1.5)
    token_axis.tick_params(axis="y", which="major", labelsize=8.8)
    token_axis.tick_params(axis="y", which="minor", length=1.3, width=0.4)
    token_axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.92,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.7,
        handletextpad=0.4,
    )
    figure.text(
        0.17,
        0.985,
        "(a) Block-wide profile",
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
    )
    figure.text(
        0.5975,
        0.85,
        "ΔF1",
        ha="center",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )
    figure.text(
        0.985,
        0.985,
        "(b) Tokens 1–8",
        ha="right",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )
    figure.text(
        0.8575,
        0.91,
        "Density/int.\nline: geo. mean\nband: min–max\n(4 datasets)",
        ha="center",
        va="top",
        fontsize=8.0,
        color="#4C4C4C",
        linespacing=0.78,
    )

    colorbar_axis = figure.add_axes((0.27, 0.865, 0.235, 0.022))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((1.0, 10.0, 100.0), labels=("1×", "10×", "100×"))
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.xaxis.set_major_locator(FixedLocator((1.0, 10.0, 100.0)))
    colorbar.ax.tick_params(labelsize=8.0, length=1.5, width=0.5, pad=1.0)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.set_title("Density / interior", fontsize=8.0, pad=1.5)

    outputs = {
        extension: Path(f"{prefix}.{extension}")
        for extension in ("svg", "pdf", "png")
    }
    try:
        for extension, path in outputs.items():
            figure.savefig(
                path,
                dpi=300 if extension == "png" else None,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.02,
            )
    finally:
        plt.close(figure)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-data", type=Path, required=True)
    parser.add_argument("--token-data", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with args.profile_data.open(encoding="utf-8") as source:
        profile_data = json.load(source)
    with args.token_data.open(encoding="utf-8") as source:
        token_data = json.load(source)
    outputs = draw_combined_block_sink(
        profile_data,
        token_data,
        args.output_prefix,
    )
    for extension in ("svg", "pdf", "png"):
        print(outputs[extension])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
