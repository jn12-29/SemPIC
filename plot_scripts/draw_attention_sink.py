"""Render the compact end-of-Experiments SemPIC attention-sink heatmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

try:
    from plot_scripts.attention_sink_data import EXPECTED_DATASETS, load_plot_data
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from attention_sink_data import EXPECTED_DATASETS, load_plot_data


OUTPUT_STEM = "attention_sink_diagnostic"
SINK_TEXT_COLOR = "#6F3B00"
GAIN_TEXT_COLOR = "#5E7C83"
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}


def _output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / f"{OUTPUT_STEM}.{extension}" for extension in ("svg", "pdf", "png")]


def _extract_rows(data: dict[str, Any]) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    bin_edges: list[float] | None = None
    heatmap_rows: list[list[float]] = []
    sink_values: list[float] = []
    f1_gains: list[float] = []

    for point in data["points"]:
        dataset_id = point["dataset_id"]
        if point["status"] != "pass":
            raise ValueError(f"Incomplete evidence point: {dataset_id}")
        profile = next(row for row in point["profiles"] if row["method_key"] == "sempic")
        interior = next(
            row["mean"]
            for row in point["regions"]
            if row["method_key"] == "sempic" and row["region"] == "interior"
        )
        if not math.isfinite(interior) or interior <= 0:
            raise ValueError(f"Invalid SemPIC interior density for {dataset_id}")

        edges = [float(profile["bins"][0]["start"]), *[float(item["end"]) for item in profile["bins"]]]
        if bin_edges is None:
            bin_edges = edges
        elif edges != bin_edges:
            raise ValueError(f"Inconsistent position bins for {dataset_id}")
        normalized = [float(item["mean"]) / float(interior) for item in profile["bins"]]
        if any(not math.isfinite(value) or value <= 0 for value in normalized):
            raise ValueError(f"Invalid normalized SemPIC profile for {dataset_id}")
        heatmap_rows.append(normalized)

        sink = point["sink_ratio"]
        if point["sink_ratio_status"] != "defined" or sink is None or not math.isfinite(sink) or sink <= 0:
            raise ValueError(f"Undefined sink ratio for {dataset_id}")
        sink_values.append(float(sink))
        behavior = {row["method_key"]: float(row["f1"]) for row in point["behavior"]}
        f1_gains.append(behavior["sempic"] - behavior["vanilla_pic"])

    if bin_edges is None:
        raise ValueError("No SemPIC profiles found")
    return bin_edges, heatmap_rows, sink_values, f1_gains


def _write_status(source_path: Path, outputs: list[Path], output_path: Path) -> None:
    status_path = output_path / f"{OUTPUT_STEM}_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "generated_unverified",
                "plot_data_path": str(source_path.resolve()),
                "plot_data_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "outputs": [
                    {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in outputs
                ],
                "verification_note": "Single-column rendering and readability checks remain external.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def plot_diagnostic(
    data: dict[str, Any],
    output_dir: str | Path,
    *,
    plot_data_path: str | Path,
) -> list[Path]:
    """Write the end-of-Experiments attention-sink figure."""

    source_path = Path(plot_data_path)
    if not source_path.is_file():
        raise ValueError(f"plot_data_path is not a file: {source_path}")
    bin_edges, heatmap_rows, sink_values, f1_gains = _extract_rows(data)

    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(3.4, 2.05))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.43),
        left=0.21,
        right=0.985,
        bottom=0.235,
        top=0.79,
        wspace=0.04,
    )
    axis = figure.add_subplot(grid[0, 0])
    annotation_axis = figure.add_subplot(grid[0, 1])
    color_norm = LogNorm(vmin=min(0.5, min(map(min, heatmap_rows))), vmax=max(100.0, max(map(max, heatmap_rows))))
    mesh = axis.pcolormesh(
        bin_edges,
        list(range(len(heatmap_rows) + 1)),
        heatmap_rows,
        cmap="YlOrBr",
        norm=color_norm,
        edgecolors="white",
        linewidth=0.38,
        shading="flat",
        rasterized=False,
    )

    row_count = len(EXPECTED_DATASETS)
    axis.add_patch(Rectangle((0, 0), 1, row_count, fill=False, edgecolor="#777777", linewidth=0.65))
    for boundary in (0.1, 0.9):
        axis.axvline(boundary, color="#4E5963", linestyle=(0, (2, 1.5)), linewidth=0.85, zorder=3)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, row_count)
    axis.invert_yaxis()
    axis.set_yticks(
        [index + 0.5 for index in range(row_count)],
        [DATASET_LABELS[dataset] for dataset in EXPECTED_DATASETS],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xticks((0.0, 0.1, 0.5, 0.9, 1.0), ("0", ".1", ".5", ".9", "1"))
    axis.set_xlabel("Normalized block position", labelpad=3)
    axis.tick_params(axis="x", length=2.3, width=0.65, pad=2)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.spines[:].set_visible(False)

    annotation_axis.set_xlim(0.0, 1.0)
    annotation_axis.set_ylim(0.0, row_count)
    annotation_axis.invert_yaxis()
    annotation_axis.set_facecolor("white")
    annotation_axis.text(
        0.24,
        -0.20,
        "Sink\nPre÷Int.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        linespacing=0.85,
        clip_on=False,
        color=SINK_TEXT_COLOR,
    )
    annotation_axis.text(
        0.77,
        -0.20,
        "ΔF1",
        ha="center",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        linespacing=0.85,
        clip_on=False,
        color=GAIN_TEXT_COLOR,
    )
    for index, (sink, gain) in enumerate(zip(sink_values, f1_gains, strict=True)):
        y_value = index + 0.5
        annotation_axis.text(
            0.24,
            y_value,
            f"{sink:.1f}",
            ha="center",
            va="center",
            fontsize=7.7,
            fontstyle="italic",
            rotation=18,
            color=SINK_TEXT_COLOR,
        )
        annotation_axis.text(
            0.77,
            y_value,
            f"{gain:+.3f}",
            ha="center",
            va="center",
            fontsize=7.7,
            fontstyle="italic",
            rotation=18,
            color=GAIN_TEXT_COLOR,
        )
    annotation_axis.set_axis_off()

    colorbar_axis = figure.add_axes((0.30, 0.875, 0.27, 0.035))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((1.0, 10.0, 100.0), labels=("1×", "10×", "100×"))
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.xaxis.set_major_locator(FixedLocator((1.0, 10.0, 100.0)))
    colorbar.ax.tick_params(labelsize=7.5, length=1.8, width=0.55, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.set_title("Density / interior", fontsize=7.8, pad=2.5)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = _output_paths(output_path)
    for path in outputs:
        figure.savefig(
            path,
            dpi=300 if path.suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)
    _write_status(source_path, outputs, output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path, help="Bundle data/plot_data.json.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_diagnostic(load_plot_data(args.plot_data), args.output_dir, plot_data_path=args.plot_data)


if __name__ == "__main__":
    main()
