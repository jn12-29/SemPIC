"""Render the compact single-column SemPIC validation figure."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from plot_scripts.attention_sink_data import EXPECTED_DATASETS, load_plot_data
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from attention_sink_data import EXPECTED_DATASETS, load_plot_data


OUTPUT_STEM = "sempic_interior_validation"
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}
BAR_COLOR = "#F7DFC3"
BAR_EDGE_COLOR = "#B77935"
KV_COLOR = "#747A80"
SEMPIC_COLOR = "#D97706"


def _output_paths(output_dir: Path) -> list[Path]:
    return [output_dir / f"{OUTPUT_STEM}.{extension}" for extension in ("svg", "pdf", "png")]


def _validation_rows(data: dict[str, Any]) -> list[tuple[float, float, float]]:
    rows = []
    for point in data["points"]:
        if point["status"] != "pass":
            raise ValueError(f"Incomplete evidence point: {point['dataset_id']}")
        relative = {row["method_key"]: row for row in point["relative_interior_attention_errors"]}
        values = []
        for method in ("kvpacket", "sempic"):
            row = relative[method]
            value = row["value"]
            if row["status"] != "defined" or value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"Undefined {method} R_int for {point['dataset_id']}")
            values.append(float(value))
        recovery = point["recovery_fraction"]
        if (
            point["recovery_fraction_status"] != "defined"
            or recovery is None
            or not math.isfinite(recovery)
            or recovery < 0
        ):
            raise ValueError(f"Undefined SemPIC recovery for {point['dataset_id']}")
        rows.append((values[0], values[1], float(recovery)))
    return rows


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


def plot_validation(
    data: dict[str, Any],
    output_dir: str | Path,
    *,
    plot_data_path: str | Path,
) -> list[Path]:
    """Write SVG, PDF, and PNG from validated Sink schema-v2 plot data."""

    source_path = Path(plot_data_path)
    if not source_path.is_file():
        raise ValueError(f"plot_data_path is not a file: {source_path}")
    rows = _validation_rows(data)

    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "legend.fontsize": 7.6,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.4, 2.18))
    x_values = list(range(len(EXPECTED_DATASETS)))
    all_values = [1.0, *(value for row in rows for value in row)]
    y_min = max(0.0, min(all_values) - 0.10)
    y_max = max(all_values) + 0.12

    for index, (kv_value, sempic_value, recovery) in enumerate(rows):
        axis.bar(
            index,
            recovery - y_min,
            bottom=y_min,
            width=0.58,
            color=BAR_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.45,
            hatch="///",
            zorder=0,
        )
        axis.annotate(
            "",
            xy=(index, sempic_value),
            xytext=(index, kv_value),
            arrowprops={
                "arrowstyle": "-|>",
                "color": SEMPIC_COLOR,
                "linewidth": 1.45,
                "mutation_scale": 9,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=3,
        )
        axis.scatter(
            index,
            kv_value,
            s=31,
            facecolor="white",
            edgecolor=KV_COLOR,
            linewidth=1.15,
            zorder=4,
        )
        axis.scatter(
            index,
            sempic_value,
            s=34,
            color=SEMPIC_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )

    axis.axhline(1.0, color="#5D636A", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    axis.set_xlim(-0.48, len(EXPECTED_DATASETS) - 0.52)
    axis.set_ylim(y_min, y_max)
    axis.set_ylabel("Normalized ratio")
    axis.set_xticks(
        x_values,
        [DATASET_LABELS[dataset] for dataset in EXPECTED_DATASETS],
        rotation=19,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.65)
    break_style = {
        "transform": axis.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 0.75,
    }
    axis.plot((-0.012, 0.012), (-0.012, 0.012), **break_style)
    axis.plot((-0.012, 0.012), (0.010, 0.034), **break_style)

    handles = [
        Patch(
            facecolor=BAR_COLOR,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=0.45,
            hatch="///",
            label="F1 recovery",
        ),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=KV_COLOR, label="Packet Rint"),
        Line2D([0, 1], [0, 0], color=SEMPIC_COLOR, marker=">", markevery=[1], label="SemPIC Rint"),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.87)

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
    parser.add_argument("plot_data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_validation(load_plot_data(args.plot_data), args.output_dir, plot_data_path=args.plot_data)


if __name__ == "__main__":
    main()
