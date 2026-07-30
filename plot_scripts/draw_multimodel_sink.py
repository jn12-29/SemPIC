"""Render one compact attention-sink evidence figure per model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, NullLocator

try:
    from plot_scripts.multimodel_paper_data import (
        validate_plot_data as validate_unified_plot_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from multimodel_paper_data import (
        validate_plot_data as validate_unified_plot_data,
    )


SCHEMA_NAME = "sempic.paper_multimodel_evidence"
SCHEMA_VERSION = 1
MODEL_ORDER = (
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
)
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}
OUTPUT_PREFIX = "attention_sink_"
COMBINED_OUTPUT_STEM = "attention_sink_all_models"
MODEL_LABEL_COLORS = {
    "Qwen3-4B-Instruct-2507": "#2F6B9A",
    "Qwen3-8B": "#D97706",
    "Llama-3.1-8B-Instruct": "#4F7A61",
}
SINK_COLOR = "#244F73"
F1_COLOR = "#A35A12"


def _required_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _required_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _positive_count(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _validate_profile(value: Any, context: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context} must be a non-empty list")
    edges: list[tuple[float, float]] = []
    previous_end: float | None = None
    for index, raw_bin in enumerate(value):
        bin_context = f"{context}[{index}]"
        item = _required_mapping(raw_bin, bin_context)
        for field in ("start", "end", "mean", "sem", "count"):
            if field not in item:
                raise ValueError(f"{bin_context} lacks required field {field!r}")
        start = _finite_number(item["start"], f"{bin_context}.start")
        end = _finite_number(item["end"], f"{bin_context}.end")
        mean = _finite_number(item["mean"], f"{bin_context}.mean")
        sem = _finite_number(item["sem"], f"{bin_context}.sem")
        _positive_count(item["count"], f"{bin_context}.count")
        if not 0.0 <= start < end <= 1.0:
            raise ValueError(f"{bin_context} must satisfy 0 <= start < end <= 1")
        if previous_end is not None and not math.isclose(
            start, previous_end, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"{context} bins must be contiguous")
        if mean <= 0.0:
            raise ValueError(f"{bin_context}.mean must be positive")
        if sem < 0.0:
            raise ValueError(f"{bin_context}.sem must be nonnegative")
        edges.append((start, end))
        previous_end = end
    if not math.isclose(edges[0][0], 0.0, rel_tol=0.0, abs_tol=1e-12) or not math.isclose(
        edges[-1][1], 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{context} bins must cover normalized position [0, 1]")
    return tuple(edges)


def validate_plot_data(data: Any) -> dict[str, Any]:
    """Validate and return the multimodel evidence payload."""

    root = _required_mapping(validate_unified_plot_data(data), "plot data")
    if root.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"plot data schema_name must be {SCHEMA_NAME!r}")
    schema_version = root.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"plot data schema_version must be {SCHEMA_VERSION}")
    models = root.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("plot data models must be a non-empty list")

    model_ids: set[str] = set()
    safe_names: set[str] = set()
    for model_index, raw_model in enumerate(models):
        model_context = f"models[{model_index}]"
        model = _required_mapping(raw_model, model_context)
        model_id = _required_string(model.get("model_id"), f"{model_context}.model_id")
        _required_string(model.get("display_name"), f"{model_context}.display_name")
        if model_id in model_ids:
            raise ValueError(f"Duplicate model_id: {model_id}")
        model_ids.add(model_id)
        safe_name = _safe_model(model_id)
        if safe_name in safe_names:
            raise ValueError(f"Model ids collide after filename sanitization: {model_id}")
        safe_names.add(safe_name)

        points = model.get("points")
        if not isinstance(points, list) or len(points) != len(DATASET_ORDER):
            raise ValueError(
                f"{model_context}.points must contain exactly four dataset points"
            )
        datasets: set[str] = set()
        expected_edges: tuple[tuple[float, float], ...] | None = None
        for point_index, raw_point in enumerate(points):
            point_context = f"{model_context}.points[{point_index}]"
            point = _required_mapping(raw_point, point_context)
            dataset_id = _required_string(
                point.get("dataset_id"), f"{point_context}.dataset_id"
            )
            if dataset_id not in DATASET_ORDER:
                raise ValueError(f"Unsupported dataset_id: {dataset_id}")
            if dataset_id in datasets:
                raise ValueError(f"Duplicate dataset point for {model_id}/{dataset_id}")
            datasets.add(dataset_id)
            if "sink_profile" not in point:
                raise ValueError(f"{point_context} lacks required field 'sink_profile'")
            edges = _validate_profile(point["sink_profile"], f"{point_context}.sink_profile")
            if expected_edges is None:
                expected_edges = edges
            elif edges != expected_edges:
                raise ValueError(f"Inconsistent position bins for model {model_id}")
            sink_ratio = _finite_number(
                point.get("sink_ratio"), f"{point_context}.sink_ratio"
            )
            if sink_ratio <= 0.0:
                raise ValueError(f"{point_context}.sink_ratio must be positive")
            _finite_number(point.get("f1_change"), f"{point_context}.f1_change")
        if datasets != set(DATASET_ORDER):
            raise ValueError(f"{model_context}.points must contain all four datasets")
    return dict(root)


def load_plot_data(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return validate_plot_data(json.load(handle))


def _safe_model(model_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._")
    if not segment:
        raise ValueError(f"Unsafe empty model filename derived from {model_id!r}")
    return segment


def _interior_density(profile: Sequence[Mapping[str, Any]]) -> float:
    weighted_sum = 0.0
    covered_width = 0.0
    for item in profile:
        start = float(item["start"])
        end = float(item["end"])
        overlap = max(0.0, min(end, 0.9) - max(start, 0.1))
        weighted_sum += overlap * float(item["mean"])
        covered_width += overlap
    if covered_width <= 0.0:
        raise ValueError("sink_profile does not cover the interior region [0.1, 0.9)")
    interior = weighted_sum / covered_width
    if not math.isfinite(interior) or interior <= 0.0:
        raise ValueError("sink_profile has invalid interior density")
    return interior


def _model_rows(
    model: Mapping[str, Any],
) -> tuple[list[float], list[list[float]], list[float], list[float]]:
    point_by_dataset = {point["dataset_id"]: point for point in model["points"]}
    heatmap_rows: list[list[float]] = []
    sink_ratios: list[float] = []
    f1_changes: list[float] = []
    bin_edges: list[float] | None = None
    for dataset_id in DATASET_ORDER:
        point = point_by_dataset[dataset_id]
        profile = point["sink_profile"]
        interior = _interior_density(profile)
        heatmap_rows.append([float(item["mean"]) / interior for item in profile])
        sink_ratios.append(float(point["sink_ratio"]))
        f1_changes.append(float(point["f1_change"]))
        if bin_edges is None:
            bin_edges = [float(profile[0]["start"]), *[float(item["end"]) for item in profile]]
    if bin_edges is None:
        raise ValueError(f"No sink profiles found for model {model['model_id']}")
    return bin_edges, heatmap_rows, sink_ratios, f1_changes


def _selected_models(
    data: Mapping[str, Any], model_ids: Sequence[str] | None
) -> list[Mapping[str, Any]]:
    models = list(data["models"])
    if not model_ids:
        return models
    by_id = {model["model_id"]: model for model in models}
    selected: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id not in by_id:
            raise ValueError(f"Unknown --model-id: {model_id}")
        if model_id not in seen:
            selected.append(by_id[model_id])
            seen.add(model_id)
    return selected


def _draw_model(model: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    bin_edges, heatmap_rows, sink_ratios, f1_changes = _model_rows(model)
    figure = plt.figure(figsize=(3.4, 2.16))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.44),
        left=0.21,
        right=0.985,
        bottom=0.235,
        top=0.78,
        wspace=0.035,
    )
    axis = figure.add_subplot(grid[0, 0])
    annotation_axis = figure.add_subplot(grid[0, 1])
    values = [value for row in heatmap_rows for value in row]
    color_norm = LogNorm(vmin=min(0.5, min(values)), vmax=max(100.0, max(values)))
    mesh = axis.pcolormesh(
        bin_edges,
        list(range(len(DATASET_ORDER) + 1)),
        heatmap_rows,
        cmap="Blues",
        norm=color_norm,
        edgecolors="white",
        linewidth=0.38,
        shading="flat",
        rasterized=False,
    )
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            len(DATASET_ORDER),
            fill=False,
            edgecolor="#666666",
            linewidth=0.7,
        )
    )
    for boundary in (0.1, 0.9):
        axis.axvline(
            boundary,
            color="#333333",
            linestyle=(0, (2, 1.5)),
            linewidth=0.9,
            zorder=3,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, len(DATASET_ORDER))
    axis.invert_yaxis()
    axis.set_yticks(
        [index + 0.5 for index in range(len(DATASET_ORDER))],
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xticks((0.0, 0.1, 0.5, 0.9, 1.0), ("0", ".1", ".5", ".9", "1"))
    axis.set_xlabel("Normalized block position", labelpad=3)
    axis.tick_params(axis="x", length=2.3, width=0.65, pad=2)
    axis.tick_params(axis="y", length=0, pad=4)
    axis.spines[:].set_visible(False)
    axis.text(
        -0.02,
        1.40,
        str(model["display_name"]),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        clip_on=False,
    )

    annotation_axis.set_xlim(0.0, 1.0)
    annotation_axis.set_ylim(0.0, len(DATASET_ORDER))
    annotation_axis.invert_yaxis()
    annotation_axis.set_facecolor("white")
    annotation_axis.text(
        0.25,
        -0.20,
        "Sink\nPre/Int",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        linespacing=0.86,
        clip_on=False,
        color=SINK_COLOR,
    )
    annotation_axis.text(
        0.78,
        -0.20,
        "ΔF1",
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        clip_on=False,
        color=F1_COLOR,
    )
    for index, (sink_ratio, f1_change) in enumerate(
        zip(sink_ratios, f1_changes, strict=True)
    ):
        y_value = index + 0.5
        annotation_axis.text(
            0.25,
            y_value,
            f"{sink_ratio:.1f}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            rotation=18,
            color=SINK_COLOR,
        )
        annotation_axis.text(
            0.78,
            y_value,
            f"{f1_change:+.3f}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            rotation=18,
            color=F1_COLOR,
        )
    annotation_axis.set_axis_off()

    colorbar_axis = figure.add_axes((0.30, 0.865, 0.27, 0.035))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((1.0, 10.0, 100.0), labels=("1×", "10×", "100×"))
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.xaxis.set_major_locator(FixedLocator((1.0, 10.0, 100.0)))
    colorbar.ax.tick_params(labelsize=8.0, length=1.8, width=0.55, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.set_title("Density / interior", fontsize=8.0, pad=2.5)

    safe_model = _safe_model(str(model["model_id"]))
    outputs = {
        extension: output_dir / f"{OUTPUT_PREFIX}{safe_model}.{extension}"
        for extension in ("svg", "pdf", "png")
    }
    for extension, path in outputs.items():
        figure.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)
    return outputs


def _combined_rows(
    data: Mapping[str, Any],
) -> tuple[
    list[float],
    list[list[float]],
    list[float],
    list[float],
    list[str],
]:
    models = list(data["models"])
    model_order = tuple(str(model["model_id"]) for model in models)
    if model_order != MODEL_ORDER:
        raise ValueError(
            "Combined figure requires the fixed three-model order: "
            + ", ".join(MODEL_ORDER)
        )
    expected_edges: list[float] | None = None
    heatmap_rows: list[list[float]] = []
    sink_ratios: list[float] = []
    f1_changes: list[float] = []
    display_names: list[str] = []
    for model in models:
        bin_edges, model_heatmap, model_sink, model_f1 = _model_rows(model)
        if expected_edges is None:
            expected_edges = bin_edges
        elif bin_edges != expected_edges:
            raise ValueError("Combined figure requires consistent position bins")
        heatmap_rows.extend(model_heatmap)
        sink_ratios.extend(model_sink)
        f1_changes.extend(model_f1)
        display_names.append(str(model["display_name"]))
    if expected_edges is None:
        raise ValueError("Combined figure has no model profiles")
    return expected_edges, heatmap_rows, sink_ratios, f1_changes, display_names


def draw_combined_sink(
    data: Any,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one shared-scale, twelve-row figure for the fixed three models."""

    validated = validate_plot_data(data)
    bin_edges, heatmap_rows, sink_ratios, f1_changes, display_names = _combined_rows(
        validated
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    row_count = len(MODEL_ORDER) * len(DATASET_ORDER)
    figure = plt.figure(figsize=(3.30, 3.38))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.46),
        left=0.31,
        right=0.985,
        bottom=0.13,
        top=0.82,
        wspace=0.035,
    )
    axis = figure.add_subplot(grid[0, 0])
    annotation_axis = figure.add_subplot(grid[0, 1])
    values = [value for row in heatmap_rows for value in row]
    color_norm = LogNorm(vmin=min(0.5, min(values)), vmax=max(100.0, max(values)))
    mesh = axis.pcolormesh(
        bin_edges,
        list(range(row_count + 1)),
        heatmap_rows,
        cmap="Blues",
        norm=color_norm,
        edgecolors="white",
        linewidth=0.32,
        shading="flat",
        rasterized=False,
    )
    axis.add_patch(
        Rectangle(
            (0, 0),
            1,
            row_count,
            fill=False,
            edgecolor="#666666",
            linewidth=0.7,
        )
    )
    for boundary in (0.1, 0.9):
        axis.axvline(
            boundary,
            color="#333333",
            linestyle=(0, (2, 1.5)),
            linewidth=0.9,
            zorder=3,
        )
    for boundary in (len(DATASET_ORDER), 2 * len(DATASET_ORDER)):
        separator = axis.axhline(
            boundary, color="#4C4C4C", linewidth=0.9, zorder=4
        )
        separator.set_gid(f"model-separator-{boundary}")
        annotation_separator = annotation_axis.axhline(
            boundary, color="#A0A0A0", linewidth=0.7, zorder=1
        )
        annotation_separator.set_gid(f"annotation-model-separator-{boundary}")

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, row_count)
    axis.invert_yaxis()
    row_datasets = list(DATASET_ORDER) * len(MODEL_ORDER)
    axis.set_yticks(
        [index + 0.5 for index in range(row_count)],
        [DATASET_LABELS[dataset] for dataset in row_datasets],
    )
    axis.set_xticks((0.0, 0.1, 0.5, 0.9, 1.0), ("0", ".1", ".5", ".9", "1"))
    axis.set_xlabel("Normalized block position", labelpad=3)
    axis.tick_params(axis="x", length=2.3, width=0.65, pad=2)
    axis.tick_params(axis="y", length=0, pad=3)
    axis.spines[:].set_visible(False)
    for model_index, display_name in enumerate(display_names):
        axis.text(
            -0.49,
            model_index * len(DATASET_ORDER) + len(DATASET_ORDER) / 2,
            display_name,
            transform=axis.get_yaxis_transform(),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            rotation=90,
            color=MODEL_LABEL_COLORS[str(MODEL_ORDER[model_index])],
            clip_on=False,
        )

    annotation_axis.set_xlim(0.0, 1.0)
    annotation_axis.set_ylim(0.0, row_count)
    annotation_axis.invert_yaxis()
    annotation_axis.set_facecolor("white")
    annotation_axis.text(
        0.25,
        1.025,
        "Sink\nPre/Int",
        transform=annotation_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        linespacing=0.86,
        clip_on=False,
        color=SINK_COLOR,
    )
    annotation_axis.text(
        0.79,
        1.025,
        "ΔF1",
        transform=annotation_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        fontstyle="italic",
        rotation=18,
        clip_on=False,
        color=F1_COLOR,
    )
    for index, (sink_ratio, f1_change) in enumerate(
        zip(sink_ratios, f1_changes, strict=True)
    ):
        y_value = index + 0.5
        annotation_axis.text(
            0.25,
            y_value,
            f"{sink_ratio:.1f}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            rotation=18,
            color=SINK_COLOR,
        )
        annotation_axis.text(
            0.79,
            y_value,
            f"{f1_change:+.3f}",
            ha="center",
            va="center",
            fontsize=8.0,
            fontstyle="italic",
            rotation=18,
            color=F1_COLOR,
        )
    annotation_axis.set_axis_off()

    colorbar_axis = figure.add_axes((0.40, 0.91, 0.26, 0.018))
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks((1.0, 10.0, 100.0), labels=("1×", "10×", "100×"))
    colorbar.ax.xaxis.set_minor_locator(NullLocator())
    colorbar.ax.xaxis.set_major_locator(FixedLocator((1.0, 10.0, 100.0)))
    colorbar.ax.tick_params(labelsize=8.0, length=1.8, width=0.55, pad=1.5)
    colorbar.outline.set_linewidth(0.55)
    colorbar.ax.set_title("Density / interior", fontsize=8.0, pad=2.5)

    outputs = {
        extension: destination / f"{COMBINED_OUTPUT_STEM}.{extension}"
        for extension in ("svg", "pdf", "png")
    }
    try:
        for extension, path in outputs.items():
            figure.savefig(
                path,
                dpi=300 if extension == "png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
    finally:
        plt.close(figure)
    return outputs


def draw_multimodel_sink(
    data: Any,
    output_dir: str | Path,
    *,
    model_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Path]]:
    """Validate data and write SVG, PDF, and PNG for each selected model."""

    validated = validate_plot_data(data)
    selected = _selected_models(validated, model_ids)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.2,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return {
        str(model["model_id"]): _draw_model(model, destination) for model in selected
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Model id to render; repeat to select multiple models. Omit for all.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Render the fixed three models together in one twelve-row figure.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_plot_data(args.plot_data)
    if args.combined:
        if args.model_id:
            raise ValueError("--combined cannot be used with --model-id")
        combined_outputs = draw_combined_sink(data, args.output_dir)
        for extension, path in combined_outputs.items():
            print(f"all models {extension}: {path}")
        return 0
    outputs = draw_multimodel_sink(
        data,
        args.output_dir,
        model_ids=args.model_id,
    )
    for model_id, formats in outputs.items():
        for extension, path in formats.items():
            print(f"{model_id} {extension}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
