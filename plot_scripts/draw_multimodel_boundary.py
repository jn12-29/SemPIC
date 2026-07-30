"""Render one compact Boundary Motivation figure per model."""

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
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}
VALUE_FIELDS = ("kv_recovery", "kv_pre_ratio", "kv_interior_ratio")
COMBINED_MODEL_ORDER = (
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
)
COMBINED_OUTPUT_STEM = "boundary_motivation_all_models"
COMBINED_MODEL_STYLES = {
    "Qwen3-4B-Instruct-2507": {
        "marker": "o",
        "hatch": "///",
        "bar_color": "#E6E6E6",
        "line_color": "#303030",
    },
    "Qwen3-8B": {
        "marker": "s",
        "hatch": "---",
        "bar_color": "#D5D5D5",
        "line_color": "#555555",
    },
    "Llama-3.1-8B-Instruct": {
        "marker": "^",
        "hatch": "xxx",
        "bar_color": "#C4C4C4",
        "line_color": "#777777",
    },
}


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return number


def _safe_model(model_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._")
    if not segment:
        raise ValueError(f"Unsafe empty model segment derived from {model_id!r}")
    return segment


def validate_plot_data(value: Any) -> dict[str, Any]:
    """Validate and normalize the schema-v1 multimodel evidence contract."""

    value = validate_unified_plot_data(value)
    if not isinstance(value, Mapping):
        raise ValueError("Plot data must be a JSON object")
    if value.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"schema_name must be {SCHEMA_NAME!r}")
    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    raw_models = value.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("models must be a non-empty list")

    models: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    for model_index, raw_model in enumerate(raw_models):
        if not isinstance(raw_model, Mapping):
            raise ValueError(f"models[{model_index}] must be an object")
        model_id = _nonempty_string(raw_model.get("model_id"), "model_id")
        display_name = _nonempty_string(raw_model.get("display_name"), "display_name")
        if model_id in seen_model_ids:
            raise ValueError(f"Duplicate model_id: {model_id}")
        seen_model_ids.add(model_id)

        raw_points = raw_model.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != len(DATASET_ORDER):
            raise ValueError(f"{model_id} must contain exactly four dataset points")
        points_by_dataset: dict[str, dict[str, Any]] = {}
        for point_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, Mapping):
                raise ValueError(f"{model_id}.points[{point_index}] must be an object")
            dataset_id = _nonempty_string(raw_point.get("dataset_id"), "dataset_id")
            if dataset_id not in DATASET_ORDER:
                raise ValueError(f"Unsupported dataset_id for {model_id}: {dataset_id}")
            if dataset_id in points_by_dataset:
                raise ValueError(f"Duplicate dataset point for {model_id}: {dataset_id}")
            point = {"dataset_id": dataset_id}
            for field in VALUE_FIELDS:
                point[field] = _finite_nonnegative(
                    raw_point.get(field), f"{model_id}/{dataset_id}/{field}"
                )
            points_by_dataset[dataset_id] = point

        missing = set(DATASET_ORDER) - set(points_by_dataset)
        if missing:
            raise ValueError(f"Missing dataset points for {model_id}: {sorted(missing)}")
        models.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "points": [points_by_dataset[dataset] for dataset in DATASET_ORDER],
            }
        )

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "models": models,
    }


def load_plot_data(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return validate_plot_data(json.load(handle))


def _select_models(
    data: Mapping[str, Any], model_ids: Sequence[str] | None
) -> list[Mapping[str, Any]]:
    models = list(data["models"])
    if not model_ids:
        return models
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("--model-id values must be unique")
    models_by_id = {model["model_id"]: model for model in models}
    unknown = [model_id for model_id in model_ids if model_id not in models_by_id]
    if unknown:
        raise ValueError(f"Unknown model_id values: {unknown}")
    return [models_by_id[model_id] for model_id in model_ids]


def _draw_model(model: Mapping[str, Any], output_dir: Path) -> list[Path]:
    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.4, 2.18))
    style = COMBINED_MODEL_STYLES[str(model["model_id"])]
    points = model["points"]
    x_values = list(range(len(DATASET_ORDER)))
    all_values = [
        1.0,
        *(float(point[field]) for point in points for field in VALUE_FIELDS),
    ]
    y_max = max(all_values) * 1.12

    for index, point in enumerate(points):
        recovery = float(point["kv_recovery"])
        pre_ratio = float(point["kv_pre_ratio"])
        interior_ratio = float(point["kv_interior_ratio"])
        bars = axis.bar(
            index,
            recovery,
            width=0.58,
            color=style["bar_color"],
            edgecolor=style["line_color"],
            linewidth=0.5,
            hatch="///",
            zorder=0,
        )
        bars.patches[0].set_gid(f"recovery-bar-{point['dataset_id']}")
        annotation = axis.annotate(
            "",
            xy=(index, interior_ratio),
            xytext=(index, pre_ratio),
            arrowprops={
                "arrowstyle": "-|>",
                "color": style["line_color"],
                "linewidth": 1.4,
                "mutation_scale": 9,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=3,
        )
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_gid(f"pre-interior-arrow-{point['dataset_id']}")
        pre_marker = axis.scatter(
            index,
            pre_ratio,
            s=31,
            color=style["line_color"],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        pre_marker.set_gid(f"pre-marker-{point['dataset_id']}")
        interior_marker = axis.scatter(
            index,
            interior_ratio,
            s=34,
            facecolor="white",
            edgecolor=style["line_color"],
            linewidth=1.15,
            zorder=4,
        )
        interior_marker.set_gid(f"interior-marker-{point['dataset_id']}")

    reference = axis.axhline(
        1.0, color="#5D636A", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1
    )
    reference.set_gid("unit-reference-line")
    axis.set_xlim(-0.48, len(DATASET_ORDER) - 0.52)
    axis.set_ylim(0.0, y_max)
    axis.set_ylabel("Normalized ratio")
    axis.set_xticks(
        x_values,
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=19,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.65)
    axis.text(
        0.0,
        1.015,
        model["display_name"],
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
    )

    handles = [
        Patch(
            facecolor=style["bar_color"],
            edgecolor=style["line_color"],
            linewidth=0.5,
            hatch="///",
            label="F1 recovery",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=style["line_color"],
            markeredgecolor="white",
            label="KV pre",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=style["line_color"],
            label="KV interior",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.035),
        ncol=3,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.3,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.87)

    stem = f"boundary_motivation_{_safe_model(model['model_id'])}"
    outputs = [output_dir / f"{stem}.{extension}" for extension in ("svg", "pdf", "png")]
    try:
        for path in outputs:
            figure.savefig(
                path,
                dpi=300 if path.suffix == ".png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
    finally:
        plt.close(figure)
    return outputs


def _combined_models(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = list(data["models"])
    model_ids = tuple(model["model_id"] for model in models)
    if model_ids != COMBINED_MODEL_ORDER:
        raise ValueError(
            "Combined mode requires exact model_id order "
            f"{list(COMBINED_MODEL_ORDER)}, found {list(model_ids)}"
        )
    return models


def _draw_combined(data: Mapping[str, Any], output_dir: Path) -> list[Path]:
    matplotlib.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    models = _combined_models(data)
    figure, axis = plt.subplots(figsize=(3.4, 2.38))
    model_offsets = (-0.23, 0.0, 0.23)
    bar_width = 0.18
    all_values = [
        1.0,
        *(
            float(point[field])
            for model in models
            for point in model["points"]
            for field in VALUE_FIELDS
        ),
    ]

    for model_index, model in enumerate(models):
        model_id = str(model["model_id"])
        style = COMBINED_MODEL_STYLES[model_id]
        safe_model = _safe_model(model_id)
        for dataset_index, point in enumerate(model["points"]):
            x_value = dataset_index + model_offsets[model_index]
            identity = f"{safe_model}-{point['dataset_id']}"
            bars = axis.bar(
                x_value,
                float(point["kv_recovery"]),
                width=bar_width,
                color=style["bar_color"],
                edgecolor=style["line_color"],
                linewidth=0.5,
                hatch=style["hatch"],
                zorder=0,
            )
            bars.patches[0].set_gid(f"combined-recovery-bar-{identity}")
            annotation = axis.annotate(
                "",
                xy=(x_value, float(point["kv_interior_ratio"])),
                xytext=(x_value, float(point["kv_pre_ratio"])),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": style["line_color"],
                    "linewidth": 1.15,
                    "mutation_scale": 7.5,
                    "shrinkA": 2.5,
                    "shrinkB": 2.5,
                },
                zorder=3,
            )
            if annotation.arrow_patch is not None:
                annotation.arrow_patch.set_gid(f"combined-pre-interior-arrow-{identity}")
            pre_marker = axis.scatter(
                x_value,
                float(point["kv_pre_ratio"]),
                s=23,
                marker=style["marker"],
                color=style["line_color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            pre_marker.set_gid(f"combined-pre-marker-{identity}")
            interior_marker = axis.scatter(
                x_value,
                float(point["kv_interior_ratio"]),
                s=26,
                marker=style["marker"],
                facecolor="white",
                edgecolor=style["line_color"],
                linewidth=1.05,
                zorder=4,
            )
            interior_marker.set_gid(f"combined-interior-marker-{identity}")

    reference = axis.axhline(
        1.0, color="#4D4D4D", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1
    )
    reference.set_gid("combined-unit-reference-line")
    axis.set_xlim(-0.48, len(DATASET_ORDER) - 0.52)
    axis.set_ylim(0.0, max(all_values) * 1.12)
    axis.set_ylabel("Normalized ratio")
    axis.set_xticks(
        range(len(DATASET_ORDER)),
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=19,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.65)

    model_handles = []
    model_labels = []
    for model in models:
        style = COMBINED_MODEL_STYLES[model["model_id"]]
        model_handles.append(
            (
                Patch(
                    facecolor=style["bar_color"],
                    edgecolor=style["line_color"],
                    linewidth=0.5,
                    hatch=style["hatch"],
                ),
                Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    linestyle="",
                    markerfacecolor=style["line_color"],
                    markeredgecolor="white",
                ),
            )
        )
        model_labels.append(str(model["display_name"]))
    semantic_handles = [
        Patch(
            facecolor="#E5E5E5",
            edgecolor="#555555",
            linewidth=0.5,
            hatch="///",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="#444444",
            markeredgecolor="white",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#444444",
        ),
    ]
    legend_handles = [
        model_handles[0],
        semantic_handles[0],
        model_handles[1],
        semantic_handles[1],
        model_handles[2],
        semantic_handles[2],
    ]
    legend_labels = [
        model_labels[0],
        "KV F1 recovery",
        model_labels[1],
        "Pre",
        model_labels[2],
        "Interior",
    ]
    axis.legend(
        handles=legend_handles,
        labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.24, top=0.75)

    outputs = [
        output_dir / f"{COMBINED_OUTPUT_STEM}.{extension}"
        for extension in ("svg", "pdf", "png")
    ]
    try:
        for path in outputs:
            figure.savefig(
                path,
                dpi=300 if path.suffix == ".png" else None,
                bbox_inches="tight",
                pad_inches=0.02,
            )
    finally:
        plt.close(figure)
    return outputs


def _plot_validated_models(
    validated: Mapping[str, Any],
    output_dir: str | Path,
    *,
    model_ids: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    selected = _select_models(validated, model_ids)
    safe_segments = [_safe_model(model["model_id"]) for model in selected]
    if len(set(safe_segments)) != len(safe_segments):
        raise ValueError("Selected model_id values produce colliding output stems")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return {
        model["model_id"]: _draw_model(model, destination) for model in selected
    }


def plot_models(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    model_ids: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Write SVG, PDF, and PNG for each selected model."""

    return _plot_validated_models(
        validate_plot_data(data), output_dir, model_ids=model_ids
    )


def _plot_validated_combined(
    validated: Mapping[str, Any], output_dir: str | Path
) -> list[Path]:
    _combined_models(validated)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return _draw_combined(validated, destination)


def plot_combined(data: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    """Write one shared-axis SVG, PDF, and PNG for the fixed three-model contract."""

    return _plot_validated_combined(validate_plot_data(data), output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Render only this model_id; repeat to select multiple models.",
    )
    mode.add_argument(
        "--combined",
        action="store_true",
        help="Render the fixed three models together on one shared axis.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = load_plot_data(args.plot_data)
    if args.combined:
        for path in _plot_validated_combined(data, args.output_dir):
            print(f"combined: {path}")
        return 0
    outputs = _plot_validated_models(
        data, args.output_dir, model_ids=args.model_id
    )
    for model_id, paths in outputs.items():
        for path in paths:
            print(f"{model_id}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
