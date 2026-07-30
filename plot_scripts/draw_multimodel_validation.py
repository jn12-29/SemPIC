"""Render individual or combined truncated-axis SemPIC validation figures."""

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
OUTPUT_PREFIX = "sempic_interior_validation"
COMBINED_OUTPUT_STEM = f"{OUTPUT_PREFIX}_all_models"
COMBINED_MODEL_ORDER = (
    "Qwen3-4B-Instruct-2507",
    "Qwen3-8B",
    "Llama-3.1-8B-Instruct",
)
RECOVERY_COLOR = "#D9D9D9"
KV_COLOR = "#4D4D4D"
SEMPIC_COLOR = "#4D4D4D"
MODEL_STYLES = {
    "Qwen3-4B-Instruct-2507": {
        "marker": "o",
        "hatch": "///",
        "color": "#2F6B9A",
        "bar_color": "#D5E1EB",
    },
    "Qwen3-8B": {
        "marker": "s",
        "hatch": "\\\\\\",
        "color": "#D97706",
        "bar_color": "#F7E4CD",
    },
    "Llama-3.1-8B-Instruct": {
        "marker": "D",
        "hatch": "xx",
        "color": "#4F7A61",
        "bar_color": "#DCE4DF",
    },
}


def _safe_model_segment(model_id: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._")
    if not segment:
        raise ValueError(f"Unsafe empty output segment derived from model_id {model_id!r}")
    return segment


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_metric(point: Mapping[str, Any], field: str, identity: str) -> float:
    value = point.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{identity}.{field} must be a JSON number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{identity}.{field} must be finite and nonnegative")
    return number


def validate_plot_data(value: Any) -> dict[str, Any]:
    """Validate and return the schema-v1 multi-model document."""

    validate_unified_plot_data(value)
    if not isinstance(value, dict):
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

    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    model_ids: set[str] = set()
    output_segments: dict[str, str] = {}
    for model_index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ValueError(f"models[{model_index}] must be an object")
        model_id = _required_string(
            model.get("model_id"), f"models[{model_index}].model_id"
        )
        _required_string(model.get("display_name"), f"models[{model_index}].display_name")
        if model_id in model_ids:
            raise ValueError(f"Duplicate model_id: {model_id}")
        model_ids.add(model_id)

        segment = _safe_model_segment(model_id)
        prior = output_segments.get(segment)
        if prior is not None:
            raise ValueError(
                f"model_id values {prior!r} and {model_id!r} share output segment {segment!r}"
            )
        output_segments[segment] = model_id

        points = model.get("points")
        if not isinstance(points, list) or len(points) != len(DATASET_ORDER):
            raise ValueError(
                f"{model_id}.points must contain exactly {len(DATASET_ORDER)} entries"
            )
        for point_index, (point, expected_dataset) in enumerate(
            zip(points, DATASET_ORDER, strict=True)
        ):
            if not isinstance(point, dict):
                raise ValueError(f"{model_id}.points[{point_index}] must be an object")
            dataset_id = point.get("dataset_id", expected_dataset)
            if dataset_id != expected_dataset:
                raise ValueError(
                    f"{model_id}.points must follow dataset order {list(DATASET_ORDER)}; "
                    f"index {point_index} is {dataset_id!r}"
                )
            identity = f"{model_id}/{expected_dataset}"
            for field in ("sempic_recovery", "kv_rint", "sempic_rint"):
                _required_metric(point, field, identity)
    return value


def load_plot_data(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        return validate_plot_data(json.load(handle))


def _axis_limits_for_models(models: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    values = [1.0]
    for model in models:
        for point in model["points"]:
            values.extend(
                min(float(point[field]), 1.0)
                if field == "sempic_recovery"
                else float(point[field])
                for field in ("sempic_recovery", "kv_rint", "sempic_rint")
            )
    low = min(values)
    high = max(values)
    span = max(high - low, 0.1)
    margin = max(0.05, span * 0.12)
    y_min = math.floor((low - margin) / 0.05) * 0.05
    y_max = math.ceil((high + margin) / 0.05) * 0.05
    if math.isclose(y_min, 0.0, abs_tol=1e-12):
        y_min = -0.05
    if y_max <= high:
        y_max = high + 0.05
    return y_min, y_max


def _axis_limits(model: Mapping[str, Any]) -> tuple[float, float]:
    return _axis_limits_for_models([model])


def _draw_axis_break(axis: Any) -> None:
    style = {
        "transform": axis.transAxes,
        "color": "black",
        "clip_on": False,
        "linewidth": 0.9,
        "solid_capstyle": "butt",
        "zorder": 8,
    }
    first = axis.plot((-0.014, 0.014), (-0.012, 0.014), **style)[0]
    second = axis.plot((-0.014, 0.014), (0.018, 0.044), **style)[0]
    first.set_gid("truncated-axis-break-1")
    second.set_gid("truncated-axis-break-2")


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.8,
            "axes.titlesize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.75,
            "hatch.linewidth": 0.55,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _save_outputs(
    figure: Any,
    output_dir: Path,
    stem: str,
) -> dict[str, Path]:
    outputs = {
        extension: output_dir / f"{stem}.{extension}"
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


def _draw_model(model: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    _configure_style()
    figure, axis = plt.subplots(figsize=(3.4, 2.25))
    y_min, y_max = _axis_limits(model)
    style = MODEL_STYLES[str(model["model_id"])]

    for index, (dataset_id, point) in enumerate(
        zip(DATASET_ORDER, model["points"], strict=True)
    ):
        recovery = min(float(point["sempic_recovery"]), 1.0)
        kv_rint = float(point["kv_rint"])
        sempic_rint = float(point["sempic_rint"])

        bars = axis.bar(
            index,
            recovery - y_min,
            bottom=y_min,
            width=0.56,
            color=str(style["bar_color"]),
            edgecolor=str(style["color"]),
            linewidth=0.55,
            hatch=str(style["hatch"]),
            zorder=0,
        )
        bars.patches[0].set_gid(f"recovery-bar-{dataset_id}")

        annotation = axis.annotate(
            "",
            xy=(index, sempic_rint),
            xytext=(index, kv_rint),
            arrowprops={
                "arrowstyle": "-|>",
                "color": str(style["color"]),
                "linewidth": 1.45,
                "mutation_scale": 9,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=3,
        )
        if annotation.arrow_patch is not None:
            annotation.arrow_patch.set_gid(f"rint-arrow-{dataset_id}")
        kv_marker = axis.scatter(
            index,
            kv_rint,
            s=31,
            marker=str(style["marker"]),
            facecolor="white",
            edgecolor=str(style["color"]),
            linewidth=1.2,
            zorder=4,
        )
        kv_marker.set_gid(f"kv-rint-{dataset_id}")
        sempic_marker = axis.scatter(
            index,
            sempic_rint,
            s=34,
            marker=str(style["marker"]),
            color=str(style["color"]),
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        sempic_marker.set_gid(f"sempic-rint-{dataset_id}")

    reference = axis.axhline(
        1.0,
        color="#555A60",
        linestyle=(0, (3, 2)),
        linewidth=0.95,
        zorder=1,
    )
    reference.set_gid("unit-reference-line")
    axis.set_xlim(-0.48, len(DATASET_ORDER) - 0.52)
    axis.set_ylim(y_min, y_max)
    axis.set_ylabel("Normalized ratio")
    axis.set_title(
        str(model["display_name"]), loc="left", pad=8.0, fontweight="bold"
    )
    axis.set_xticks(
        range(len(DATASET_ORDER)),
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E1E3E5", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.7)
    _draw_axis_break(axis)

    handles = [
        Patch(
            facecolor=str(style["bar_color"]),
            edgecolor=str(style["color"]),
            linewidth=0.55,
            hatch=str(style["hatch"]),
            label="F1 recovery (cap 1)",
        ),
        Line2D(
            [0],
            [0],
            marker=str(style["marker"]),
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=str(style["color"]),
            markeredgewidth=1.2,
            label="KV Rint",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color=str(style["color"]),
            marker=">",
            markevery=[1],
            label="SemPIC Rint",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.3,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.82)

    safe_model = _safe_model_segment(str(model["model_id"]))
    stem = f"{OUTPUT_PREFIX}_{safe_model}"
    return _save_outputs(figure, output_dir, stem)


def _combined_models(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    models = data["models"]
    model_ids = tuple(model["model_id"] for model in models)
    if model_ids != COMBINED_MODEL_ORDER:
        raise ValueError(
            "Combined mode requires the fixed model order "
            f"{list(COMBINED_MODEL_ORDER)}, found {list(model_ids)}"
        )
    return list(models)


def _draw_combined(
    models: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    _configure_style()
    figure, axis = plt.subplots(figsize=(3.4, 2.30))
    y_min, y_max = _axis_limits_for_models(models)
    offsets = (-0.24, 0.0, 0.24)
    bar_width = 0.19

    for model_index, model in enumerate(models):
        model_id = str(model["model_id"])
        safe_model = _safe_model_segment(model_id)
        style = MODEL_STYLES[model_id]
        for dataset_index, (dataset_id, point) in enumerate(
            zip(DATASET_ORDER, model["points"], strict=True)
        ):
            x_value = dataset_index + offsets[model_index]
            raw_recovery = float(point["sempic_recovery"])
            recovery = min(raw_recovery, 1.0)
            kv_rint = float(point["kv_rint"])
            sempic_rint = float(point["sempic_rint"])
            cap_token = "capped-" if raw_recovery > 1.0 else ""

            bars = axis.bar(
                x_value,
                recovery - y_min,
                bottom=y_min,
                width=bar_width,
                color=str(style["bar_color"]),
                edgecolor=str(style["color"]),
                linewidth=0.5,
                hatch=str(style["hatch"]),
                zorder=0,
            )
            bars.patches[0].set_gid(
                f"combined-recovery-bar-{cap_token}{safe_model}-{dataset_id}"
            )

            annotation = axis.annotate(
                "",
                xy=(x_value, sempic_rint),
                xytext=(x_value, kv_rint),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": str(style["color"]),
                    "linewidth": 1.15,
                    "mutation_scale": 7.5,
                    "shrinkA": 2.5,
                    "shrinkB": 2.5,
                },
                zorder=3,
            )
            if annotation.arrow_patch is not None:
                annotation.arrow_patch.set_gid(
                    f"combined-rint-arrow-{safe_model}-{dataset_id}"
                )
            kv_marker = axis.scatter(
                x_value,
                kv_rint,
                s=25,
                marker=str(style["marker"]),
                facecolor="white",
                edgecolor=str(style["color"]),
                linewidth=1.05,
                zorder=4,
            )
            kv_marker.set_gid(f"combined-kv-rint-{safe_model}-{dataset_id}")
            sempic_marker = axis.scatter(
                x_value,
                sempic_rint,
                s=28,
                marker=str(style["marker"]),
                color=str(style["color"]),
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
            sempic_marker.set_gid(
                f"combined-sempic-rint-{safe_model}-{dataset_id}"
            )

    reference = axis.axhline(
        1.0,
        color="#555A60",
        linestyle=(0, (3, 2)),
        linewidth=0.95,
        zorder=1,
    )
    reference.set_gid("combined-unit-reference-line")
    axis.set_xlim(-0.48, len(DATASET_ORDER) - 0.52)
    axis.set_ylim(y_min, y_max)
    axis.set_ylabel("Normalized ratio")
    axis.set_xticks(
        range(len(DATASET_ORDER)),
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E1E3E5", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.7)
    _draw_axis_break(axis)

    metric_handles = [
        Patch(
            facecolor=RECOVERY_COLOR,
            edgecolor="#303030",
            linewidth=0.5,
            hatch="///",
            label="F1 recovery (cap 1)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=KV_COLOR,
            markeredgewidth=1.1,
            label="KV Rint",
        ),
        Line2D(
            [0, 1],
            [0, 0],
            color=SEMPIC_COLOR,
            marker=">",
            markevery=[1],
            label="SemPIC Rint",
        ),
    ]
    model_handles = []
    model_labels = []
    for model in models:
        model_id = str(model["model_id"])
        style = MODEL_STYLES[model_id]
        model_handles.append(
            (
                Patch(
                    facecolor=str(style["bar_color"]),
                    edgecolor=str(style["color"]),
                    linewidth=0.5,
                    hatch=str(style["hatch"]),
                ),
                Line2D(
                    [0],
                    [0],
                    marker=str(style["marker"]),
                    linestyle="",
                    markerfacecolor=str(style["color"]),
                    markeredgecolor="white",
                ),
            )
        )
        model_labels.append(str(model["display_name"]))

    metric_legend = figure.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=0.7,
        handlelength=1.25,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    metric_legend.set_gid("combined-metric-legend")
    model_legend = figure.legend(
        handles=model_handles,
        labels=model_labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.91),
        ncol=3,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.4,
        handletextpad=0.3,
        borderaxespad=0.0,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)},
    )
    model_legend.set_gid("combined-model-legend")
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.83)
    return _save_outputs(figure, output_dir, COMBINED_OUTPUT_STEM)


def draw_multimodel_validation(
    plot_data: str | Path,
    output_dir: str | Path,
    *,
    model_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Path]]:
    data = load_plot_data(plot_data)
    models_by_id = {model["model_id"]: model for model in data["models"]}
    if model_ids is None:
        selected_ids = list(models_by_id)
    else:
        selected_ids = list(model_ids)
        if not selected_ids:
            raise ValueError("model_ids must be omitted or contain at least one model_id")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Repeated --model-id values are not allowed")
        unknown = [model_id for model_id in selected_ids if model_id not in models_by_id]
        if unknown:
            raise ValueError(f"Unknown model_id selection: {unknown}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return {
        model_id: _draw_model(models_by_id[model_id], destination)
        for model_id in selected_ids
    }


def draw_combined_validation(
    plot_data: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write one three-model SVG/PDF/PNG figure from the unified schema."""

    data = load_plot_data(plot_data)
    models = _combined_models(data)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return _draw_combined(models, destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot_data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Render only this exact model_id; repeat to select multiple models.",
    )
    mode.add_argument(
        "--combined",
        action="store_true",
        help="Render one figure containing the fixed three-model evidence matrix.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.combined:
        outputs = draw_combined_validation(args.plot_data, args.output_dir)
        for extension, path in outputs.items():
            print(f"combined {extension}: {path}")
        return 0
    outputs = draw_multimodel_validation(
        args.plot_data,
        args.output_dir,
        model_ids=args.model_id,
    )
    for model_id, paths in outputs.items():
        for extension, path in paths.items():
            print(f"{model_id} {extension}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
