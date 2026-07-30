"""Draw block-local token attention concentration curves for three models."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


SCHEMA_NAME = "sempic.block_token_sink_curve"
SCHEMA_VERSION = 1
MAX_TOKEN_OFFSET = 8
DATASET_COUNT = 4
FULL_SAMPLE_COUNT = 100
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
EXPECTED_COORDINATE = (
    "One-based canonical reusable-block token offset; max_token_offset is the "
    "inclusive maximum offset and wrapper filler is excluded."
)
EXPECTED_AGGREGATION = (
    "Query heads and query rows are equally averaged by the raw reducer; layers "
    "are averaged within each chunk for the token offset and matched interior "
    "separately; chunks are equally averaged within each sample; samples are "
    "equally averaged; ratio is aggregate token-density mean divided by aggregate "
    "interior-density mean."
)
MODEL_STYLES = {
    "Qwen3-4B-Instruct-2507": {
        "display_name": "Qwen3-4B",
        "color": "#2F6B9A",
        "marker": "o",
        "linestyle": "-",
    },
    "Qwen3-8B": {
        "display_name": "Qwen3-8B",
        "color": "#D97706",
        "marker": "s",
        "linestyle": "--",
    },
    "Llama-3.1-8B-Instruct": {
        "display_name": "Llama-3.1-8B",
        "color": "#4F7A61",
        "marker": "D",
        "linestyle": "-.",
    },
}


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _finite_number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{context} must be a finite {qualifier}number")
    return result


def validate_plot_data(data: Any) -> list[dict[str, Any]]:
    """Validate the fixed plotting contract and return normalized model records."""

    root = _mapping(data, "plot data")
    if root.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"schema_name must be {SCHEMA_NAME!r}")
    if type(root.get("schema_version")) is not int or root["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if type(root.get("max_token_offset")) is not int or root["max_token_offset"] != MAX_TOKEN_OFFSET:
        raise ValueError(f"max_token_offset must be {MAX_TOKEN_OFFSET}")
    estimand = _mapping(root.get("estimand"), "estimand")
    if estimand.get("method") != "sempic":
        raise ValueError("estimand.method must be 'sempic'")
    if estimand.get("query_pass_id") != "shifted_prediction":
        raise ValueError("estimand.query_pass_id must be 'shifted_prediction'")
    if estimand.get("coordinate") != EXPECTED_COORDINATE:
        raise ValueError("estimand.coordinate must use the fixed block-local contract")
    if estimand.get("interior") != [0.1, 0.9]:
        raise ValueError("estimand.interior must be [0.1, 0.9]")
    if estimand.get("aggregation") != EXPECTED_AGGREGATION:
        raise ValueError("estimand.aggregation must use the fixed equal-weight contract")

    raw_models = root.get("models")
    if not isinstance(raw_models, list) or len(raw_models) != len(MODEL_STYLES):
        raise ValueError("models must contain exactly the three supported models")

    models_by_id: dict[str, dict[str, Any]] = {}
    expected_offsets = list(range(1, MAX_TOKEN_OFFSET + 1))
    for model_index, raw_model in enumerate(raw_models):
        model_context = f"models[{model_index}]"
        model = _mapping(raw_model, model_context)
        model_id = _string(model.get("model_id"), f"{model_context}.model_id")
        display_name = _string(
            model.get("display_name"), f"{model_context}.display_name"
        )
        if model_id not in MODEL_STYLES:
            raise ValueError(f"Unsupported model_id: {model_id}")
        if model_id in models_by_id:
            raise ValueError(f"Duplicate model_id: {model_id}")
        if display_name != MODEL_STYLES[model_id]["display_name"]:
            raise ValueError(
                f"{model_context}.display_name must be "
                f"{MODEL_STYLES[model_id]['display_name']!r}"
            )

        raw_points = model.get("points")
        if not isinstance(raw_points, list) or len(raw_points) != DATASET_COUNT:
            raise ValueError(f"{model_context}.points must contain exactly four datasets")
        dataset_ids: set[str] = set()
        observed_dataset_order: list[str] = []
        ratios_by_offset: list[list[float]] = [
            [] for _ in range(MAX_TOKEN_OFFSET)
        ]
        for point_index, raw_point in enumerate(raw_points):
            point_context = f"{model_context}.points[{point_index}]"
            point = _mapping(raw_point, point_context)
            dataset_id = _string(
                point.get("dataset_id"), f"{point_context}.dataset_id"
            )
            if dataset_id in dataset_ids:
                raise ValueError(f"Duplicate dataset_id for {model_id}: {dataset_id}")
            dataset_ids.add(dataset_id)
            observed_dataset_order.append(dataset_id)
            sample_count = point.get("sample_count")
            if type(sample_count) is not int or sample_count != FULL_SAMPLE_COUNT:
                raise ValueError(
                    f"{point_context}.sample_count must be {FULL_SAMPLE_COUNT}"
                )

            raw_offsets = point.get("offsets")
            if not isinstance(raw_offsets, list) or len(raw_offsets) != MAX_TOKEN_OFFSET:
                raise ValueError(
                    f"{point_context}.offsets must contain offsets 1 through 8"
                )
            observed_offsets: list[int] = []
            point_interior_density: float | None = None
            for offset_index, raw_offset in enumerate(raw_offsets):
                offset_context = f"{point_context}.offsets[{offset_index}]"
                offset = _mapping(raw_offset, offset_context)
                token_offset = offset.get("token_offset")
                if isinstance(token_offset, bool) or not isinstance(token_offset, int):
                    raise ValueError(f"{offset_context}.token_offset must be an integer")
                observed_offsets.append(token_offset)
                token_density = _finite_number(
                    offset.get("token_density"),
                    f"{offset_context}.token_density",
                )
                if token_density < 0.0:
                    raise ValueError(
                        f"{offset_context}.token_density must be nonnegative"
                    )
                interior_density = _finite_number(
                    offset.get("interior_density"),
                    f"{offset_context}.interior_density",
                    positive=True,
                )
                if point_interior_density is None:
                    point_interior_density = interior_density
                elif not math.isclose(
                    interior_density,
                    point_interior_density,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise ValueError(
                        f"{point_context} must use one matched interior density"
                    )
                ratio = _finite_number(
                    offset.get("ratio"), f"{offset_context}.ratio"
                )
                if ratio < 0.0:
                    raise ValueError(f"{offset_context}.ratio must be nonnegative")
                expected_ratio = token_density / interior_density
                if not math.isclose(ratio, expected_ratio, rel_tol=1e-6, abs_tol=0.0):
                    raise ValueError(
                        f"{offset_context}.ratio must equal token_density / interior_density"
                    )
                if token_offset not in expected_offsets:
                    raise ValueError(f"{offset_context}.token_offset is out of range")
                ratios_by_offset[token_offset - 1].append(ratio)
            if observed_offsets != expected_offsets:
                raise ValueError(f"{point_context}.offsets must be ordered 1 through 8")
        if observed_dataset_order != list(DATASET_ORDER):
            raise ValueError(
                f"{model_context}.points must follow the fixed dataset order"
            )

        models_by_id[model_id] = {
            "model_id": model_id,
            "display_name": display_name,
            "ratios_by_offset": ratios_by_offset,
        }

    missing = [model_id for model_id in MODEL_STYLES if model_id not in models_by_id]
    if missing:
        raise ValueError(f"Missing supported models: {', '.join(missing)}")
    return [models_by_id[model_id] for model_id in MODEL_STYLES]


def _geometric_mean(values: Sequence[float]) -> float:
    if any(value == 0.0 for value in values):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in values))


def draw_block_token_curve(data: Any, output_prefix: str | Path) -> dict[str, Path]:
    """Validate data and write the block-local token curve as SVG, PDF, and PNG."""

    models = validate_plot_data(data)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update(
        {
            "font.size": 8.3,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "lines.solid_capstyle": "round",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    x_values = list(range(1, MAX_TOKEN_OFFSET + 1))
    figure, axis = plt.subplots(figsize=(3.53, 2.51))
    for model in models:
        style = MODEL_STYLES[model["model_id"]]
        ratios_by_offset = model["ratios_by_offset"]
        centers = [_geometric_mean(values) for values in ratios_by_offset]
        lower = [min(values) for values in ratios_by_offset]
        upper = [max(values) for values in ratios_by_offset]
        plotted_centers = [value if value > 0.0 else math.nan for value in centers]
        plotted_lower = [value if value > 0.0 else math.nan for value in lower]
        axis.fill_between(
            x_values,
            plotted_lower,
            upper,
            color=style["color"],
            alpha=0.09,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x_values,
            plotted_centers,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.55,
            marker=style["marker"],
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=0.9,
            label=style["display_name"],
            zorder=3,
        )

    axis.axhline(
        1.0,
        color="#777777",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    axis.set_yscale("log")
    axis.set_xlim(0.82, MAX_TOKEN_OFFSET + 0.18)
    axis.set_xticks(x_values)
    axis.set_xlabel("Token offset in block", labelpad=2.5)
    axis.set_ylabel("Attention density / interior", labelpad=2.5)
    axis.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12)
    )
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.grid(axis="y", which="major", color="#D8D8D8", linewidth=0.55, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="major", length=2.8, width=0.65, pad=2)
    axis.tick_params(axis="y", which="minor", length=1.5, width=0.45)
    axis.legend(
        loc="upper right",
        fontsize=8.0,
        frameon=True,
        facecolor="white",
        edgecolor="#C8C8C8",
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=2.2,
        handletextpad=0.55,
    )
    figure.subplots_adjust(left=0.175, right=0.985, bottom=0.205, top=0.98)

    outputs = {
        extension: Path(f"{prefix}.{extension}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with args.data.open(encoding="utf-8") as source:
        data = json.load(source)
    outputs = draw_block_token_curve(data, args.output_prefix)
    for extension in ("svg", "pdf", "png"):
        print(outputs[extension])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
