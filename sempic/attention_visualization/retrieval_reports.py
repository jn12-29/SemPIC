"""Compact multi-page retrieval reports rendered from processed records."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .common import (
    A3_LANDSCAPE,
    configure_pdf_metadata,
    dataset_label,
    decorate_page,
    display_scale,
    ensure_parent,
    label_colorbar,
    mark_unavailable,
    masked_estimate,
    masked_values,
    method_color,
    method_label,
    ordered_datasets,
    ordered_methods,
    ordered_query_passes,
    query_pass_label,
    same_coordinates,
    set_heatmap_ticks,
)


_REPORTS = (
    ("retrieval_nrmse", "retrieval_nrmse.pdf", "Retrieval NRMSE"),
    (
        "retrieval_cosine_distance",
        "retrieval_cosine_distance.pdf",
        "Retrieval cosine distance",
    ),
    ("attention_mass_error", "retrieval_mass_error.pdf", "Retrieval mass error"),
)
_VIEWS = ("layer_head_heatmap", "layer_curve", "global_bar")


def plan_retrieval_report_pages(
    records: list[dict[str, object]],
) -> dict[str, list[str]]:
    """Return ordered query-page identities for every retrieval report."""
    return {
        metric: ordered_query_passes([
            record for record in records if record["metric_key"] == metric
        ])
        for metric, _, _ in _REPORTS
    }


def validate_retrieval_report_records(records: list[dict[str, object]]) -> None:
    """Require the complete record triplets needed by compact retrieval reports."""
    if not records:
        raise ValueError("Retrieval reports require processed metric records.")
    models = {record["model_id"] for record in records}
    if len(models) != 1:
        raise ValueError("Retrieval report records must describe exactly one model.")

    query_passes = ordered_query_passes(records)
    datasets = ordered_datasets(records)
    methods = ordered_methods(records, include_full=False)
    if not query_passes or not datasets or not methods:
        raise ValueError("Retrieval reports require query passes, datasets, and candidates.")

    page_plan = plan_retrieval_report_pages(records)
    for metric, _, _ in _REPORTS:
        if page_plan[metric] != query_passes:
            raise ValueError(
                f"Retrieval report {metric} does not cover query passes {query_passes}."
            )

    required_metrics = {metric for metric, _, _ in _REPORTS}
    lookup: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for record in records:
        metric = record["metric_key"]
        if metric not in required_metrics:
            continue
        if record["method_key"] == "full_recompute":
            raise ValueError(f"Retrieval report metric {metric} cannot use Full Recompute.")
        if record["view_key"] not in _VIEWS or record["facets"]:
            raise ValueError(
                "Retrieval report records require an empty facet set and one of "
                f"{_VIEWS}: {metric}/{record['view_key']}."
            )
        key = (
            record["dataset_id"],
            record["query_pass_id"],
            metric,
            record["view_key"],
            record["method_key"],
        )
        if key in lookup:
            raise ValueError(f"Duplicate retrieval report record: {key}.")
        lookup[key] = record

    for dataset in datasets:
        available_methods = _dataset_methods(records, dataset, methods)
        for query_pass in query_passes:
            for metric, _, _ in _REPORTS:
                heatmaps = []
                for method in available_methods:
                    triplet = []
                    for view in _VIEWS:
                        key = (dataset, query_pass, metric, view, method)
                        record = lookup.get(key)
                        if record is None:
                            raise ValueError(f"Missing retrieval report record: {key}.")
                        triplet.append(record)
                    heatmap, curve, _ = triplet
                    same_coordinates([heatmap, curve], "layer")
                    heatmaps.append(heatmap)
                if heatmaps:
                    same_coordinates(heatmaps, "layer")
                    same_coordinates(heatmaps, "query_head")


def plot_retrieval_reports(
    records: list[dict[str, object]],
    metric_specs: dict[str, dict[str, object]],
    output_dir: str | Path,
    *,
    model_id: str,
    fingerprint: str,
) -> list[Path]:
    """Write the three compact retrieval PDFs for one model."""
    if {record["model_id"] for record in records} != {model_id}:
        raise ValueError("Retrieval report model_id does not match the records.")
    for metric, _, _ in _REPORTS:
        if metric not in metric_specs:
            raise ValueError(f"Retrieval report metric spec is missing: {metric}.")

    page_plan = plan_retrieval_report_pages(records)
    root = Path(output_dir)
    outputs = []
    for metric, filename, report_label in _REPORTS:
        path = ensure_parent(root / filename)
        query_passes = page_plan[metric]
        with PdfPages(path) as pdf:
            configure_pdf_metadata(
                pdf, title=f"{model_id} - {report_label}", fingerprint=fingerprint
            )
            for page_index, query_pass in enumerate(query_passes, start=1):
                figure = _plot_retrieval_page(
                    records,
                    metric_specs[metric],
                    metric=metric,
                    report_label=report_label,
                    model_id=model_id,
                    query_pass=query_pass,
                    page_index=page_index,
                    page_count=len(query_passes),
                )
                pdf.savefig(figure)
                plt.close(figure)
        outputs.append(path)
    return outputs


def _plot_retrieval_page(
    records,
    metric_spec,
    *,
    metric,
    report_label,
    model_id,
    query_pass,
    page_index,
    page_count,
):
    datasets = ordered_datasets(records)
    methods = ordered_methods(records, include_full=False)
    page_records = [
        record for record in records
        if record["query_pass_id"] == query_pass
        and record["metric_key"] == metric
    ]
    lookup = {
        (record["dataset_id"], record["method_key"], record["view_key"]): record
        for record in page_records
    }
    figure = plt.figure(figsize=A3_LANDSCAPE, constrained_layout=True)
    grid = figure.add_gridspec(
        len(datasets),
        len(methods) + 2,
        width_ratios=[1.0] * len(methods) + [0.09, 1.8],
        wspace=0.18,
        hspace=0.28,
    )

    for row, dataset in enumerate(datasets):
        available_methods = _dataset_methods(records, dataset, methods)
        heatmaps = [
            lookup[(dataset, method, "layer_head_heatmap")]
            for method in available_methods
        ]
        scale = _retrieval_scale(metric, heatmaps)
        images = []
        for column, method in enumerate(methods):
            axis = figure.add_subplot(grid[row, column])
            if method not in available_methods:
                mark_unavailable(axis)
                axis.set_title(method_label(method), fontsize=8, pad=7)
            else:
                heatmap = lookup[(dataset, method, "layer_head_heatmap")]
                global_record = lookup[(dataset, method, "global_bar")]
                image = axis.imshow(
                    masked_values(heatmap),
                    origin="lower",
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    norm=scale.norm,
                )
                images.append(image)
                set_heatmap_ticks(
                    axis,
                    heatmap,
                    show_x=row == len(datasets) - 1,
                    show_y=column == 0,
                )
                axis.set_title(
                    f"{method_label(method)}\n{_global_label(global_record)}",
                    fontsize=7.5,
                    pad=6,
                )
            if column == 0:
                axis.set_ylabel(f"{dataset_label(dataset)}\nLayer", fontsize=8)
            if row == len(datasets) - 1:
                axis.set_xlabel("Query head", fontsize=7.5)

        colorbar_axis = figure.add_subplot(grid[row, len(methods)])
        if images:
            colorbar = figure.colorbar(images[0], cax=colorbar_axis)
            label_colorbar(colorbar, metric_spec["value_label"], scale)
        else:
            mark_unavailable(colorbar_axis)

        curve_axis = figure.add_subplot(grid[row, len(methods) + 1])
        curve_records = []
        for method in available_methods:
            curve = lookup[(dataset, method, "layer_curve")]
            curve_records.append(curve)
            layers = np.asarray(curve["coordinates"]["layer"])
            mean, sem = masked_estimate(curve)
            color = method_color(method)
            curve_axis.plot(
                layers, mean, color=color, linewidth=1.6, label=method_label(method)
            )
            curve_axis.fill_between(
                layers,
                np.maximum(0, mean - sem),
                mean + sem,
                color=color,
                alpha=0.13,
            )
        _configure_curve_axis(curve_axis, metric, curve_records, metric_spec)
        curve_axis.set_title("Layer-wise comparison", fontsize=8)
        if row == len(datasets) - 1:
            curve_axis.set_xlabel("Layer", fontsize=7.5)
        curve_axis.legend(loc="best", frameon=False, fontsize=6.5)

    decorate_page(
        figure,
        model=model_id,
        identity=f"{query_pass_label(query_pass)} · {report_label}",
        purpose="Per-head retrieval error and layer-wise method comparison",
        page_index=page_index,
        page_count=page_count,
        content_top=0.88,
    )
    return figure


def _dataset_methods(records, dataset: str, methods: list[str]) -> list[str]:
    available = {
        record["method_key"]
        for record in records
        if record["dataset_id"] == dataset
        and record["method_key"] != "full_recompute"
    }
    return [method for method in methods if method in available]


def _retrieval_scale(metric: str, heatmaps):
    arrays = [masked_values(record) for record in heatmaps]
    if metric == "retrieval_cosine_distance":
        return display_scale(arrays, fixed_range=(0, 2))
    return display_scale(arrays, allow_symlog=metric == "retrieval_nrmse")


def _configure_curve_axis(axis, metric: str, records, metric_spec) -> None:
    axis.grid(alpha=0.2)
    ylabel = metric_spec["value_label"]
    if metric == "retrieval_cosine_distance":
        axis.set_ylim(0, 2)
    elif metric == "retrieval_nrmse":
        scale = display_scale(
            [masked_values(record) for record in records], allow_symlog=True
        )
        if scale.nonlinear:
            axis.set_yscale(
                "symlog",
                linthresh=scale.norm.linthresh,
                linscale=0.8,
                base=10,
            )
            ylabel += " (symlog)"
        axis.set_ylim(bottom=0)
    else:
        axis.set_ylim(bottom=0)
    axis.set_ylabel(ylabel, fontsize=7.5)
    axis.tick_params(labelsize=7)


def _global_label(record) -> str:
    mean = float(record["mean"].item())
    sem = float(record["sem"].item())
    return f"mean {mean:.3g} ± {sem:.2g}"


__all__ = [
    "plan_retrieval_report_pages",
    "plot_retrieval_reports",
    "validate_retrieval_report_records",
]
