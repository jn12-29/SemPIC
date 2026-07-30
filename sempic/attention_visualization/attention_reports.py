"""Compact multi-page attention-map and error-structure reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .common import (
    A3_LANDSCAPE, ATTENTION_VIEW_ORDER, REGION_ORDER,
    attention_view_label, configure_pdf_metadata, dataset_label, decorate_page,
    display_scale, ensure_parent, label_colorbar, mark_unavailable,
    masked_values, method_label, ordered_datasets, ordered_methods,
    ordered_query_passes, query_pass_label, same_coordinates, set_heatmap_ticks,
)


def plan_attention_report_pages(records) -> dict[str, list[tuple[str, str]]]:
    """Return ordered `(query_pass, attention_view)` identities per report."""
    queries = ordered_query_passes(records)
    pages = [(query, view) for query in queries for view in ATTENTION_VIEW_ORDER]
    return {"attention_maps": pages, "attention_error_structure": pages}


def validate_attention_report_records(records, processing_config) -> None:
    """Require every record combination consumed by the two reports."""
    if not records or len({record["model_id"] for record in records}) != 1:
        raise ValueError("Attention reports require records for exactly one model.")
    ratios = processing_config["edge_ratios"]
    datasets = ordered_datasets(records)
    methods = ordered_methods(records)
    candidates = [method for method in methods if method != "full_recompute"]
    if not datasets or not candidates or methods[:1] != ["full_recompute"]:
        raise ValueError("Attention reports require Full and candidate methods.")
    queries = ordered_query_passes(records)
    expected_pages = [(query, view) for query in queries for view in ATTENTION_VIEW_ORDER]

    for dataset in datasets:
        available = _dataset_methods(records, dataset, candidates)
        if not available:
            raise ValueError(f"Attention report dataset has no candidates: {dataset}.")
        for query, view in expected_pages:
            full = _one(records, dataset=dataset, query=query,
                        metric="attention_profile", plot_view="layer_position_heatmap",
                        method="full_recompute", facets={"attention_view": view})
            heatmaps = [full]
            curves = []
            for method in available:
                heatmaps.append(_one(
                    records, dataset=dataset, query=query,
                    metric="attention_absolute_deviation",
                    plot_view="layer_position_heatmap", method=method,
                    facets={"attention_view": view},
                ))
                for region in REGION_ORDER:
                    for ratio in ratios:
                        facets = {
                            "attention_view": view,
                            "edge_ratio": ratio,
                            "region": region,
                        }
                        curve = _one(
                            records, dataset=dataset, query=query,
                            metric="attention_absolute_deviation",
                            plot_view="layer_curve", method=method,
                            facets=facets, exact_facets=True,
                        )
                        _one(
                            records, dataset=dataset, query=query,
                            metric="attention_absolute_deviation",
                            plot_view="global_bar", method=method,
                            facets=facets, exact_facets=True,
                        )
                        curves.append(curve)
            same_coordinates(heatmaps, "layer")
            same_coordinates(heatmaps, "position_bin")
            same_coordinates(curves, "layer")


def plot_attention_reports(
    records, metric_specs, output_dir, *, model_id, processing_config, fingerprint,
) -> list[Path]:
    """Write attention maps and structured regional-error reports."""
    if {record["model_id"] for record in records} != {model_id}:
        raise ValueError("Attention report model_id does not match the records.")
    for metric in ("attention_profile", "attention_absolute_deviation"):
        if metric not in metric_specs:
            raise ValueError(f"Attention report metric spec is missing: {metric}.")
    root = Path(output_dir)
    pages = plan_attention_report_pages(records)
    maps_path = ensure_parent(root / "attention_maps.pdf")
    error_path = ensure_parent(root / "attention_error_structure.pdf")
    with PdfPages(maps_path) as pdf:
        configure_pdf_metadata(pdf, title=f"{model_id} - Attention maps",
                               fingerprint=fingerprint)
        for index, (query, view) in enumerate(pages["attention_maps"], start=1):
            figure = _plot_maps_page(
                records, metric_specs, model_id, query, view,
                index, len(pages["attention_maps"]),
            )
            pdf.savefig(figure)
            plt.close(figure)
    with PdfPages(error_path) as pdf:
        configure_pdf_metadata(pdf, title=f"{model_id} - Attention error structure",
                               fingerprint=fingerprint)
        for index, (query, view) in enumerate(
            pages["attention_error_structure"], start=1
        ):
            figure = _plot_error_page(
                records, metric_specs, model_id, query, view,
                processing_config["edge_ratios"], index,
                len(pages["attention_error_structure"]),
            )
            pdf.savefig(figure)
            plt.close(figure)
    return [maps_path, error_path]


def _plot_maps_page(records, specs, model, query, view, page_index, page_count):
    datasets = ordered_datasets(records)
    methods = ordered_methods(records)
    candidates = methods[1:]
    figure = plt.figure(figsize=A3_LANDSCAPE, constrained_layout=True)
    grid = figure.add_gridspec(
        len(datasets), len(methods) + 2,
        width_ratios=[1, 0.07] + [1] * len(candidates) + [0.07],
        wspace=0.14, hspace=0.18,
    )
    for row, dataset in enumerate(datasets):
        available = _dataset_methods(records, dataset, candidates)
        full = _one(records, dataset=dataset, query=query,
                    metric="attention_profile", plot_view="layer_position_heatmap",
                    method="full_recompute", facets={"attention_view": view})
        deviations = [
            _one(records, dataset=dataset, query=query,
                 metric="attention_absolute_deviation",
                 plot_view="layer_position_heatmap", method=method,
                 facets={"attention_view": view})
            for method in available
        ]
        full_scale = display_scale([masked_values(full)])
        deviation_scale = display_scale([masked_values(item) for item in deviations])
        full_axis = figure.add_subplot(grid[row, 0])
        full_image = full_axis.imshow(
            masked_values(full), origin="lower", aspect="auto",
            interpolation="nearest", cmap="viridis", norm=full_scale.norm,
        )
        _configure_map_axis(full_axis, full, row, len(datasets), 0, dataset)
        if row == 0:
            full_axis.set_title(method_label("full_recompute"), fontsize=9)
        full_cbar = figure.colorbar(full_image, cax=figure.add_subplot(grid[row, 1]))
        label_colorbar(full_cbar, specs["attention_profile"]["value_label"], full_scale)

        deviation_images = []
        for offset, method in enumerate(candidates, start=2):
            axis = figure.add_subplot(grid[row, offset])
            if method not in available:
                mark_unavailable(axis)
            else:
                record = next(item for item in deviations if item["method_key"] == method)
                image = axis.imshow(
                    masked_values(record), origin="lower", aspect="auto",
                    interpolation="nearest", cmap="viridis", norm=deviation_scale.norm,
                )
                deviation_images.append(image)
                _configure_map_axis(axis, record, row, len(datasets), offset, dataset)
            if row == 0:
                axis.set_title(method_label(method), fontsize=9)
        dev_cbar_axis = figure.add_subplot(grid[row, len(methods) + 1])
        if deviation_images:
            dev_cbar = figure.colorbar(deviation_images[0], cax=dev_cbar_axis)
            label_colorbar(
                dev_cbar, specs["attention_absolute_deviation"]["value_label"],
                deviation_scale,
            )
        else:
            mark_unavailable(dev_cbar_axis)
    decorate_page(
        figure, model=model,
        identity=f"{query_pass_label(query)} · {attention_view_label(view)}",
        purpose="Full attention reference and Full-relative candidate deviation",
        page_index=page_index, page_count=page_count,
    )
    return figure


def _plot_error_page(
    records, specs, model, query, view, ratios, page_index, page_count,
):
    datasets = ordered_datasets(records)
    candidates = ordered_methods(records, include_full=False)
    row_labels = [
        f"{region[0].upper()}{round(float(ratio) * 100):g}"
        for region in REGION_ORDER for ratio in ratios
    ]
    figure = plt.figure(figsize=A3_LANDSCAPE, constrained_layout=True)
    outer = figure.add_gridspec(
        len(datasets), len(candidates) + 1,
        width_ratios=[1] * len(candidates) + [0.07],
        wspace=0.15, hspace=0.2,
    )
    for row, dataset in enumerate(datasets):
        available = _dataset_methods(records, dataset, candidates)
        matrices = {}
        summaries = {}
        for method in available:
            curves = []
            globals_ = []
            for region in REGION_ORDER:
                for ratio in ratios:
                    facets = {"attention_view": view, "edge_ratio": ratio, "region": region}
                    curves.append(_one(
                        records, dataset=dataset, query=query,
                        metric="attention_absolute_deviation", plot_view="layer_curve",
                        method=method, facets=facets, exact_facets=True,
                    ))
                    globals_.append(_one(
                        records, dataset=dataset, query=query,
                        metric="attention_absolute_deviation", plot_view="global_bar",
                        method=method, facets=facets, exact_facets=True,
                    ))
            same_coordinates(curves, "layer")
            matrices[method] = np.stack([masked_values(item) for item in curves])
            summaries[method] = np.asarray([
                float(masked_values(item).item()) for item in globals_
            ]).reshape(-1, 1)
        scale = display_scale(
            [*matrices.values(), *summaries.values()], allow_symlog=True
        )
        images = []
        for column, method in enumerate(candidates):
            if method not in available:
                axis = figure.add_subplot(outer[row, column])
                mark_unavailable(axis)
                if row == 0:
                    axis.set_title(method_label(method), fontsize=9)
                continue
            inner = outer[row, column].subgridspec(
                1, 2, width_ratios=[1, 0.12], wspace=0.08
            )
            matrix_axis = figure.add_subplot(inner[0, 0])
            summary_axis = figure.add_subplot(inner[0, 1], sharey=matrix_axis)
            image = matrix_axis.imshow(
                matrices[method], origin="upper", aspect="auto",
                interpolation="nearest", cmap="viridis", norm=scale.norm,
            )
            images.append(image)
            summary_axis.imshow(
                summaries[method], origin="upper", aspect="auto",
                interpolation="nearest", cmap="viridis", norm=scale.norm,
            )
            _configure_error_axes(
                matrix_axis, summary_axis, matrices[method].shape[1], row_labels,
                row=row, rows=len(datasets), column=column, dataset=dataset,
            )
            if row == 0:
                matrix_axis.set_title(method_label(method), fontsize=9)
                summary_axis.set_title("Mean", fontsize=7, pad=4)
        colorbar_axis = figure.add_subplot(outer[row, len(candidates)])
        if images:
            colorbar = figure.colorbar(images[0], cax=colorbar_axis)
            label_colorbar(
                colorbar, specs["attention_absolute_deviation"]["value_label"], scale
            )
        else:
            mark_unavailable(colorbar_axis)
    decorate_page(
        figure, model=model,
        identity=f"{query_pass_label(query)} · {attention_view_label(view)}",
        purpose="Layer-wise Prefix/Interior/Suffix error across edge ratios",
        page_index=page_index, page_count=page_count, content_top=0.90,
    )
    return figure


def _configure_map_axis(axis, record, row, row_count, column, dataset):
    set_heatmap_ticks(
        axis, record, show_x=row == row_count - 1, show_y=column == 0
    )
    if column == 0:
        axis.set_ylabel(f"{dataset_label(dataset)}\nLayer", fontsize=8)
    if row == row_count - 1:
        axis.set_xlabel("Chunk-local position", fontsize=7.5)


def _configure_error_axes(
    matrix_axis, summary_axis, layer_count, row_labels, *, row, rows, column, dataset,
):
    indices = np.unique(
        np.linspace(0, layer_count - 1, min(7, layer_count)).round().astype(int)
    )
    if row == rows - 1:
        matrix_axis.set_xticks(indices, indices, fontsize=7)
        matrix_axis.set_xlabel("Layer", fontsize=7.5)
    else:
        matrix_axis.set_xticks([])
    matrix_axis.set_yticks(np.arange(len(row_labels)))
    if column == 0:
        matrix_axis.set_yticklabels(row_labels, fontsize=6.5)
        matrix_axis.set_ylabel(dataset_label(dataset), fontsize=8)
    else:
        matrix_axis.set_yticklabels([])
    summary_axis.set_xticks([])
    summary_axis.tick_params(left=False, labelleft=False)
    for spine in summary_axis.spines.values():
        spine.set_linewidth(0.8)


def _dataset_methods(records, dataset, candidates):
    available = {
        record["method_key"] for record in records
        if record["dataset_id"] == dataset and record["method_key"] != "full_recompute"
    }
    return [method for method in candidates if method in available]


def _one(
    records, *, dataset, query, metric, plot_view, method, facets,
    exact_facets=False,
):
    matches = []
    for record in records:
        if (
            record["dataset_id"] != dataset
            or record["query_pass_id"] != query
            or record["metric_key"] != metric
            or record["view_key"] != plot_view
            or record["method_key"] != method
        ):
            continue
        if exact_facets:
            facet_match = record["facets"] == facets
        else:
            facet_match = all(record["facets"].get(key) == value for key, value in facets.items())
        if facet_match:
            matches.append(record)
    identity = (dataset, query, metric, plot_view, method, facets)
    if len(matches) != 1:
        raise ValueError(f"Expected one attention report record for {identity}, found {len(matches)}.")
    return matches[0]


__all__ = [
    "plan_attention_report_pages", "plot_attention_reports",
    "validate_attention_report_records",
]
