"""Shared identities, ordering, scales, and page decoration for reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
import numpy as np


A3_LANDSCAPE = (16.54, 11.69)
QUERY_PASS_ORDER = ("terminal_query", "gold_answer", "shifted_prediction")
ATTENTION_VIEW_ORDER = ("raw", "chunk_conditional")
REGION_ORDER = ("prefix", "interior", "suffix")
METHOD_LABELS = {
    "full_recompute": "Full Recompute",
    "vanilla_pic": "Vanilla PIC",
    "kvpacket": "KVPacket",
    "sempic": "SemPIC",
    "sempic_kvpacket": "SemPIC + KVPacket",
}
METHOD_COLORS = {
    "full_recompute": "#4C566A",
    "vanilla_pic": "#D08770",
    "kvpacket": "#EBCB8B",
    "sempic": "#5E81AC",
    "sempic_kvpacket": "#A3BE8C",
}
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}


@dataclass(frozen=True, slots=True)
class DisplayScale:
    norm: Normalize
    display_max: float
    true_max: float
    clipped: bool
    nonlinear: bool


def preferred_order(values: Iterable[str], preferred: tuple[str, ...]) -> list[str]:
    available = set(values)
    return [value for value in preferred if value in available] + sorted(
        available.difference(preferred)
    )


def ordered_methods(records, *, include_full: bool = True) -> list[str]:
    methods = preferred_order(
        (record["method_key"] for record in records), tuple(METHOD_LABELS)
    )
    return methods if include_full else [item for item in methods if item != "full_recompute"]


def ordered_datasets(records) -> list[str]:
    return sorted({record["dataset_id"] for record in records})


def ordered_query_passes(records) -> list[str]:
    return preferred_order(
        (record["query_pass_id"] for record in records), QUERY_PASS_ORDER
    )


def masked_values(record) -> np.ndarray:
    return np.where(record["count"].numpy() > 0, record["mean"].numpy(), np.nan)


def masked_estimate(record) -> tuple[np.ndarray, np.ndarray]:
    valid = record["count"].numpy() > 0
    return (
        np.where(valid, record["mean"].numpy(), np.nan),
        np.where(valid, record["sem"].numpy(), np.nan),
    )


def display_scale(
    arrays: Iterable[np.ndarray], *, allow_symlog: bool = False,
    fixed_range: tuple[float, float] | None = None,
) -> DisplayScale:
    arrays = list(arrays)
    finite_parts = [array[np.isfinite(array)].ravel() for array in arrays]
    finite_parts = [part for part in finite_parts if part.size]
    if fixed_range is not None:
        low, high = fixed_range
        return DisplayScale(Normalize(low, high), high, high, False, False)
    if not finite_parts:
        return DisplayScale(Normalize(0, 1), 1, 1, False, False)
    finite = np.concatenate(finite_parts)
    true_max = max(float(np.max(finite)), 0.0)
    display_max = float(np.percentile(finite, 99.5))
    if display_max <= 0:
        display_max = true_max if true_max > 0 else 1.0
    clipped = true_max > display_max * (1 + 1e-12)
    positive = finite[finite > 0]
    nonlinear = False
    if allow_symlog and positive.size:
        median = float(np.median(positive))
        p95 = float(np.percentile(positive, 95))
        p995 = float(np.percentile(positive, 99.5))
        nonlinear = median > 0 and (
            p95 / median > 20 or p995 / median > 100
        )
    if nonlinear:
        norm = SymLogNorm(
            linthresh=max(float(np.median(positive)), np.finfo(float).tiny),
            linscale=0.8, vmin=0, vmax=display_max, base=10,
        )
    else:
        norm = Normalize(0, display_max)
    return DisplayScale(norm, display_max, max(true_max, display_max), clipped, nonlinear)


def label_colorbar(colorbar, label: str, scale: DisplayScale) -> None:
    suffix = " (symlog)" if scale.nonlinear else ""
    colorbar.set_label(label + suffix, fontsize=7)
    if scale.clipped:
        colorbar.ax.set_title(
            f"clip p99.5={scale.display_max:.2g}\ntrue max={scale.true_max:.2g}",
            fontsize=6.5, pad=3,
        )
    colorbar.ax.tick_params(labelsize=6.5)


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def method_color(method: str):
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    return plt.get_cmap("tab10")(sum(map(ord, method)) % 10)


def dataset_label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset.replace("_", " ").title())


def query_pass_label(query_pass: str) -> str:
    return query_pass.replace("_", " ").title()


def attention_view_label(view: str) -> str:
    return {"raw": "Raw", "chunk_conditional": "Chunk conditional"}.get(
        view, view.replace("_", " ").title()
    )


def mark_unavailable(axis) -> None:
    axis.set_facecolor("#F2F2F2")
    axis.text(0.5, 0.5, "N/A", transform=axis.transAxes,
              ha="center", va="center", color="#666666", fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def set_heatmap_ticks(axis, record, *, show_x: bool, show_y: bool) -> None:
    for dimension, show, setter in ((0, show_y, "y"), (1, show_x, "x")):
        if not show:
            getattr(axis, f"set_{setter}ticks")([])
            continue
        name = record["axes"][dimension]
        values = record["coordinates"][name]
        indices = np.unique(
            np.linspace(0, len(values) - 1, min(5, len(values))).round().astype(int)
        )
        labels = [_coordinate_label(values[index]) for index in indices]
        getattr(axis, f"set_{setter}ticks")(indices, labels, fontsize=7)


def decorate_page(
    figure, *, model: str, identity: str, purpose: str,
    page_index: int, page_count: int, content_top: float = 0.94,
) -> None:
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.015, 1.0, content_top))
    figure.text(
        0.5, 0.987, f"{model} · {identity}",
        ha="center", va="top", fontsize=13,
    )
    figure.text(
        0.5, 0.962, purpose,
        ha="center", va="top", fontsize=12,
    )
    figure.text(
        0.99, 0.008, f"page {page_index}/{page_count}",
        ha="right", va="bottom", fontsize=7, color="#666666",
    )


def configure_pdf_metadata(pdf, *, title: str, fingerprint: str) -> None:
    metadata = pdf.infodict()
    metadata["Title"] = title
    metadata["Subject"] = "SemPIC compact attention analysis"
    metadata["Keywords"] = f"SemPIC attention {fingerprint}"
    metadata["Creator"] = "SemPIC attention visualization"


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def same_coordinates(records, axis: str) -> list[object]:
    values = [record["coordinates"][axis] for record in records]
    if not values or any(value != values[0] for value in values[1:]):
        raise ValueError(f"Report records have mismatched {axis} coordinates.")
    return values[0]


def _coordinate_label(value: object) -> str:
    return f"{value:.2f}" if isinstance(value, float) else str(value)


__all__ = [
    "A3_LANDSCAPE", "ATTENTION_VIEW_ORDER", "DisplayScale", "REGION_ORDER",
    "attention_view_label", "configure_pdf_metadata", "dataset_label",
    "decorate_page", "display_scale", "ensure_parent", "label_colorbar",
    "mark_unavailable", "masked_estimate", "masked_values", "method_color",
    "method_label", "ordered_datasets", "ordered_methods",
    "ordered_query_passes", "preferred_order", "query_pass_label",
    "same_coordinates", "set_heatmap_ticks",
]
