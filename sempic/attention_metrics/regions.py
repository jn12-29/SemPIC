"""Directed chunk-position regions used by processed attention metrics."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import numpy as np


REGION_ORDER = ("prefix", "middle", "suffix")


def canonical_ratio(value: object) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError("Edge ratios must be decimal strings or numbers.")
    try:
        ratio = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Edge ratios must be finite decimals.") from exc
    if not ratio.is_finite() or not Decimal("0") < ratio < Decimal("0.5"):
        raise ValueError("Edge ratios must be strictly between 0 and 0.5.")
    return format(ratio.normalize(), "f")


def relative_positions(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Canonical chunks must contain at least one token.")
    return (np.arange(length, dtype=np.float64) + 0.5) / length


def region_masks(length: int, edge_ratio: float) -> dict[str, np.ndarray]:
    if not 0 < edge_ratio < 0.5:
        raise ValueError("edge_ratio must be strictly between 0 and 0.5.")
    positions = relative_positions(length)
    return {
        "prefix": positions < edge_ratio,
        "middle": (positions >= edge_ratio) & (positions < 1 - edge_ratio),
        "suffix": positions >= 1 - edge_ratio,
    }


__all__ = ["REGION_ORDER", "canonical_ratio", "region_masks", "relative_positions"]
