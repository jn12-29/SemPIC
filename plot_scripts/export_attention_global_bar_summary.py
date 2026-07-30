"""Export scalar global-bar attention records from a processed metrics bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

import torch


FIELDNAMES = (
    "model",
    "dataset",
    "query_pass",
    "metric",
    "metric_label",
    "view",
    "method",
    "facets",
    "layer",
    "position_bin",
    "query_head",
    "mean",
    "sem",
    "count",
)


def _scalar(value: object, name: str) -> int | float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, found shape {tuple(value.shape)}.")
        value = value.item()
    if name == "count":
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("count must be a positive integer.")
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def global_bar_rows(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError("Processed metrics payload must be an object.")
    specs = payload.get("metric_specs")
    records = payload.get("records")
    if not isinstance(specs, dict) or not isinstance(records, list):
        raise ValueError("Processed metrics payload has no metric_specs or records.")

    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("view_key") != "global_bar":
            continue
        if record.get("axes") != [] or record.get("coordinates") != {}:
            raise ValueError("global_bar records must be scalar and axis-free.")
        metric_key = record.get("metric_key")
        spec = specs.get(metric_key)
        if not isinstance(metric_key, str) or not isinstance(spec, dict):
            raise ValueError("global_bar record has an unknown metric specification.")
        label = spec.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"Metric {metric_key!r} has no label.")
        facets = record.get("facets")
        if not isinstance(facets, dict):
            raise ValueError("global_bar facets must be an object.")
        rows.append(
            {
                "model": record["model_id"],
                "dataset": record["dataset_id"],
                "query_pass": record["query_pass_id"],
                "metric": metric_key,
                "metric_label": label,
                "view": "global_bar",
                "method": record["method_key"],
                "facets": json.dumps(facets, sort_keys=True, separators=(",", ":")),
                "layer": "",
                "position_bin": "",
                "query_head": "",
                "mean": _scalar(record.get("mean"), "mean"),
                "sem": _scalar(record.get("sem"), "sem"),
                "count": _scalar(record.get("count"), "count"),
            }
        )
    if not rows:
        raise ValueError("Processed metrics payload contains no global_bar records.")
    rows.sort(
        key=lambda row: (
            str(row["model"]),
            str(row["dataset"]),
            str(row["query_pass"]),
            str(row["metric"]),
            str(row["method"]),
            str(row["facets"]),
        )
    )
    return rows


def export_summary(metrics_path: str | Path, output_path: str | Path, *, overwrite: bool) -> Path:
    source = Path(metrics_path).resolve()
    target = Path(output_path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing summary: {target}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    rows = global_bar_rows(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(f"Refusing to replace existing summary: {target}") from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(export_summary(args.metrics, args.output, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
