"""Draw the compact single-column Boundary Motivation figure."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_COLUMNS = {
    "model_id",
    "dataset_id",
    "status",
    "full_recompute_status",
    "no_recompute_status",
    "kvpacket_status",
    "f1_recovery_fraction",
    "f1_recovery_status",
    "prefix_attention_ratio",
    "prefix_attention_ratio_status",
    "prefix_vanilla_count",
    "prefix_kvpacket_count",
    "interior_attention_ratio",
    "interior_attention_ratio_status",
    "interior_vanilla_count",
    "interior_kvpacket_count",
}
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
DATASET_LABELS = {
    "biography": "Biography",
    "hotpot_qa": "HotpotQA",
    "musique": "MuSiQue",
    "niah": "NIAH",
}
BAR_COLOR = "#DCEAF4"
ARROW_COLOR = "#245B8E"
MODEL_MARKERS = {"4B": "o", "8B": "D"}


def _optional_float(row: Mapping[str, str], field: str) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"Invalid numeric {field}: {raw!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"Non-finite numeric {field}: {raw!r}")
    return value


def _sample_count(rows: Sequence[Mapping[str, Any]]) -> int:
    fields = (
        "prefix_vanilla_count",
        "prefix_kvpacket_count",
        "interior_vanilla_count",
        "interior_kvpacket_count",
    )
    try:
        counts = {int(row[field]) for row in rows for field in fields}
    except (TypeError, ValueError) as error:
        raise ValueError("Attention counts must be positive integers") from error
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError(f"Expected one positive attention sample count, found {sorted(counts)}")
    return next(iter(counts))


def _short_model(model_id: str) -> str:
    for short_name in MODEL_MARKERS:
        if short_name in model_id:
            return short_name
    raise ValueError(f"Unsupported model identity: {model_id}")


def load_plot_data(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Plot data lacks required columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for field in (
                "f1_recovery_fraction",
                "prefix_attention_ratio",
                "interior_attention_ratio",
            ):
                row[field] = _optional_float(raw, field)
            rows.append(row)

    identities = [(row["model_id"], row["dataset_id"]) for row in rows]
    if len(rows) != 8 or len(set(identities)) != 8:
        raise ValueError(f"Expected eight unique model-dataset rows, found {identities}")
    if {row["dataset_id"] for row in rows} != set(DATASET_ORDER):
        raise ValueError("Plot data must retain all four declared datasets")
    model_ids = sorted({row["model_id"] for row in rows})
    if {_short_model(model_id) for model_id in model_ids} != set(MODEL_MARKERS):
        raise ValueError(f"Plot data must contain one 4B and one 8B model, found {model_ids}")
    order = {dataset: index for index, dataset in enumerate(DATASET_ORDER)}
    rows.sort(key=lambda row: (order[row["dataset_id"]], _short_model(row["model_id"])))
    return rows


def _safe_stem(stem: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stem):
        raise ValueError(f"Unsafe output stem: {stem!r}")
    return stem


def _require_complete(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        identity = f"{row['model_id']}/{row['dataset_id']}"
        if row["status"] != "pass":
            raise ValueError(f"Incomplete evidence point: {identity}")
        for field in ("full_recompute_status", "no_recompute_status", "kvpacket_status"):
            if row[field] != "matched":
                raise ValueError(f"Unmatched behavior result for {identity}: {field}")
        for value_field, status_field in (
            ("f1_recovery_fraction", "f1_recovery_status"),
            ("prefix_attention_ratio", "prefix_attention_ratio_status"),
            ("interior_attention_ratio", "interior_attention_ratio_status"),
        ):
            value = row[value_field]
            if row[status_field] != "defined" or value is None or value < 0:
                raise ValueError(f"Undefined {value_field} for {identity}")


def draw_diagnostic(
    plot_data_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "boundary_motivation_diagnostic",
    overwrite: bool = False,
) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = load_plot_data(plot_data_path)
    _require_complete(rows)
    _sample_count(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_stem(stem)
    outputs = {suffix: destination / f"{safe_stem}.{suffix}" for suffix in ("svg", "pdf", "png")}
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite diagnostic outputs: {existing}")

    model_offsets = {"4B": -0.17, "8B": 0.17}
    x_groups = list(range(len(DATASET_ORDER)))
    recoveries = [float(row["f1_recovery_fraction"]) for row in rows]
    attention = [
        float(row[field])
        for row in rows
        for field in ("prefix_attention_ratio", "interior_attention_ratio")
    ]
    y_max = max(1.0, *(min(value, 1.0) for value in recoveries), *attention) * 1.08

    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(3.4, 2.18))

    for row in rows:
        short_model = _short_model(row["model_id"])
        x_value = DATASET_ORDER.index(row["dataset_id"]) + model_offsets[short_model]
        recovery = float(row["f1_recovery_fraction"])
        prefix = float(row["prefix_attention_ratio"])
        interior = float(row["interior_attention_ratio"])
        marker = MODEL_MARKERS[short_model]

        axis.bar(
            x_value,
            min(recovery, 1.0),
            width=0.29,
            color=BAR_COLOR,
            edgecolor="none",
            zorder=0,
        )
        if recovery > 1.0:
            for center in (-0.10, 0.10):
                axis.plot(
                    [x_value + center - 0.025, x_value + center + 0.025],
                    [0.975, 1.025],
                    color="#55748B",
                    linewidth=1.0,
                    zorder=2,
                )
        axis.annotate(
            "",
            xy=(x_value, interior),
            xytext=(x_value, prefix),
            arrowprops={
                "arrowstyle": "-|>",
                "color": ARROW_COLOR,
                "linewidth": 1.35,
                "mutation_scale": 9,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=3,
        )
        axis.scatter(
            x_value,
            prefix,
            s=28,
            marker=marker,
            color=ARROW_COLOR,
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        axis.scatter(
            x_value,
            interior,
            s=31,
            marker=marker,
            facecolor="white",
            edgecolor=ARROW_COLOR,
            linewidth=1.15,
            zorder=4,
        )

    axis.axhline(1.0, color="#5D636A", linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    axis.set_xlim(-0.48, len(DATASET_ORDER) - 0.52)
    axis.set_ylim(0.0, y_max)
    axis.set_ylabel("Normalized ratio")
    axis.set_xticks(
        x_groups,
        [DATASET_LABELS[dataset] for dataset in DATASET_ORDER],
        rotation=19,
        ha="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="y", color="#E5E8EB", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, width=0.65)

    handles = [
        Patch(facecolor=BAR_COLOR, edgecolor="none", label="F1 recovery"),
        Line2D([0, 1], [0, 0], color=ARROW_COLOR, marker=">", markevery=[1], label="Rpre → Rint"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=ARROW_COLOR, label="4B"),
        Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor=ARROW_COLOR, label="8B"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        ncol=4,
        frameon=False,
        columnspacing=0.75,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.1,
    )
    figure.subplots_adjust(left=0.17, right=0.985, bottom=0.25, top=0.82)

    for suffix, path in outputs.items():
        figure.savefig(
            path,
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(figure)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stem", default="boundary_motivation_diagnostic")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = draw_diagnostic(
        args.plot_data,
        args.output_dir,
        stem=args.stem,
        overwrite=args.overwrite,
    )
    for suffix, path in outputs.items():
        print(f"{suffix}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
