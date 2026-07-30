from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .metrics import metric_is_lower_better, metric_numeric
from .schema import METRIC_PREFIX, NormalizedRecord


FIXED_COLUMNS = [
    "source_path",
    "result_dir",
    "run_label",
    "modified_at",
    "config",
    "result",
    "model_path",
    "model_name",
    "dataset_name",
    "benchmark_label",
    "method_raw",
    "method",
    "method_label",
    "dataset_seed",
    "run_seed",
    "comparison_id",
    "comparison_label",
    "checkpoint_id",
    "checkpoint_label",
    "checkpoint_source_dataset",
    "checkpoint_scope",
    "algorithm_variant_id",
    "algorithm_variant_label",
    "series_id",
    "series_label",
    "config_json",
    "result_json",
]

CHECKPOINT_OPTION_COLUMNS = [
    "checkpoint_id",
    "checkpoint_label",
    "method",
    "method_label",
    "checkpoint_source_dataset",
]

RUN_LEADERBOARD_COLUMNS = [
    "Method",
    "Run label",
    "Checkpoint",
    "F1",
    "TTFT Mean",
    "TTFT P50",
    "TTFT P99",
    "FLOPs",
    "Result path",
]

_RUN_LEADERBOARD_SORT_COLUMNS = {
    "metric.f1": "F1",
    "metric.ttft": "TTFT Mean",
    "metric.ttft_mean": "TTFT Mean",
    "metric.ttft_p50": "TTFT P50",
    "metric.ttft_p99": "TTFT P99",
    "metric.flops": "FLOPs",
}


@dataclass(frozen=True, slots=True)
class SmartTableResult:
    table: pd.DataFrame
    included_rows: int
    excluded_cross_dataset: int
    excluded_unresolved: int
    provenance: pd.DataFrame = field(default_factory=pd.DataFrame)
    aggregation_label: str = ""
    grouping_description: str = ""
    included_checkpoints: tuple[str, ...] = ()
    estimator: str = ""


EXACT_METRIC_COLUMNS = [
    "Comparison",
    "Method",
    "Run label",
    "Checkpoint",
    "Value",
    "Result path",
]

PROVENANCE_COLUMNS = [
    "included",
    "status",
    "reason",
    "comparison_id",
    "comparison_label",
    "algorithm_variant_id",
    "algorithm_variant_label",
    "series_id",
    "series_label",
    "method",
    "method_label",
    "checkpoint_id",
    "checkpoint_label",
    "run_label",
    "source_path",
    "metric",
    "metric_value",
]

SUMMARY_COLUMNS = [
    "comparison_id",
    "comparison_label",
    "model_path",
    "model_name",
    "dataset_name",
    "benchmark_label",
    "series_id",
    "series_label",
    "method",
    "method_label",
    "run_label",
    "metric",
    "mean",
    "std",
    "count",
]

SUMMARY_PROVENANCE_COLUMNS = SUMMARY_COLUMNS + [
    "estimator",
    "grouping_description",
    "source_count",
    "checkpoint_count",
    "source_paths",
    "checkpoint_ids",
]


def _raw_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=True)


def records_to_frame(records: Iterable[NormalizedRecord]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in records:
        source_path = str(record.source_path.resolve())
        if source_path in seen:
            continue
        seen.add(source_path)
        row: dict[str, object] = {
            "source_path": source_path,
            "result_dir": str(record.result_dir.resolve()),
            "run_label": record.run_label,
            "modified_at": record.modified_at,
            "config": record.config,
            "result": record.result,
            "model_path": record.model_path,
            "model_name": record.model_name,
            "dataset_name": record.dataset_name,
            "benchmark_label": record.benchmark_label,
            "method_raw": record.method_raw,
            "method": record.method,
            "method_label": record.method_label,
            "dataset_seed": record.dataset_seed,
            "run_seed": record.run_seed,
            "comparison_id": record.comparison_id,
            "comparison_label": record.comparison_label,
            "checkpoint_id": record.checkpoint_id,
            "checkpoint_label": record.checkpoint_label,
            "checkpoint_source_dataset": record.checkpoint_source_dataset,
            "checkpoint_scope": record.checkpoint_scope,
            "algorithm_variant_id": record.algorithm_variant_id,
            "algorithm_variant_label": record.algorithm_variant_label,
            "series_id": record.series_id,
            "series_label": record.series_label,
            "config_json": _raw_json(record.config),
            "result_json": _raw_json(record.result),
        }
        row.update({f"{METRIC_PREFIX}{key}": value for key, value in record.metrics.items()})
        rows.append(row)
    metric_columns = sorted(
        {column for row in rows for column in row if column.startswith(METRIC_PREFIX)}
    )
    return pd.DataFrame(rows, columns=FIXED_COLUMNS + metric_columns)


def build_run_leaderboard(
    frame: pd.DataFrame,
    comparison_id: str | None = None,
    sort_metric: str = "metric.f1",
    ascending: bool | None = None,
) -> pd.DataFrame:
    """Build a sortable one-row-per-result leaderboard without aggregation."""
    if sort_metric not in _RUN_LEADERBOARD_SORT_COLUMNS:
        raise ValueError(f"unsupported run leaderboard sort metric: {sort_metric}")
    if frame.empty:
        return pd.DataFrame(columns=RUN_LEADERBOARD_COLUMNS)

    selected = frame
    if comparison_id:
        selected = selected.loc[selected["comparison_id"] == comparison_id]
    if selected.empty:
        return pd.DataFrame(columns=RUN_LEADERBOARD_COLUMNS)

    def text_column(column: str) -> pd.Series:
        values = selected.get(column, pd.Series(index=selected.index, dtype=object))
        return values.map(lambda value: value if isinstance(value, str) else "")

    leaderboard = pd.DataFrame(
        {
            "Method": text_column("method_label"),
            "Run label": text_column("run_label"),
            "Checkpoint": text_column("checkpoint_label"),
            "F1": metric_numeric(selected, "metric.f1"),
            "TTFT Mean": metric_numeric(selected, "metric.ttft_mean"),
            "TTFT P50": metric_numeric(selected, "metric.ttft_p50"),
            "TTFT P99": metric_numeric(selected, "metric.ttft_p99"),
            "FLOPs": metric_numeric(selected, "metric.flops"),
            "Result path": text_column("source_path"),
        },
        index=selected.index,
    )
    sort_column = _RUN_LEADERBOARD_SORT_COLUMNS[sort_metric]
    if ascending is None:
        ascending = metric_is_lower_better(sort_metric)
    return leaderboard.sort_values(
        [sort_column, "Result path"],
        ascending=[ascending, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)


def _shortest_unique_prefixes(ids: list[str], minimum: int = 8) -> dict[str, str]:
    if len(ids) <= 1:
        return {identity: identity[:minimum] for identity in ids}
    for length in range(minimum, 65):
        prefixes = {identity: identity[:length] for identity in ids}
        if len(set(prefixes.values())) == len(ids):
            return prefixes
    return {identity: identity for identity in ids}


def _unique_display_labels(labels: pd.Series, identities: pd.Series) -> pd.Series:
    base_labels = labels.astype(str)
    result = base_labels.copy()
    identity_strings = identities.astype(str)
    for _, indexes in base_labels.groupby(base_labels).groups.items():
        unique_ids = list(dict.fromkeys(identity_strings.loc[indexes]))
        if len(unique_ids) <= 1:
            continue
        prefixes = _shortest_unique_prefixes(unique_ids)
        for index in indexes:
            result.loc[index] = (
                f"{base_labels.loc[index]} [{prefixes[identity_strings.loc[index]]}]"
            )

    while True:
        conflicting_groups = [
            indexes
            for _, indexes in result.groupby(result).groups.items()
            if identity_strings.loc[indexes].nunique() > 1
        ]
        if not conflicting_groups:
            return result
        for indexes in conflicting_groups:
            unique_ids = list(dict.fromkeys(identity_strings.loc[indexes]))
            prefixes = _shortest_unique_prefixes(unique_ids)
            for index in indexes:
                result.loc[index] = (
                    f"{base_labels.loc[index]} [{prefixes[identity_strings.loc[index]]}]"
                )


def checkpoint_options(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "checkpoint_id" not in frame.columns:
        return pd.DataFrame(columns=CHECKPOINT_OPTION_COLUMNS)
    selected = frame.loc[
        frame["checkpoint_id"].map(lambda value: isinstance(value, str) and bool(value)),
        CHECKPOINT_OPTION_COLUMNS,
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=CHECKPOINT_OPTION_COLUMNS)
    selected = selected.drop_duplicates(subset="checkpoint_id", keep="first")
    selected["checkpoint_label"] = _unique_display_labels(
        selected["checkpoint_label"], selected["checkpoint_id"]
    )
    return selected.sort_values(
        ["checkpoint_label", "checkpoint_id"], kind="stable"
    ).reset_index(drop=True)


def _stable_mean(values: pd.Series) -> float:
    numeric = [float(value) for value in values]
    scale = max(abs(value) for value in numeric)
    if scale == 0.0:
        return 0.0
    scaled_mean = math.fsum(value / scale for value in numeric) / len(numeric)
    return scale * scaled_mean


def _stable_std(values: pd.Series) -> float:
    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        return math.nan
    scale = max(abs(value) for value in numeric)
    if scale == 0.0:
        return 0.0
    scaled_mean = math.fsum(value / scale for value in numeric) / len(numeric)
    variance = math.fsum(
        ((value / scale) - scaled_mean) ** 2 for value in numeric
    ) / (len(numeric) - 1)
    return scale * math.sqrt(variance)


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = sorted(
        column for column in frame.columns if column.startswith(METRIC_PREFIX)
    )
    if frame.empty or not metric_columns:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    group_columns = SUMMARY_COLUMNS[:11]
    long_frame = frame.melt(
        id_vars=group_columns,
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="coerce")
    long_frame = long_frame.dropna(subset=["value"])
    if long_frame.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = (
        long_frame.groupby(group_columns + ["metric"], dropna=False, sort=True)["value"]
        .agg(mean=_stable_mean, std=_stable_std, count="count")
        .reset_index()
    )
    summary["count"] = summary["count"].astype(int)
    return summary[SUMMARY_COLUMNS]


def summarize_results_with_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the legacy metric summary with auditable source membership."""
    summary = summarize_results(frame)
    if summary.empty:
        return pd.DataFrame(columns=SUMMARY_PROVENANCE_COLUMNS)

    metric_columns = sorted(
        column for column in frame.columns if column.startswith(METRIC_PREFIX)
    )
    group_columns = SUMMARY_COLUMNS[:11]
    membership = frame.melt(
        id_vars=group_columns + ["source_path", "checkpoint_id"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    membership["value"] = pd.to_numeric(membership["value"], errors="coerce")
    membership = membership.dropna(subset=["value"])

    def serialized_values(values: pd.Series) -> str:
        return json.dumps(
            sorted(
                {
                    str(value)
                    for value in values
                    if isinstance(value, str) and value
                }
            ),
            ensure_ascii=False,
        )

    provenance = (
        membership.groupby(group_columns + ["metric"], dropna=False, sort=True)
        .agg(
            source_count=("source_path", "nunique"),
            checkpoint_count=(
                "checkpoint_id",
                lambda values: len(
                    {
                        value
                        for value in values
                        if isinstance(value, str) and value
                    }
                ),
            ),
            source_paths=("source_path", serialized_values),
            checkpoint_ids=("checkpoint_id", serialized_values),
        )
        .reset_index()
    )
    result = summary.merge(
        provenance,
        on=group_columns + ["metric"],
        how="left",
        validate="one_to_one",
    )
    result["estimator"] = "Arithmetic mean; sample standard deviation (n-1)"
    result["grouping_description"] = (
        "comparison_id + series_id + run_label + metric"
    )
    return result[SUMMARY_PROVENANCE_COLUMNS]


def _display_values(frame: pd.DataFrame, label: str, identity: str) -> pd.Series:
    return _unique_display_labels(frame[label], frame[identity])


def _empty_smart_table(
    *,
    mode: str,
    provenance: pd.DataFrame | None = None,
    excluded_cross_dataset: int = 0,
    excluded_unresolved: int = 0,
) -> SmartTableResult:
    return SmartTableResult(
        table=pd.DataFrame(
            columns=EXACT_METRIC_COLUMNS if mode == "exact_runs" else ["Comparison"]
        ),
        included_rows=0,
        excluded_cross_dataset=excluded_cross_dataset,
        excluded_unresolved=excluded_unresolved,
        provenance=(
            provenance
            if provenance is not None
            else pd.DataFrame(columns=PROVENANCE_COLUMNS)
        ),
        aggregation_label=_aggregation_metadata(mode)[0],
        grouping_description=_aggregation_metadata(mode)[1],
        estimator=_aggregation_metadata(mode)[2],
    )


def _ordinary_table(pivot: pd.DataFrame) -> pd.DataFrame:
    if pivot.empty:
        return pd.DataFrame(columns=["Comparison"])
    table = pivot.rename_axis(index="Comparison", columns=None).reset_index()
    table.index = pd.RangeIndex(len(table))
    return table


def _aggregation_metadata(mode: str) -> tuple[str, str, str]:
    if mode == "exact_runs":
        return (
            "Exact observations",
            "No aggregation; one row per numeric source observation.",
            "None",
        )
    if mode == "dataset_matched":
        return (
            "Algorithm rollup (dataset-matched)",
            "Mean grouped by comparison and inferred algorithm variant after checkpoint-scope filtering.",
            "Arithmetic mean",
        )
    if mode == "shared_checkpoint":
        return (
            "Checkpoint rollup",
            "Mean grouped by comparison and inferred algorithm variant for one exact checkpoint.",
            "Arithmetic mean",
        )
    raise ValueError(f"unknown smart table mode: {mode}")


def _metric_membership(
    frame: pd.DataFrame,
    metric: str,
    mode: str,
    checkpoint_id: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """Decide metric eligibility once for both tables and provenance."""
    positions = pd.RangeIndex(len(frame))
    numeric = pd.Series(metric_numeric(frame, metric).to_numpy(), index=positions)
    reasons = pd.Series("included", index=positions, dtype=object)
    if mode == "dataset_matched":
        scopes = pd.Series(
            frame.get(
                "checkpoint_scope", pd.Series(index=frame.index, dtype=object)
            ).to_numpy(),
            index=positions,
        )
        reasons.loc[scopes == "cross_dataset"] = "cross_dataset_checkpoint"
        reasons.loc[scopes == "unresolved"] = "unresolved_checkpoint"
        reasons.loc[
            ~scopes.isin({"none", "matched", "cross_dataset", "unresolved"})
        ] = "ineligible_checkpoint_scope"
    elif mode == "shared_checkpoint":
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            reasons[:] = "checkpoint_not_selected"
        else:
            checkpoint_ids = pd.Series(
                frame.get(
                    "checkpoint_id", pd.Series(index=frame.index, dtype=object)
                ).to_numpy(),
                index=positions,
            )
            reasons.loc[~checkpoint_ids.eq(checkpoint_id).fillna(False)] = (
                "checkpoint_mismatch"
            )
    reasons.loc[reasons.eq("included") & numeric.isna()] = (
        "metric_missing_or_non_numeric"
    )
    included = reasons.eq("included")

    provenance = frame.reindex(columns=PROVENANCE_COLUMNS[3:-2]).reset_index(drop=True)
    provenance["included"] = included.to_numpy()
    provenance["status"] = included.map(
        {True: "included", False: "excluded"}
    ).to_numpy()
    provenance["reason"] = reasons.to_numpy()
    provenance["metric"] = metric
    provenance["metric_value"] = numeric.to_numpy()
    provenance = provenance[PROVENANCE_COLUMNS]

    included_positions = positions[included]
    eligible = frame.iloc[included_positions].copy()
    if len(included_positions):
        eligible[metric] = numeric.loc[included].to_numpy()
    excluded_cross_dataset = int(
        (provenance["reason"] == "cross_dataset_checkpoint").sum()
    )
    excluded_unresolved = int(
        (provenance["reason"] == "unresolved_checkpoint").sum()
    )
    return eligible, provenance, excluded_cross_dataset, excluded_unresolved


def _exact_metric_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Comparison": _display_values(
                frame, "comparison_label", "comparison_id"
            ),
            "Method": frame["method_label"],
            "Run label": frame["run_label"],
            "Checkpoint": frame["checkpoint_label"],
            "Value": frame[metric],
            "Result path": frame["source_path"],
        }
    )
    return table.sort_values(
        ["Comparison", "Method", "Run label", "Result path"], kind="stable"
    ).reset_index(drop=True)


def build_metric_table(
    frame: pd.DataFrame,
    metric: str,
    mode: str = "exact_runs",
    checkpoint_id: str | None = None,
) -> SmartTableResult:
    if mode not in {
        "exact_runs",
        "dataset_matched",
        "shared_checkpoint",
    }:
        raise ValueError(f"unknown smart table mode: {mode}")

    eligible, provenance, excluded_cross_dataset, excluded_unresolved = (
        _metric_membership(frame, metric, mode, checkpoint_id)
    )
    included_rows = len(eligible)
    if eligible.empty:
        return _empty_smart_table(
            mode=mode,
            provenance=provenance,
            excluded_cross_dataset=excluded_cross_dataset,
            excluded_unresolved=excluded_unresolved,
        )

    if mode == "exact_runs":
        table = _exact_metric_table(eligible, metric)
    else:
        selected = eligible[
            [
                "comparison_id",
                "comparison_label",
                "algorithm_variant_id",
                "algorithm_variant_label",
                metric,
            ]
        ].copy()
        selected["Comparison"] = _display_values(
            selected, "comparison_label", "comparison_id"
        )
        selected["variant"] = _display_values(
            selected, "algorithm_variant_label", "algorithm_variant_id"
        )
        means = (
            selected.groupby(
                ["comparison_id", "Comparison", "algorithm_variant_id", "variant"],
                dropna=False,
                sort=True,
            )[metric]
            .agg(_stable_mean)
            .reset_index()
        )
        pivot = means.pivot(index="Comparison", columns="variant", values=metric)
        table = _ordinary_table(pivot.sort_index())

    included_checkpoint_values = provenance.loc[
        provenance["included"], "checkpoint_id"
    ].dropna()
    included_checkpoints = tuple(
        sorted(
            {
                str(value)
                for value in included_checkpoint_values
                if isinstance(value, str) and value
            }
        )
    )
    aggregation_label, grouping_description, estimator = _aggregation_metadata(mode)
    return SmartTableResult(
        table=table,
        included_rows=included_rows,
        excluded_cross_dataset=excluded_cross_dataset,
        excluded_unresolved=excluded_unresolved,
        provenance=provenance,
        aggregation_label=aggregation_label,
        grouping_description=grouping_description,
        included_checkpoints=included_checkpoints,
        estimator=estimator,
    )


def frame_to_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    columns = [cell(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def frame_to_latex(frame: pd.DataFrame) -> str:
    return frame.to_latex(index=False)
