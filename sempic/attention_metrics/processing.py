"""Turn query-pass reducer statistics into flat, sample-weighted records."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from .profile_identity import fingerprint, strict_dict
from .profiles import validate_partition
from .regions import canonical_ratio, region_masks


_PROCESSING_FIELDS = {
    "position_mode",
    "num_position_bins",
    "edge_ratios",
}
_ATTENTION_SPECS = {
    "attention_profile": {
        "label": "Full attention profile",
        "value_label": "Attention probability",
        "axis_policy": "nonnegative_auto",
    },
    "attention_absolute_deviation": {
        "label": "Attention absolute deviation",
        "value_label": "Absolute deviation",
        "axis_policy": "nonnegative_auto",
    },
}
_RETRIEVAL_SPECS = {
    "retrieval_nrmse": {
        "label": "Retrieval NRMSE",
        "value_label": "NRMSE",
        "axis_policy": "nonnegative_auto",
    },
    "retrieval_cosine_distance": {
        "label": "Retrieval cosine distance",
        "value_label": "Cosine distance",
        "axis_policy": "nonnegative_auto",
    },
    "attention_mass_error": {
        "label": "Attention mass error",
        "value_label": "Absolute mass error",
        "axis_policy": "nonnegative_auto",
    },
}


def normalize_processing_config(value: object) -> dict[str, object]:
    config = strict_dict(value, _PROCESSING_FIELDS, "processing_config")
    position_mode = config["position_mode"]
    if position_mode not in ("absolute", "normalized", "auto"):
        raise ValueError("position_mode must be absolute, normalized, or auto.")
    num_bins = config["num_position_bins"]
    if not isinstance(num_bins, int) or isinstance(num_bins, bool) or num_bins <= 0:
        raise ValueError("num_position_bins must be positive.")
    ratios = config["edge_ratios"]
    if not isinstance(ratios, list) or not ratios:
        raise ValueError("edge_ratios must be a non-empty list.")
    normalized_ratios = [canonical_ratio(ratio) for ratio in ratios]
    if len(set(normalized_ratios)) != len(normalized_ratios):
        raise ValueError("edge_ratios must be unique.")
    return {
        "position_mode": position_mode,
        "num_position_bins": num_bins,
        "edge_ratios": normalized_ratios,
    }


def process_partitions(
    partitions: Sequence[dict[str, object]], processing_config: object
) -> dict[str, object]:
    """Process one or more model/dataset/query-pass statistics partitions."""

    if not partitions:
        raise ValueError("At least one statistics partition is required.")
    config = normalize_processing_config(processing_config)
    validated = [validate_partition(partition) for partition in partitions]
    identities = [
        (
            partition["partition_identity"]["model_id"],
            partition["partition_identity"]["dataset_id"],
            partition["partition_identity"]["query_pass_id"],
        )
        for partition in validated
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Processed partitions contain duplicate identities.")

    records = []
    for partition in validated:
        records.extend(_process_partition(partition, config))
    metric_specs = {
        key: spec
        for key, spec in {**_ATTENTION_SPECS, **_RETRIEVAL_SPECS}.items()
        if any(record["metric_key"] == key for record in records)
    }
    return {
        "processing_config": config,
        "processing_fingerprint": fingerprint(config),
        "source_partitions": [
            {
                "partition_fingerprint": partition["partition_fingerprint"],
                "model_id": partition["partition_identity"]["model_id"],
                "dataset_id": partition["partition_identity"]["dataset_id"],
                "query_pass_id": partition["partition_identity"]["query_pass_id"],
            }
            for partition in validated
        ],
        "metric_specs": metric_specs,
        "records": records,
    }


def _process_partition(
    partition: dict[str, object], config: dict[str, object]
) -> list[dict[str, object]]:
    identity = partition["partition_identity"]
    base = {
        "model_id": identity["model_id"],
        "dataset_id": identity["dataset_id"],
        "query_pass_id": identity["query_pass_id"],
    }
    methods = [method["method_key"] for method in identity["methods"]]
    candidate_methods = methods[1:]
    reducers = set(identity["query_spec"]["reducers"])
    lengths = [
        chunk["token_length"]
        for sample in partition["samples"]
        for chunk in sample["chunks"]
    ]
    mode = config["position_mode"]
    if mode == "auto":
        mode = "absolute" if len(set(lengths)) == 1 else "normalized"
    if mode == "absolute" and len(set(lengths)) != 1:
        raise ValueError("absolute position mode requires equal chunk lengths.")
    num_bins = lengths[0] if mode == "absolute" else config["num_position_bins"]
    position_coordinates = (
        list(range(num_bins))
        if mode == "absolute"
        else [float((index + 0.5) / num_bins) for index in range(num_bins)]
    )
    layer_coordinates = list(range(partition["layer_count"]))
    head_coordinates = list(range(partition["query_head_count"]))
    records: list[dict[str, object]] = []

    attention_fields = {
        "full_recompute": (
            "attention_profile",
            (("raw", "raw"), ("chunk_conditional", "chunk_conditional")),
        )
    }
    for method in candidate_methods:
        attention_fields[method] = (
            "attention_absolute_deviation",
            (
                ("raw", "raw_absolute_error"),
                ("chunk_conditional", "chunk_conditional_absolute_error"),
            ),
        )
    if "attention_profile" in reducers:
        for method, (metric_key, views) in attention_fields.items():
          for attention_view, field in views:
            chunk_values = _attention_chunks(
                partition["samples"], method, field, num_bins, mode
            )
            records.append(_record(
                base,
                metric_key,
                "layer_position_heatmap",
                method,
                {"attention_view": attention_view, "position_mode": mode},
                ["layer", "position_bin"],
                {"layer": layer_coordinates, "position_bin": position_coordinates},
                [_equal_chunks(chunks) for chunks in chunk_values],
            ))
            records.append(_record(
                base,
                metric_key,
                "global_bar",
                method,
                {"attention_view": attention_view},
                [],
                {},
                [_equal_chunks([_nanmean(chunk) for chunk in chunks]) for chunks in chunk_values],
            ))
            if method != "full_recompute":
                records.extend(_attention_region_records(
                    base=base,
                    samples=partition["samples"],
                    method=method,
                    field=field,
                    attention_view=attention_view,
                    ratios=config["edge_ratios"],
                    layer_coordinates=layer_coordinates,
                ))

    retrieval_metrics: tuple[
        tuple[str, Callable[[dict[str, object]], torch.Tensor]], ...
    ] = (
        ("retrieval_nrmse", _retrieval_nrmse),
        ("retrieval_cosine_distance", _retrieval_cosine_distance),
        ("attention_mass_error", _attention_mass_error),
    )
    if "pic_retrieval" in reducers:
        for method in candidate_methods:
          for metric_key, calculation in retrieval_metrics:
            chunks_by_sample = [
                [
                    calculation(chunk["reducer_outputs"]["pic_retrieval"][method])
                    for chunk in sample["chunks"]
                ]
                for sample in partition["samples"]
            ]
            records.extend((
                _record(
                    base, metric_key, "layer_head_heatmap", method, {},
                    ["layer", "query_head"],
                    {"layer": layer_coordinates, "query_head": head_coordinates},
                    [_equal_chunks(chunks) for chunks in chunks_by_sample],
                ),
                _record(
                    base, metric_key, "layer_curve", method, {}, ["layer"],
                    {"layer": layer_coordinates},
                    [_equal_chunks([_nanmean(chunk, dim=1) for chunk in chunks])
                     for chunks in chunks_by_sample],
                ),
                _record(
                    base, metric_key, "global_bar", method, {}, [], {},
                    [_equal_chunks([_nanmean(chunk) for chunk in chunks])
                     for chunks in chunks_by_sample],
                ),
            ))
    return records


def _attention_chunks(samples, method, field, num_bins, mode):
    result = []
    for sample in samples:
        sample_chunks = []
        for chunk in sample["chunks"]:
            values = chunk["reducer_outputs"]["attention_profile"][method][field].to(
                torch.float64
            )
            length = chunk["token_length"]
            bins = (
                np.arange(length, dtype=np.int64)
                if mode == "absolute"
                else np.minimum(
                    (((np.arange(length) + 0.5) / length) * num_bins).astype(np.int64),
                    num_bins - 1,
                )
            )
            binned = torch.full(
                (values.size(0), num_bins), torch.nan, dtype=torch.float64
            )
            for bin_index in np.unique(bins):
                binned[:, int(bin_index)] = values[:, bins == bin_index].mean(dim=1)
            sample_chunks.append(binned)
        result.append(sample_chunks)
    return result


def _attention_region_records(
    *, base, samples, method, field, attention_view, ratios, layer_coordinates
):
    records = []
    for ratio_text in ratios:
        ratio = float(ratio_text)
        for region in ("prefix", "interior", "suffix"):
            sample_curves = []
            sample_globals = []
            mask_key = "middle" if region == "interior" else region
            for sample in samples:
                chunk_curves = []
                for chunk in sample["chunks"]:
                    values = chunk["reducer_outputs"]["attention_profile"][method][field].to(
                        torch.float64
                    )
                    mask = region_masks(chunk["token_length"], ratio)[mask_key]
                    chunk_curves.append(
                        values[:, mask].mean(dim=1)
                        if mask.any()
                        else torch.full((values.size(0),), torch.nan, dtype=torch.float64)
                    )
                sample_curves.append(_equal_chunks(chunk_curves))
                sample_globals.append(_equal_chunks([_nanmean(item) for item in chunk_curves]))
            facets = {
                "attention_view": attention_view,
                "edge_ratio": ratio_text,
                "region": region,
            }
            records.extend((
                _record(
                    base, "attention_absolute_deviation", "layer_curve", method,
                    facets, ["layer"], {"layer": layer_coordinates}, sample_curves,
                ),
                _record(
                    base, "attention_absolute_deviation", "global_bar", method,
                    facets, [], {}, sample_globals,
                ),
            ))
    return records


def _retrieval_nrmse(statistics: dict[str, object]) -> torch.Tensor:
    squared_error = statistics["squared_error_sum"].to(torch.float64)
    reference_energy = statistics["reference_energy_sum"].to(torch.float64)
    return torch.where(
        reference_energy > 0,
        torch.sqrt(squared_error / reference_energy.clamp_min(torch.finfo(torch.float64).tiny)),
        torch.nan,
    )


def _retrieval_cosine_distance(statistics: dict[str, object]) -> torch.Tensor:
    total = statistics["cosine_distance_sum"].to(torch.float64)
    count = statistics["cosine_valid_count"].to(torch.float64)
    return torch.where(count > 0, total / count.clamp_min(1), torch.nan)


def _attention_mass_error(statistics: dict[str, object]) -> torch.Tensor:
    return statistics["absolute_mass_error_sum"].to(torch.float64) / statistics[
        "query_count"
    ]


def _record(base, metric, view, method, facets, axes, coordinates, samples):
    return {
        **base,
        "metric_key": metric,
        "view_key": view,
        "method_key": method,
        "facets": dict(facets),
        "axes": list(axes),
        "coordinates": dict(coordinates),
        **_estimate(samples),
    }


def _equal_chunks(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return _nanmean(torch.stack([value.to(torch.float64) for value in values]), dim=0)


def _nanmean(values: torch.Tensor, dim: int | tuple[int, ...] | None = None):
    valid = torch.isfinite(values)
    count = valid.sum(dim=dim)
    total = torch.where(valid, values, 0.0).sum(dim=dim)
    return torch.where(count > 0, total / count.clamp_min(1), torch.nan)


def _estimate(values: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = torch.stack([value.to(torch.float64) for value in values])
    valid = torch.isfinite(stacked)
    count = valid.sum(dim=0).to(torch.int64)
    total = torch.where(valid, stacked, 0.0).sum(dim=0)
    mean = torch.where(count > 0, total / count.clamp_min(1), torch.nan)
    centered = torch.where(valid, stacked - mean, 0.0)
    variance = torch.where(
        count > 1,
        centered.square().sum(dim=0) / (count - 1).clamp_min(1),
        torch.zeros_like(mean),
    )
    sem = torch.where(count > 0, torch.sqrt(variance / count.clamp_min(1)), torch.nan)
    return {"mean": mean, "sem": sem, "count": count}


__all__ = [
    "normalize_processing_config",
    "process_partitions",
]
