"""Build full100 SemPIC attention density curves at block-local token offsets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
from pathlib import Path
import statistics
from typing import Any, Sequence

import torch

try:
    from plot_scripts.attention_sink_data import _interval_layer_density
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from attention_sink_data import _interval_layer_density

from sempic.attention_metrics.profile_storage import load_partition
from sempic.utils.run_storage import atomic_write_json


SCHEMA_NAME = "sempic.block_token_sink_curve"
SCHEMA_VERSION = 1
FULL_SAMPLE_COUNT = 100
DATASET_ORDER = ("biography", "hotpot_qa", "musique", "niah")
MODEL_SPECS = (
    ("qwen3_4b", "Qwen3-4B-Instruct-2507", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B", "Qwen3-8B"),
    ("llama3_1_8b", "Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
)
EXPECTED_IDENTITIES = tuple(
    (model_key, dataset_id)
    for model_key, _, _ in MODEL_SPECS
    for dataset_id in DATASET_ORDER
)
INTERIOR_START = 0.1
INTERIOR_END = 0.9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_partitions(values: Sequence[str]) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    expected = set(EXPECTED_IDENTITIES)
    for value in values:
        try:
            raw_identity, raw_path = value.split("=", 1)
            model_key, dataset_id = raw_identity.split("/", 1)
        except ValueError as error:
            raise ValueError(
                f"--partition must use MODEL_ID/DATASET_ID=PATH: {value}"
            ) from error
        identity = (model_key, dataset_id)
        if identity not in expected:
            raise ValueError(f"Unknown partition identity: {raw_identity}")
        if identity in paths:
            raise ValueError(f"Duplicate partition identity: {raw_identity}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Partition path is not a file: {path}")
        paths[identity] = path
    missing = [f"{model}/{dataset}" for model, dataset in EXPECTED_IDENTITIES if (model, dataset) not in paths]
    if missing:
        raise ValueError(
            "--partition must cover the fixed 12 model/dataset identities; missing: "
            + ", ".join(missing)
        )
    return paths


def _validate_full100_partition(
    partition: dict[str, Any],
    *,
    model_id: str,
    dataset_id: str,
) -> list[dict[str, Any]]:
    identity = partition["partition_identity"]
    if (
        identity.get("model_id") != model_id
        or identity.get("dataset_id") != dataset_id
        or identity.get("query_pass_id") != "shifted_prediction"
    ):
        raise ValueError(
            f"Partition identity must be {model_id}/{dataset_id}/shifted_prediction."
        )
    reducers = identity.get("query_spec", {}).get("reducers", [])
    if "raw_attention_profile" not in reducers:
        raise ValueError(f"Partition {model_id}/{dataset_id} lacks raw_attention_profile.")
    methods = tuple(record.get("method_key") for record in identity.get("methods", []))
    if "sempic" not in methods:
        raise ValueError(f"Partition {model_id}/{dataset_id} lacks method=sempic.")

    dataset_config = identity.get("dataset_config")
    configured_count = (
        dataset_config.get("num_samples") if isinstance(dataset_config, dict) else None
    )
    max_samples = identity.get("max_samples")
    if type(configured_count) is not int or configured_count <= 0:
        raise ValueError("Partition dataset_config.num_samples must be positive.")
    if max_samples is not None and (type(max_samples) is not int or max_samples <= 0):
        raise ValueError("Partition max_samples must be null or positive.")
    expected_count = (
        configured_count if max_samples is None else min(configured_count, max_samples)
    )
    samples = partition["samples"]
    if expected_count != FULL_SAMPLE_COUNT or len(samples) != FULL_SAMPLE_COUNT:
        raise ValueError(
            f"Partition {model_id}/{dataset_id} must be full100; "
            f"configured expected={expected_count}, actual={len(samples)}."
        )
    return samples


def _aggregate_point(
    path: Path,
    *,
    model_id: str,
    dataset_id: str,
    max_offset: int,
) -> dict[str, Any]:
    partition = load_partition(path)
    samples = _validate_full100_partition(
        partition, model_id=model_id, dataset_id=dataset_id
    )
    sample_offsets: list[list[float]] = []
    sample_interiors: list[float] = []
    block_lengths: list[int] = []
    chunk_count = 0

    for sample in samples:
        chunk_offsets: list[torch.Tensor] = []
        chunk_interiors: list[float] = []
        for chunk in sample["chunks"]:
            token_length = int(chunk["token_length"])
            if token_length < max_offset:
                raise ValueError(
                    f"Block {model_id}/{dataset_id}/{sample['sample_id']}/"
                    f"{chunk['chunk_id']} has length {token_length}, below "
                    f"--max-offset={max_offset}."
                )
            profile = chunk["reducer_outputs"]["raw_attention_profile"]["sempic"][
                "raw"
            ]
            chunk_offsets.append(profile[:, :max_offset].double().mean(dim=0))
            chunk_interiors.append(
                float(
                    _interval_layer_density(
                        profile, INTERIOR_START, INTERIOR_END
                    ).mean()
                )
            )
            block_lengths.append(token_length)
            chunk_count += 1
        sample_offsets.append(
            [float(value) for value in torch.stack(chunk_offsets).mean(dim=0)]
        )
        sample_interiors.append(statistics.fmean(chunk_interiors))

    token_densities = [
        statistics.fmean(sample[offset] for sample in sample_offsets)
        for offset in range(max_offset)
    ]
    interior_density = statistics.fmean(sample_interiors)
    if not math.isfinite(interior_density) or interior_density <= 0:
        raise ValueError(
            f"Partition {model_id}/{dataset_id} has nonpositive interior density."
        )
    if any(not math.isfinite(value) or value < 0 for value in token_densities):
        raise ValueError(f"Partition {model_id}/{dataset_id} has invalid token density.")

    return {
        "dataset_id": dataset_id,
        "sample_count": len(samples),
        "chunk_count": chunk_count,
        "block_token_length": {
            "min": min(block_lengths),
            "median": statistics.median(block_lengths),
            "max": max(block_lengths),
        },
        "source": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        },
        "offsets": [
            {
                "token_offset": offset,
                "token_density": token_density,
                "interior_density": interior_density,
                "ratio": token_density / interior_density,
            }
            for offset, token_density in enumerate(token_densities, start=1)
        ],
    }


def build_curve_data(
    partition_paths: dict[tuple[str, str], Path], *, max_offset: int
) -> dict[str, Any]:
    if (
        isinstance(max_offset, bool)
        or not isinstance(max_offset, int)
        or max_offset <= 0
    ):
        raise ValueError("max_offset must be a positive integer.")
    if set(partition_paths) != set(EXPECTED_IDENTITIES):
        raise ValueError("partition_paths must contain exactly the fixed 12 identities.")

    models = []
    for model_key, model_id, display_name in MODEL_SPECS:
        points = []
        for dataset_id in DATASET_ORDER:
            points.append(
                _aggregate_point(
                    partition_paths[(model_key, dataset_id)],
                    model_id=model_id,
                    dataset_id=dataset_id,
                    max_offset=max_offset,
                )
            )
            gc.collect()
        models.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "points": points,
            }
        )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "max_token_offset": max_offset,
        "estimand": {
            "method": "sempic",
            "query_pass_id": "shifted_prediction",
            "coordinate": (
                "One-based canonical reusable-block token offset; max_token_offset "
                "is the inclusive maximum offset and wrapper filler is excluded."
            ),
            "interior": [INTERIOR_START, INTERIOR_END],
            "aggregation": (
                "Query heads and query rows are equally averaged by the raw reducer; "
                "layers are averaged within each chunk for the token offset and matched "
                "interior separately; chunks are equally averaged within each sample; "
                "samples are equally averaged; ratio is aggregate token-density mean "
                "divided by aggregate interior-density mean."
            ),
        },
        "models": models,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        action="append",
        default=[],
        metavar="MODEL_ID/DATASET_ID=PATH",
        help=(
            "Full100 shifted-prediction partition; repeat for every fixed model/dataset "
            "identity."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-offset",
        type=int,
        default=8,
        help="Inclusive maximum one-based block-local token offset (default: 8).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = _parse_partitions(args.partition)
    output_path = args.output.resolve()
    if any(output_path == path.resolve() for path in paths.values()):
        raise ValueError("--output must not overwrite an input partition.")
    data = build_curve_data(paths, max_offset=args.max_offset)
    atomic_write_json(args.output, data)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
