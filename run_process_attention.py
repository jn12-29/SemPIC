"""Process reusable attention profiles without loading a model."""

import argparse
from pathlib import Path

import torch

from sempic.attention_metrics.analysis_pipeline import (
    load_processing_config,
    process_attention_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a processing and visualization variant from an attention run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--processing-config",
        type=Path,
        default=Path("attention_config/processing_default.json"),
    )
    parser.add_argument(
        "--suffix",
        help=(
            "Manual variant suffix; it must use safe path characters and contain "
            "at least one non-digit."
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    variant_dir = process_attention_run(
        args.run_dir,
        processing_config=load_processing_config(args.processing_config),
        suffix=args.suffix,
    )
    print(variant_dir)


if __name__ == "__main__":
    main()
