"""Run real attention statistics, processing, and visualization."""

import argparse
from pathlib import Path

import torch

from sempic.attention_metrics.analysis_pipeline import load_processing_config
from sempic.attention_metrics.attention_run import (
    create_attention_run,
    resume_attention_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect, process, and visualize real attention profiles."
    )
    parser.add_argument("config_files", nargs="*")
    parser.add_argument("--analysis-config", type=Path)
    parser.add_argument(
        "--processing-config",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if args.resume_run is not None:
        if (
            args.config_files
            or args.analysis_config is not None
            or args.processing_config is not None
            or args.output_dir is not None
            or args.run_name is not None
            or args.max_samples is not None
        ):
            raise ValueError(
                "--resume-run cannot override configs, paths, run name, or max samples."
            )
        run_dir = resume_attention_run(args.resume_run)
    else:
        if not args.config_files or args.analysis_config is None:
            raise ValueError("New runs require config files and --analysis-config.")
        processing_config = args.processing_config or Path(
            "attention_config/processing_default.json"
        )
        run_dir = create_attention_run(
            args.config_files,
            analysis_config_path=args.analysis_config,
            processing_config_path=processing_config,
            processing_config=load_processing_config(processing_config),
            output_dir=args.output_dir or Path("attention_results"),
            run_name=args.run_name,
            max_samples=args.max_samples,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
