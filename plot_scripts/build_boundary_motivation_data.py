"""Build the provenance-linked Boundary Motivation evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from plot_scripts.boundary_motivation_data import (
        build_motivation_data,
        materialize_pinned_configs,
        record_bundle_files,
        write_bundle,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from boundary_motivation_data import (
        build_motivation_data,
        materialize_pinned_configs,
        record_bundle_files,
        write_bundle,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the fixed eight-point Boundary Motivation evidence matrix."
    )
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("MANIFEST", "SUMMARY"),
        required=True,
        help="Frozen attention run manifest and its authoritative readable summary.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Candidate result JSON or directory. Repeat for multiple inputs.",
    )
    parser.add_argument("--repo-root", default=".", help="Root used to normalize path spellings.")
    parser.add_argument("--output-dir", required=True, help="Stable evidence bundle directory.")
    parser.add_argument(
        "--pinned-config-dir",
        required=True,
        help="Directory for the 16 frozen Full/No-Recompute configs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only a previously marked generated bundle; never arbitrary files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = build_motivation_data(
        sources=[tuple(source) for source in args.source],
        candidate_paths=args.result,
        repo_root=args.repo_root,
    )
    outputs = write_bundle(data, args.output_dir, overwrite=args.overwrite)
    pinned = materialize_pinned_configs(data["points"], args.pinned_config_dir)
    bundle_dir = Path(args.output_dir).resolve()
    pinned_dir = Path(args.pinned_config_dir).resolve()
    if pinned_dir.is_relative_to(bundle_dir):
        record_bundle_files(bundle_dir, pinned.values())
    print(
        json.dumps(
            {
                "point_count": len(data["plot_rows"]),
                "complete_point_count": sum(
                    row["status"] == "pass" for row in data["plot_rows"]
                ),
                "pinned_config_count": len(pinned),
                "outputs": {key: str(path) for key, path in outputs.items()},
                "pinned_config_dir": str(Path(args.pinned_config_dir).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
