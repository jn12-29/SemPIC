"""Generate one immutable teacher generation-cache artifact."""

import argparse
from pathlib import Path
from typing import Sequence

from sempic.utils.config import load_config_file
from sempic.utils.generation_cache_run import (
    generate_cache_artifact,
    load_generation_cache_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one teacher generation-cache artifact from a JSON config."
    )
    parser.add_argument("config_file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_config = load_config_file(
        str(args.config_file),
        default_config_file="_default.json",
    )
    config = load_generation_cache_config(raw_config)
    generate_cache_artifact(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
