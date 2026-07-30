"""Render attention figures from processed metrics only."""

import argparse
from pathlib import Path

from sempic.attention_metrics.analysis_pipeline import plot_attention_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render figures for one processed attention analysis variant."
    )
    parser.add_argument("variant_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_attention_variant(args.variant_dir)


if __name__ == "__main__":
    main()
