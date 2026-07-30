import argparse
from collections.abc import Sequence

from sempic.eval_dashboard.app import render_dashboard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse and compare sempic evaluation results.",
    )
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Server-local directory to scan recursively. May be repeated.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    render_dashboard(args.roots or ["eval_config"])


if __name__ == "__main__":
    main()
