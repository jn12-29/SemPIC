import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

try:
    from plot_scripts.main_results_data import (
        REQUIRED_METRICS,
        SCHEMA_VERSION,
        suffix_output_path,
        validate_plot_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from main_results_data import (
        REQUIRED_METRICS,
        SCHEMA_VERSION,
        suffix_output_path,
        validate_plot_data,
    )
from sempic.utils.run_storage import allocate_run_dir, atomic_write_json
from sempic.utils.runtime import RuntimeContext


LOGGER = logging.getLogger("sempic.plot_scripts.build_main_results_data")
DEFAULT_OUTPUT_FILE = "plot_data/main_results.json"


def _load_result(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level value must be an object")
    return payload


def _required_mapping(parent: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be an object")
    return value


def _required_name(parent: dict[str, Any], key: str, location: str, path: Path) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: {location}.{key} must be a non-empty string")
    return value


def _required_metric(result: dict[str, Any], metric: str, path: Path) -> float:
    value = result.get(metric)
    location = f"{path}: result.{metric}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{location} must be finite")
    return float(value)


def discover_result_files(result_paths: list[str | Path]) -> list[Path]:
    if not result_paths:
        raise ValueError("At least one result file or directory is required")

    discovered: list[Path] = []
    seen_paths: set[Path] = set()
    for result_path in result_paths:
        path = Path(result_path)
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for candidate in path.rglob("*_result.json")
                if candidate.is_file()
            )
        else:
            raise ValueError(f"Result path does not exist or is not accessible: {path}")

        for candidate in candidates:
            resolved_candidate = candidate.resolve()
            if resolved_candidate not in seen_paths:
                seen_paths.add(resolved_candidate)
                discovered.append(candidate)

    if not discovered:
        raise ValueError("No result files found")
    return discovered


def build_plot_data(result_paths: list[str | Path]) -> dict[str, Any]:
    result_files = discover_result_files(result_paths)

    datasets: list[dict[str, Any]] = []
    datasets_by_name: dict[str, dict[str, Any]] = {}
    series_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}

    for result_file in result_files:
        path = Path(result_file)
        resolved_path = path.resolve()

        payload = _load_result(path)
        config = _required_mapping(payload, "config", path)
        dataset_config = _required_mapping(config, "dataset", path)
        cache_comb = _required_mapping(config, "cache_comb", path)
        result = _required_mapping(payload, "result", path)

        dataset_name = _required_name(
            dataset_config, "dataset_name", "config.dataset", path
        )
        series_name = _required_name(cache_comb, "method", "config.cache_comb", path)

        run_suffix = config.get("run_suffix")
        if run_suffix is not None and not isinstance(run_suffix, str):
            raise ValueError(f"{path}: config.run_suffix must be a string or null")
        label = run_suffix.strip() if isinstance(run_suffix, str) else ""
        if not label:
            label = path.stem.removesuffix("_result")

        point: dict[str, Any] = {
            "label": label,
            **{metric: _required_metric(result, metric, path) for metric in REQUIRED_METRICS},
            "source_file": str(resolved_path),
        }

        dataset = datasets_by_name.get(dataset_name)
        if dataset is None:
            dataset = {"name": dataset_name, "series": []}
            datasets_by_name[dataset_name] = dataset
            series_by_dataset[dataset_name] = {}
            datasets.append(dataset)

        series = series_by_dataset[dataset_name].get(series_name)
        if series is None:
            series = {"name": series_name, "points": []}
            series_by_dataset[dataset_name][series_name] = series
            dataset["series"].append(series)
        series["points"].append(point)

    data = {"schema_version": SCHEMA_VERSION, "datasets": datasets}
    return validate_plot_data(data)


def write_plot_data(data: dict[str, Any], output_file: str | Path) -> None:
    validate_plot_data(data)
    atomic_write_json(output_file, data)


def build_and_write_plot_data(
    result_paths: list[str | Path], output_file: str | Path
) -> dict[str, Any]:
    result_files = discover_result_files(result_paths)
    output_path = Path(output_file).resolve()
    if output_path in {path.resolve() for path in result_files}:
        raise ValueError("Output file must not overwrite an input result file")
    data = build_plot_data(result_files)
    write_plot_data(data, output_file)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build editable main-results plot data from result JSON files or directories."
    )
    parser.add_argument(
        "result_paths",
        nargs="+",
        help="Result JSON files or directories; directories are searched recursively for *_result.json.",
    )
    parser.add_argument(
        "--output-file", default=DEFAULT_OUTPUT_FILE, help="Output plot-data JSON file."
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional suffix inserted before the output file extension.",
    )
    args = parser.parse_args()

    output_file = suffix_output_path(args.output_file, args.run_suffix)
    run_dir = allocate_run_dir("./logs/build_main_results_data", args.run_suffix)
    with RuntimeContext(
        entrypoint="build_main_results_data",
        run_dir=run_dir,
        config_file=None,
        resolved_config=None,
        config_snapshot_name=None,
        cli_args=vars(args).copy(),
    ):
        data = build_and_write_plot_data(args.result_paths, output_file)
        point_count = sum(
            len(series["points"])
            for dataset in data["datasets"]
            for series in dataset["series"]
        )
        LOGGER.info("Wrote %d points to %s", point_count, output_file)


if __name__ == "__main__":
    main()
