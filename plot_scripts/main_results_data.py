from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from sempic.utils.run_storage import validate_run_suffix


SCHEMA_VERSION = 1
MODEL_SCHEMA_VERSION = 2
REQUIRED_METRICS = ("f1", "ttft", "flops")


def suffix_output_path(output_file: str | Path, run_suffix: str | None) -> Path:
    validate_run_suffix(run_suffix)
    path = Path(output_file)
    output_path = (
        path
        if run_suffix is None
        else path.with_name(f"{path.stem}_{run_suffix}{path.suffix}")
    )
    if output_path.is_dir():
        raise ValueError(
            f"output_file must be a file path, not a directory: {output_path}"
        )
    return output_path


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _require_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _require_finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{location} must be finite")
    return float(value)


def _validate_datasets(datasets: Any, location: str) -> list[dict[str, Any]]:
    if not isinstance(datasets, list):
        raise ValueError(f"{location} must be a list")
    if not datasets:
        raise ValueError(f"{location} must contain at least one dataset")

    dataset_names: set[str] = set()
    for dataset_index, dataset_value in enumerate(datasets):
        dataset_location = f"{location}[{dataset_index}]"
        dataset = _require_mapping(dataset_value, dataset_location)
        dataset_name = _require_nonempty_string(
            dataset.get("name"), f"{dataset_location}.name"
        )
        if dataset_name in dataset_names:
            raise ValueError(
                f"{dataset_location}.name duplicates dataset name: {dataset_name}"
            )
        dataset_names.add(dataset_name)
        series_values = dataset.get("series")
        if not isinstance(series_values, list):
            raise ValueError(f"{dataset_location}.series must be a list")
        if not series_values:
            raise ValueError(f"{dataset_location}.series must contain at least one series")

        series_names: set[str] = set()
        for series_index, series_value in enumerate(series_values):
            series_location = f"{dataset_location}.series[{series_index}]"
            series = _require_mapping(series_value, series_location)
            series_name = _require_nonempty_string(
                series.get("name"), f"{series_location}.name"
            )
            if series_name in series_names:
                raise ValueError(
                    f"{series_location}.name duplicates series name in dataset "
                    f"{dataset_name}: {series_name}"
                )
            series_names.add(series_name)
            points = series.get("points")
            if not isinstance(points, list):
                raise ValueError(f"{series_location}.points must be a list")
            if not points:
                raise ValueError(f"{series_location}.points must contain at least one point")

            for optional_style in ("color", "marker"):
                if optional_style in series:
                    _require_nonempty_string(
                        series[optional_style], f"{series_location}.{optional_style}"
                    )

            for point_index, point_value in enumerate(points):
                point_location = f"{series_location}.points[{point_index}]"
                point = _require_mapping(point_value, point_location)
                for metric in REQUIRED_METRICS:
                    _require_finite_number(point.get(metric), f"{point_location}.{metric}")
                for optional_text in ("label", "source_file"):
                    if optional_text in point and not isinstance(point[optional_text], str):
                        raise ValueError(f"{point_location}.{optional_text} must be a string")
                if "annotate" in point and not isinstance(point["annotate"], bool):
                    raise ValueError(f"{point_location}.annotate must be a boolean")

    return datasets


def _require_schema_version(root: dict[str, Any], expected: int) -> None:
    schema_version = root.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != expected
    ):
        raise ValueError(f"schema_version must be {expected}")


def validate_plot_data(data: Any) -> dict[str, Any]:
    root = _require_mapping(data, "plot data")
    _require_schema_version(root, SCHEMA_VERSION)
    _validate_datasets(root.get("datasets"), "datasets")

    return root


def validate_model_plot_data(data: Any) -> dict[str, Any]:
    root = _require_mapping(data, "plot data")
    _require_schema_version(root, MODEL_SCHEMA_VERSION)
    models = root.get("models")
    if not isinstance(models, list):
        raise ValueError("models must be a list")
    if not models:
        raise ValueError("models must contain at least one model")

    model_names: set[str] = set()
    for model_index, model_value in enumerate(models):
        model_location = f"models[{model_index}]"
        model = _require_mapping(model_value, model_location)
        model_name = _require_nonempty_string(
            model.get("name"), f"{model_location}.name"
        )
        _require_nonempty_string(
            model.get("display_name"), f"{model_location}.display_name"
        )
        if model_name in model_names:
            raise ValueError(f"duplicate model name: {model_name}")
        model_names.add(model_name)
        if "metadata" in model:
            _require_mapping(model["metadata"], f"{model_location}.metadata")
        _validate_datasets(model.get("datasets"), f"{model_location}.datasets")

    return root


def load_plot_data(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        with input_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {input_path}: {error}") from error
    try:
        return validate_plot_data(data)
    except ValueError as error:
        raise ValueError(f"{input_path}: {error}") from error


def load_model_plot_data(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        with input_path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {input_path}: {error}") from error
    try:
        return validate_model_plot_data(data)
    except ValueError as error:
        raise ValueError(f"{input_path}: {error}") from error


def _overlay_dataset_lists(
    target_datasets: list[dict[str, Any]], source_datasets: list[dict[str, Any]]
) -> None:
    datasets_by_name = {dataset["name"]: dataset for dataset in target_datasets}
    for source_dataset in source_datasets:
        dataset_name = source_dataset["name"]
        target_dataset = datasets_by_name.get(dataset_name)
        if target_dataset is None:
            target_dataset = deepcopy(source_dataset)
            datasets_by_name[dataset_name] = target_dataset
            target_datasets.append(target_dataset)
            continue

        series_indexes = {
            series["name"]: index
            for index, series in enumerate(target_dataset["series"])
        }
        for source_series in source_dataset["series"]:
            series_name = source_series["name"]
            series_index = series_indexes.get(series_name)
            if series_index is None:
                series_indexes[series_name] = len(target_dataset["series"])
                target_dataset["series"].append(deepcopy(source_series))
            else:
                target_dataset["series"][series_index] = deepcopy(source_series)


def overlay_plot_data(plot_data_values: list[dict[str, Any]]) -> dict[str, Any]:
    if not plot_data_values:
        raise ValueError("At least one plot-data value is required")

    validated_values = [validate_plot_data(value) for value in plot_data_values]
    merged = deepcopy(validated_values[0])
    for data in validated_values[1:]:
        _overlay_dataset_lists(merged["datasets"], data["datasets"])
    return validate_plot_data(merged)


def overlay_model_plot_data(
    plot_data_values: list[dict[str, Any]],
) -> dict[str, Any]:
    if not plot_data_values:
        raise ValueError("At least one model plot-data value is required")

    validated_values = [validate_model_plot_data(value) for value in plot_data_values]
    merged = deepcopy(validated_values[0])
    models_by_name = {model["name"]: model for model in merged["models"]}

    for data in validated_values[1:]:
        for source_model in data["models"]:
            model_name = source_model["name"]
            target_model = models_by_name.get(model_name)
            if target_model is None:
                target_model = deepcopy(source_model)
                models_by_name[model_name] = target_model
                merged["models"].append(target_model)
                continue
            _overlay_dataset_lists(
                target_model["datasets"], source_model["datasets"]
            )

    return validate_model_plot_data(merged)


def load_plot_data_files(paths: list[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one plot-data file is required")
    return overlay_plot_data([load_plot_data(path) for path in paths])


def load_plot_document_files(paths: list[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one plot-data file is required")

    loaded: list[dict[str, Any]] = []
    versions: set[int] = set()
    for path in paths:
        input_path = Path(path)
        try:
            with input_path.open(encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {input_path}: {error}") from error
        root = _require_mapping(data, str(input_path))
        version = root.get("schema_version")
        if version == SCHEMA_VERSION and not isinstance(version, bool):
            validator = validate_plot_data
        elif version == MODEL_SCHEMA_VERSION and not isinstance(version, bool):
            validator = validate_model_plot_data
        else:
            raise ValueError(
                f"{input_path}: schema_version must be {SCHEMA_VERSION} or "
                f"{MODEL_SCHEMA_VERSION}"
            )
        try:
            loaded.append(validator(root))
        except ValueError as error:
            raise ValueError(f"{input_path}: {error}") from error
        versions.add(version)

    if len(versions) != 1:
        raise ValueError("Cannot mix schema_version 1 and schema_version 2 inputs")
    if versions == {SCHEMA_VERSION}:
        return overlay_plot_data(loaded)
    return overlay_model_plot_data(loaded)
