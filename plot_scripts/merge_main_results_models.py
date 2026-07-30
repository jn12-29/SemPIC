import argparse
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

try:
    from plot_scripts.main_results_data import (
        MODEL_SCHEMA_VERSION,
        load_plot_data,
        overlay_plot_data,
        suffix_output_path,
        validate_model_plot_data,
    )
except ModuleNotFoundError as error:
    if error.name != "plot_scripts":
        raise
    from main_results_data import (
        MODEL_SCHEMA_VERSION,
        load_plot_data,
        overlay_plot_data,
        suffix_output_path,
        validate_model_plot_data,
    )
from sempic.utils.run_storage import allocate_run_dir, atomic_write_json
from sempic.utils.runtime import RuntimeContext


LOGGER = logging.getLogger("sempic.plot_scripts.merge_main_results_models")
DEFAULT_OUTPUT_FILE = "plot_data/main_results_models.json"


def merge_v1_model_files(
    model_inputs: list[tuple[str, str | Path]],
) -> dict[str, Any]:
    if not model_inputs:
        raise ValueError("At least one --model MODEL_NAME V1_JSON pair is required")

    model_order: list[str] = []
    values_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_name, input_file in model_inputs:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model name must be a non-empty string")
        if model_name not in values_by_model:
            model_order.append(model_name)
            values_by_model[model_name] = []
        values_by_model[model_name].append(load_plot_data(input_file))

    models: list[dict[str, Any]] = []
    for model_name in model_order:
        merged_v1 = overlay_plot_data(values_by_model[model_name])
        model: dict[str, Any] = {
            "name": model_name,
            "display_name": model_name,
            "datasets": deepcopy(merged_v1["datasets"]),
        }
        metadata = {
            key: deepcopy(value)
            for key, value in merged_v1.items()
            if key not in {"schema_version", "datasets"}
        }
        if metadata:
            model["metadata"] = metadata
        models.append(model)

    return validate_model_plot_data(
        {"schema_version": MODEL_SCHEMA_VERSION, "models": models}
    )


def merge_and_write_model_files(
    model_inputs: list[tuple[str, str | Path]], output_file: str | Path
) -> dict[str, Any]:
    output_path = Path(output_file).resolve()
    input_paths = {Path(input_file).resolve() for _, input_file in model_inputs}
    if output_path in input_paths:
        raise ValueError("Output file must not overwrite an input plot-data file")
    data = merge_v1_model_files(model_inputs)
    atomic_write_json(output_file, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge named schema-version-1 plot data into a multi-model schema-version-2 JSON."
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        required=True,
        metavar=("MODEL_NAME", "V1_JSON"),
        help="Assign a v1 JSON to a model; repeat in overlay order.",
    )
    parser.add_argument(
        "--output-file", default=DEFAULT_OUTPUT_FILE, help="Output schema-v2 JSON file."
    )
    parser.add_argument(
        "--run-suffix",
        default=None,
        help="Optional suffix inserted before the output file extension.",
    )
    args = parser.parse_args()

    output_file = suffix_output_path(args.output_file, args.run_suffix)
    run_dir = allocate_run_dir("./logs/merge_main_results_models", args.run_suffix)
    with RuntimeContext(
        entrypoint="merge_main_results_models",
        run_dir=run_dir,
        config_file=None,
        resolved_config=None,
        config_snapshot_name=None,
        cli_args=vars(args).copy(),
    ):
        model_inputs = [(name, path) for name, path in args.model]
        data = merge_and_write_model_files(model_inputs, output_file)
        LOGGER.info(
            "Wrote %d models from %d v1 inputs to %s",
            len(data["models"]),
            len(model_inputs),
            output_file,
        )


if __name__ == "__main__":
    main()
