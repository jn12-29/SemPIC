import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_scripts.draw_main_results import plot_results
from plot_scripts.main_results_data import (
    load_plot_document_files,
    overlay_model_plot_data,
    validate_model_plot_data,
)
from plot_scripts.merge_main_results_models import (
    merge_and_write_model_files,
    merge_v1_model_files,
)


def point(label: str, value: float) -> dict:
    return {"label": label, "f1": value, "ttft": value, "flops": value * 100}


def v1_data(*, dataset: str, series: list[dict], **metadata) -> dict:
    return {
        "schema_version": 1,
        **metadata,
        "datasets": [{"name": dataset, "series": series}],
    }


def series(name: str, *points: dict, **metadata) -> dict:
    return {"name": name, **metadata, "points": list(points)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class MainResultsModelSchemaTests(unittest.TestCase):
    def test_converter_groups_models_and_overlays_repeated_model_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            llama_base = root / "llama_base.json"
            llama_override = root / "llama_override.json"
            qwen = root / "qwen.json"
            write_json(
                llama_base,
                v1_data(
                    dataset="biography",
                    title="Llama source",
                    series=[
                        series("sempic", point("base", 0.7), color="blue"),
                        series("kvpacket", point("keep", 0.6)),
                    ],
                ),
            )
            write_json(
                llama_override,
                v1_data(
                    dataset="biography",
                    title="ignored title",
                    series=[series("sempic", point("override", 0.9), color="red")],
                ),
            )
            write_json(
                qwen,
                v1_data(
                    dataset="niah",
                    series=[series("sempic", point("qwen", 0.8))],
                ),
            )

            data = merge_v1_model_files(
                [
                    ("Llama-3.1-8B", llama_base),
                    ("Qwen3-8B", qwen),
                    ("Llama-3.1-8B", llama_override),
                ]
            )

            self.assertEqual(data["schema_version"], 2)
            self.assertEqual(
                [model["name"] for model in data["models"]],
                ["Llama-3.1-8B", "Qwen3-8B"],
            )
            llama = data["models"][0]
            self.assertEqual(llama["display_name"], "Llama-3.1-8B")
            self.assertEqual(llama["metadata"], {"title": "Llama source"})
            llama_series = llama["datasets"][0]["series"]
            self.assertEqual([item["name"] for item in llama_series], ["sempic", "kvpacket"])
            self.assertEqual(llama_series[0]["points"][0]["label"], "override")
            self.assertEqual(llama_series[0]["color"], "red")
            self.assertEqual(llama_series[1]["points"][0]["label"], "keep")

    def test_v2_validator_requires_display_name_and_unique_models(self):
        dataset = v1_data(
            dataset="biography", series=[series("sempic", point("run", 0.8))]
        )["datasets"][0]
        with self.assertRaisesRegex(ValueError, "display_name"):
            validate_model_plot_data(
                {"schema_version": 2, "models": [{"name": "model", "datasets": [dataset]}]}
            )
        with self.assertRaisesRegex(ValueError, "duplicate model name"):
            validate_model_plot_data(
                {
                    "schema_version": 2,
                    "models": [
                        {"name": "model", "display_name": "A", "datasets": [dataset]},
                        {"name": "model", "display_name": "B", "datasets": [dataset]},
                    ],
                }
            )

    def test_v2_overlay_isolated_by_model_and_replaces_whole_series(self):
        base = {
            "schema_version": 2,
            "title": "first root",
            "models": [
                {
                    "name": "llama",
                    "display_name": "Llama",
                    "datasets": v1_data(
                        dataset="biography",
                        series=[series("sempic", point("base", 0.7), marker="o")],
                    )["datasets"],
                }
            ],
        }
        override = {
            "schema_version": 2,
            "title": "ignored root",
            "models": [
                {
                    "name": "llama",
                    "display_name": "Ignored Llama",
                    "datasets": v1_data(
                        dataset="biography",
                        series=[series("sempic", point("override", 0.9), color="red")],
                    )["datasets"],
                },
                {
                    "name": "qwen",
                    "display_name": "Qwen",
                    "datasets": v1_data(
                        dataset="biography",
                        series=[series("sempic", point("qwen", 0.8))],
                    )["datasets"],
                },
            ],
        }

        merged = overlay_model_plot_data([base, override])

        self.assertEqual(merged["title"], "first root")
        self.assertEqual([model["name"] for model in merged["models"]], ["llama", "qwen"])
        llama = merged["models"][0]
        self.assertEqual(llama["display_name"], "Llama")
        replaced = llama["datasets"][0]["series"][0]
        self.assertEqual(replaced["points"][0]["label"], "override")
        self.assertNotIn("marker", replaced)

        override["models"][0]["datasets"][0]["series"][0]["points"][0][
            "label"
        ] = "mutated input"
        self.assertEqual(replaced["points"][0]["label"], "override")
        replaced["points"][0]["label"] = "mutated result"
        self.assertEqual(
            override["models"][0]["datasets"][0]["series"][0]["points"][0][
                "label"
            ],
            "mutated input",
        )

    def test_loader_rejects_mixed_schema_versions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v1_file = root / "v1.json"
            v2_file = root / "v2.json"
            v1 = v1_data(
                dataset="biography", series=[series("sempic", point("v1", 0.8))]
            )
            write_json(v1_file, v1)
            write_json(
                v2_file,
                {
                    "schema_version": 2,
                    "models": [
                        {
                            "name": "model",
                            "display_name": "Model",
                            "datasets": v1["datasets"],
                        }
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "Cannot mix"):
                load_plot_document_files([v1_file, v2_file])

    def test_converter_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.json"
            write_json(
                source,
                v1_data(
                    dataset="biography",
                    series=[series("sempic", point("run", 0.8))],
                ),
            )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                merge_and_write_model_files([("Model", source)], source)

    def test_v2_render_shows_model_headers_and_one_global_legend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "models.svg"
            llama_datasets = [
                v1_data(
                    dataset="biography",
                    series=[series("sempic", point("llama", 0.9))],
                )["datasets"][0],
                v1_data(
                    dataset="niah",
                    series=[series("kvpacket", point("llama-kv", 0.8))],
                )["datasets"][0],
            ]
            qwen_datasets = v1_data(
                dataset="biography",
                series=[series("sempic", point("qwen", 0.85))],
            )["datasets"]
            data = {
                "schema_version": 2,
                "models": [
                    {
                        "name": "llama",
                        "display_name": "Llama-3.1-8B",
                        "datasets": llama_datasets,
                    },
                    {
                        "name": "qwen",
                        "display_name": "Qwen3-8B",
                        "datasets": qwen_datasets,
                    },
                ],
            }

            plot_results(validate_model_plot_data(data), output)

            svg = output.read_text(encoding="utf-8")
            self.assertIn("Llama-3.1-8B", svg)
            self.assertIn("Qwen3-8B", svg)
            self.assertIn("Needle-in-a-Haystack", svg)
            self.assertEqual(svg.count("<!-- SemPIC -->"), 1)


if __name__ == "__main__":
    unittest.main()
