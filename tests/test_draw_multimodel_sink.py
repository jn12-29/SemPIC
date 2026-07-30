import json
import math
import re
import tempfile
import unittest
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")

from plot_scripts.draw_multimodel_sink import (
    COMBINED_OUTPUT_STEM,
    DATASET_ORDER,
    MODEL_ORDER,
    SCHEMA_NAME,
    draw_combined_sink,
    draw_multimodel_sink,
    load_plot_data,
    main,
    parse_args,
    validate_plot_data,
)


def _profile(scale: float = 1.0) -> list[dict[str, float | int]]:
    means = [6.0, 2.0, 1.0, 1.0, 2.0, 4.0]
    edges = [0.0, 0.1, 0.3, 0.7, 0.9, 0.95, 1.0]
    return [
        {
            "start": edges[index],
            "end": edges[index + 1],
            "mean": mean * scale,
            "sem": 0.01 * scale,
            "count": 100,
        }
        for index, mean in enumerate(means)
    ]


def _point(dataset_id: str, scale: float) -> dict:
    full = 0.9
    no_recompute = 0.2
    kvpacket = 0.7
    sempic = 0.8
    denominator = full - no_recompute
    return {
        "dataset_id": dataset_id,
        "f1": {
            "full": full,
            "no_cache": 0.1,
            "no_recompute": no_recompute,
            "kvpacket": kvpacket,
            "sempic": sempic,
            "joint": 0.75,
        },
        "kv_recovery": (kvpacket - no_recompute) / denominator,
        "sempic_recovery": (sempic - no_recompute) / denominator,
        "f1_change": sempic - no_recompute,
        "kv_pre_ratio": 0.25,
        "kv_interior_ratio": 0.75,
        "kv_rint": 0.75,
        "sempic_rint": 0.95,
        "sink_profile": _profile(scale),
        "sink_ratio": 4.8,
        "source_ids": {
            "f1": "f1_authority",
            "boundary_attention": "attention_fixture",
            "interior_attention": "attention_fixture",
            "sink_attention": "attention_fixture",
        },
    }


def _payload() -> dict:
    display_names = ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B")
    models = []
    for model_index, (model_id, display_name) in enumerate(
        zip(MODEL_ORDER, display_names, strict=True)
    ):
        models.append(
            {
                "model_id": model_id,
                "display_name": display_name,
                "points": [
                    _point(dataset_id, 1.0 + 0.1 * (model_index + dataset_index))
                    for dataset_index, dataset_id in enumerate(DATASET_ORDER)
                ],
            }
        )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "sources": [
            {
                "source_id": "f1_authority",
                "path": "res/paper_f1_authority.csv",
                "sha256": "0" * 64,
            },
            {
                "source_id": "attention_fixture",
                "path": "tests/fixtures/attention.json",
                "sha256": "1" * 64,
            },
        ],
        "models": models,
    }


def _combined_payload() -> dict:
    return _payload()


class MultimodelSinkPlotTests(unittest.TestCase):
    def test_load_and_render_all_models_with_exact_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plot_data.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")

            data = load_plot_data(source)
            outputs = draw_multimodel_sink(data, root / "figures")

            self.assertEqual(set(outputs), set(MODEL_ORDER))
            self.assertEqual(
                {path.name for formats in outputs.values() for path in formats.values()},
                {
                    "attention_sink_Qwen3-4B-Instruct-2507.svg",
                    "attention_sink_Qwen3-4B-Instruct-2507.pdf",
                    "attention_sink_Qwen3-4B-Instruct-2507.png",
                    "attention_sink_Qwen3-8B.svg",
                    "attention_sink_Qwen3-8B.pdf",
                    "attention_sink_Qwen3-8B.png",
                    "attention_sink_Llama-3.1-8B-Instruct.svg",
                    "attention_sink_Llama-3.1-8B-Instruct.pdf",
                    "attention_sink_Llama-3.1-8B-Instruct.png",
                },
            )
            for formats in outputs.values():
                self.assertEqual(set(formats), {"svg", "pdf", "png"})
                self.assertTrue(
                    all(path.is_file() and path.stat().st_size > 100 for path in formats.values())
                )
            svg = outputs[MODEL_ORDER[0]]["svg"].read_text(encoding="utf-8")
            for text in (
                "Qwen3-4B",
                "Biography",
                "HotpotQA",
                "MuSiQue",
                "NIAH",
                "Sink",
                "Pre/Int",
                "ΔF1",
                "4.8",
                "+0.600",
            ):
                self.assertIn(text, svg)
            self.assertNotIn("Pre÷Int", svg)
            font_sizes = [
                float(value) for value in re.findall(r"font-size: ([0-9.]+)px", svg)
            ]
            self.assertTrue(font_sizes)
            self.assertGreaterEqual(min(font_sizes), 8.0)

    def test_model_selection_is_repeatable_and_optional(self):
        args = parse_args(
            [
                "plot.json",
                "--output-dir",
                "figures",
                "--model-id",
                MODEL_ORDER[1],
                "--model-id",
                MODEL_ORDER[0],
            ]
        )
        self.assertEqual(args.model_id, [MODEL_ORDER[1], MODEL_ORDER[0]])

        combined_args = parse_args(
            ["plot.json", "--output-dir", "figures", "--combined"]
        )
        self.assertTrue(combined_args.combined)

        with tempfile.TemporaryDirectory() as directory:
            outputs = draw_multimodel_sink(
                _payload(), directory, model_ids=[MODEL_ORDER[1], MODEL_ORDER[1]]
            )
            self.assertEqual(list(outputs), [MODEL_ORDER[1]])
            self.assertEqual(len(list(Path(directory).iterdir())), 3)
            with self.assertRaisesRegex(ValueError, "Unknown --model-id"):
                draw_multimodel_sink(_payload(), directory, model_ids=["missing"])

    def test_main_loads_and_renders_only_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plot_data.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")
            output_dir = root / "figures"

            status = main(
                [
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--model-id",
                    MODEL_ORDER[1],
                ]
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "attention_sink_Qwen3-8B.svg",
                    "attention_sink_Qwen3-8B.pdf",
                    "attention_sink_Qwen3-8B.png",
                },
            )

    def test_combined_renderer_writes_twelve_row_triformat_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = draw_combined_sink(_combined_payload(), directory)

            self.assertEqual(set(outputs), {"svg", "pdf", "png"})
            self.assertEqual(
                {path.name for path in outputs.values()},
                {
                    f"{COMBINED_OUTPUT_STEM}.svg",
                    f"{COMBINED_OUTPUT_STEM}.pdf",
                    f"{COMBINED_OUTPUT_STEM}.png",
                },
            )
            self.assertTrue(
                all(path.is_file() and path.stat().st_size > 100 for path in outputs.values())
            )
            svg = outputs["svg"].read_text(encoding="utf-8")
            svg_width = float(
                re.search(r'<svg[^>]+width="([0-9.]+)pt"', svg).group(1)
            ) / 72.0
            svg_height = float(
                re.search(r'<svg[^>]+height="([0-9.]+)pt"', svg).group(1)
            ) / 72.0
            self.assertLessEqual(svg_width, 3.4)
            self.assertGreaterEqual(svg_height, 3.0)
            self.assertLessEqual(svg_height, 3.2)
            pdf = outputs["pdf"].read_bytes().decode("latin-1")
            pdf_media_box = re.search(
                r"/MediaBox \[ 0 0 ([0-9.]+) ([0-9.]+)", pdf
            )
            pdf_width = float(pdf_media_box.group(1)) / 72.0
            pdf_height = float(pdf_media_box.group(2)) / 72.0
            self.assertLessEqual(pdf_width, 3.4)
            self.assertGreaterEqual(pdf_height, 3.0)
            self.assertLessEqual(pdf_height, 3.2)
            with Image.open(outputs["png"]) as png:
                dpi = float(png.info["dpi"][0])
                self.assertLessEqual(png.width / dpi, 3.4)
                self.assertGreaterEqual(png.height / dpi, 3.0)
                self.assertLessEqual(png.height / dpi, 3.2)
            for model_name in ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B"):
                self.assertIn(model_name, svg)
            for dataset_name in ("Biography", "HotpotQA", "MuSiQue", "NIAH"):
                self.assertEqual(svg.count(f">{dataset_name}</text>"), 3)
            for separator_id in (
                "model-separator-4",
                "model-separator-8",
                "annotation-model-separator-4",
                "annotation-model-separator-8",
            ):
                self.assertIn(f'id="{separator_id}"', svg)
            self.assertIn("Sink", svg)
            self.assertIn("Pre/Int", svg)
            self.assertIn("ΔF1", svg)
            font_sizes = [
                float(value) for value in re.findall(r"font-size: ([0-9.]+)px", svg)
            ]
            self.assertTrue(font_sizes)
            self.assertGreaterEqual(min(font_sizes), 8.0)

    def test_combined_main_and_contract_require_fixed_models_and_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plot_data.json"
            source.write_text(json.dumps(_combined_payload()), encoding="utf-8")
            output_dir = root / "figures"

            status = main(
                [str(source), "--output-dir", str(output_dir), "--combined"]
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    f"{COMBINED_OUTPUT_STEM}.svg",
                    f"{COMBINED_OUTPUT_STEM}.pdf",
                    f"{COMBINED_OUTPUT_STEM}.png",
                },
            )

        wrong_order = _combined_payload()
        wrong_order["models"][0], wrong_order["models"][1] = (
            wrong_order["models"][1],
            wrong_order["models"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fixed three-model order"):
                draw_combined_sink(wrong_order, directory)

        inconsistent_bins = _combined_payload()
        for point in inconsistent_bins["models"][2]["points"]:
            point["sink_profile"][0]["end"] = 0.05
            point["sink_profile"][1]["start"] = 0.05
            point["sink_ratio"] = 3.2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "consistent position bins"):
                draw_combined_sink(inconsistent_bins, directory)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plot_data.json"
            source.write_text(json.dumps(_combined_payload()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be used"):
                main(
                    [
                        str(source),
                        "--output-dir",
                        str(root / "figures"),
                        "--combined",
                        "--model-id",
                        MODEL_ORDER[0],
                    ]
                )

    def test_rejects_schema_and_fixed_dataset_contract_errors(self):
        invalid_schema = _payload()
        invalid_schema["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "version 1"):
            validate_plot_data(invalid_schema)

        boolean_version = _payload()
        boolean_version["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "version 1"):
            validate_plot_data(boolean_version)

        float_version = _payload()
        float_version["schema_version"] = 1.0
        with self.assertRaisesRegex(ValueError, "version 1"):
            validate_plot_data(float_version)

        missing_dataset = _payload()
        missing_dataset["models"][0]["points"].pop()
        with self.assertRaisesRegex(ValueError, "all four datasets"):
            validate_plot_data(missing_dataset)

        duplicate_dataset = _payload()
        duplicate_dataset["models"][0]["points"][1]["dataset_id"] = "biography"
        with self.assertRaisesRegex(ValueError, "all four datasets"):
            validate_plot_data(duplicate_dataset)

    def test_rejects_invalid_profile_and_metric_values(self):
        nonfinite = _payload()
        nonfinite["models"][0]["points"][0]["sink_ratio"] = math.inf
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            validate_plot_data(nonfinite)

        noncontiguous = _payload()
        noncontiguous["models"][0]["points"][0]["sink_profile"][1]["start"] = 0.11
        with self.assertRaisesRegex(ValueError, "Invalid sink profile bin"):
            validate_plot_data(noncontiguous)

        inconsistent = _payload()
        profile = inconsistent["models"][0]["points"][0]["sink_profile"]
        profile[0]["end"] = 0.05
        profile[1]["start"] = 0.05
        inconsistent["models"][0]["points"][0]["sink_ratio"] = 3.2
        with self.assertRaisesRegex(ValueError, "Inconsistent position bins"):
            validate_plot_data(inconsistent)

        invalid_count = _payload()
        invalid_count["models"][0]["points"][0]["sink_profile"][0]["count"] = 0
        with self.assertRaisesRegex(ValueError, "Invalid sink profile bin"):
            validate_plot_data(invalid_count)

    def test_both_modes_reject_non_full100_and_unrecomputed_sink_ratio(self):
        non_full100 = _payload()
        non_full100["models"][0]["points"][0]["sink_profile"][0]["count"] = 20
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Invalid sink profile bin"):
                draw_multimodel_sink(non_full100, directory)

        altered_sink_ratio = _combined_payload()
        altered_sink_ratio["models"][0]["points"][0]["sink_ratio"] += 1.0
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "sink_ratio does not recompute from sink_profile"
            ):
                draw_combined_sink(altered_sink_ratio, directory)


if __name__ == "__main__":
    unittest.main()
