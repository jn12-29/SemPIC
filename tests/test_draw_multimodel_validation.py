import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_scripts.draw_multimodel_validation import (
    COMBINED_MODEL_ORDER,
    DATASET_ORDER,
    MODEL_STYLES,
    _axis_limits,
    _axis_limits_for_models,
    draw_combined_validation,
    draw_multimodel_validation,
    load_plot_data,
    main,
    parse_args,
    validate_plot_data,
)


def point(index: int) -> dict:
    kv_rint = (0.99, 1.05, 1.00, 1.26)[index]
    sempic_recovery = (1.01, 0.88, 0.71, 1.18)[index]
    no_recompute_f1 = 0.4
    full_f1 = 0.8
    sempic_f1 = no_recompute_f1 + sempic_recovery * (
        full_f1 - no_recompute_f1
    )
    return {
        "dataset_id": DATASET_ORDER[index],
        "f1": {
            "full": full_f1,
            "no_cache": 0.2,
            "no_recompute": no_recompute_f1,
            "kvpacket": 0.7,
            "sempic": sempic_f1,
            "joint": 0.82,
        },
        "kv_recovery": 0.75,
        "sempic_recovery": sempic_recovery,
        "f1_change": sempic_f1 - no_recompute_f1,
        "kv_pre_ratio": 0.4,
        "kv_interior_ratio": kv_rint,
        "kv_rint": kv_rint,
        "sempic_rint": (0.93, 0.81, 0.74, 0.99)[index],
        "sink_profile": [
            {
                "start": bin_index / 20,
                "end": (bin_index + 1) / 20,
                "mean": 2.0 if bin_index < 2 else 1.0,
                "sem": 0.01,
                "count": 100,
            }
            for bin_index in range(20)
        ],
        "sink_ratio": 2.0,
        "source_ids": {
            "f1": "f1_authority",
            "boundary_attention": "source",
            "interior_attention": "source",
            "sink_attention": "source",
        },
    }


def model(model_id: str, display_name: str, offset: float = 0.0) -> dict:
    points = [point(index) for index in range(len(DATASET_ORDER))]
    for item in points:
        item["kv_interior_ratio"] += offset
        item["kv_rint"] += offset
        item["sempic_rint"] += offset
    return {
        "model_id": model_id,
        "display_name": display_name,
        "points": points,
    }


def document() -> dict:
    return {
        "schema_name": "sempic.paper_multimodel_evidence",
        "schema_version": 1,
        "sources": [
            {
                "source_id": "source",
                "path": "/tmp/source.json",
                "sha256": "a" * 64,
            },
            {
                "source_id": "f1_authority",
                "path": "/tmp/f1.csv",
                "sha256": "b" * 64,
            },
        ],
        "models": [
            model("Qwen3-4B-Instruct-2507", "Qwen3-4B"),
            model("Qwen3-8B", "Qwen3-8B", -0.02),
            model("Llama-3.1-8B-Instruct", "Llama-3.1-8B", -0.04),
        ],
    }


class MultiModelValidationTests(unittest.TestCase):
    def write_document(self, root: Path, value: dict | None = None) -> Path:
        path = root / "plot_data.json"
        path.write_text(
            json.dumps(document() if value is None else value), encoding="utf-8"
        )
        return path

    def test_renders_one_tight_three_format_figure_per_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.write_document(root)
            output_dir = root / "figures"

            outputs = draw_multimodel_validation(source, output_dir)

            self.assertEqual(list(outputs), list(COMBINED_MODEL_ORDER))
            expected_stems = {
                "Qwen3-4B-Instruct-2507": "sempic_interior_validation_Qwen3-4B-Instruct-2507",
                "Qwen3-8B": "sempic_interior_validation_Qwen3-8B",
                "Llama-3.1-8B-Instruct": "sempic_interior_validation_Llama-3.1-8B-Instruct",
            }
            for model_id, paths in outputs.items():
                self.assertEqual(set(paths), {"svg", "pdf", "png"})
                for extension, path in paths.items():
                    self.assertEqual(path.name, f"{expected_stems[model_id]}.{extension}")
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 100)

            svg = outputs["Qwen3-4B-Instruct-2507"]["svg"].read_text(encoding="utf-8")
            for label in (
                "Qwen3-4B",
                "Biography",
                "HotpotQA",
                "MuSiQue",
                "NIAH",
                "Normalized ratio",
                "F1 recovery (cap 1)",
                "KV Rint",
                "SemPIC Rint",
            ):
                self.assertIn(label, svg)
            self.assertEqual(svg.count('id="recovery-bar-'), 4)
            self.assertEqual(svg.count('id="rint-arrow-'), 4)
            self.assertIn('id="unit-reference-line"', svg)
            self.assertIn('id="truncated-axis-break-1"', svg)
            self.assertIn('id="truncated-axis-break-2"', svg)
            self.assertIn("stroke-dasharray", svg)

            y_min, y_max = _axis_limits(document()["models"][0])
            plotted_values = [
                min(float(item[field]), 1.0)
                if field == "sempic_recovery"
                else float(item[field])
                for item in document()["models"][0]["points"]
                for field in ("sempic_recovery", "kv_rint", "sempic_rint")
            ]
            self.assertNotEqual(y_min, 0.0)
            self.assertLess(y_min, min(plotted_values))
            self.assertGreater(y_max, max(plotted_values))

    def test_combined_mode_writes_one_three_format_fixed_model_figure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.write_document(root)
            output_dir = root / "combined"

            outputs = draw_combined_validation(source, output_dir)

            self.assertEqual(set(outputs), {"svg", "pdf", "png"})
            for extension, path in outputs.items():
                self.assertEqual(
                    path.name,
                    f"sempic_interior_validation_all_models.{extension}",
                )
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 100)

            svg = outputs["svg"].read_text(encoding="utf-8")
            for label in (
                "Qwen3-4B",
                "Qwen3-8B",
                "Llama-3.1-8B",
                "Biography",
                "HotpotQA",
                "MuSiQue",
                "NIAH",
                "F1 recovery (cap 1)",
                "KV Rint",
                "SemPIC Rint",
            ):
                self.assertIn(label, svg)
            self.assertEqual(svg.count('id="combined-recovery-bar-'), 12)
            self.assertEqual(svg.count('id="combined-recovery-bar-capped-'), 6)
            self.assertEqual(svg.count('id="combined-rint-arrow-'), 12)
            self.assertEqual(svg.count('id="combined-kv-rint-'), 12)
            self.assertEqual(svg.count('id="combined-sempic-rint-'), 12)
            self.assertIn('id="combined-unit-reference-line"', svg)
            self.assertIn('id="combined-metric-legend"', svg)
            self.assertIn('id="combined-model-legend"', svg)
            self.assertIn('id="truncated-axis-break-1"', svg)
            self.assertIn('id="truncated-axis-break-2"', svg)
            self.assertIn("stroke-dasharray", svg)
            self.assertGreaterEqual(svg.count("<pattern"), 3)
            self.assertEqual(
                len({style["marker"] for style in MODEL_STYLES.values()}), 3
            )
            self.assertEqual(
                len({style["hatch"] for style in MODEL_STYLES.values()}), 3
            )

            y_min, y_max = _axis_limits_for_models(document()["models"])
            self.assertNotEqual(y_min, 0.0)
            self.assertGreater(y_min, 0.0)
            self.assertGreater(y_max, 1.0)

            cli_output = root / "combined_cli"
            status = main(
                [
                    str(source),
                    "--output-dir",
                    str(cli_output),
                    "--combined",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in cli_output.iterdir()},
                {
                    "sempic_interior_validation_all_models.svg",
                    "sempic_interior_validation_all_models.pdf",
                    "sempic_interior_validation_all_models.png",
                },
            )

    def test_combined_mode_requires_fixed_three_model_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing = document()
            missing["models"].pop()
            source = self.write_document(root, missing)
            with self.assertRaisesRegex(ValueError, "fixed three-model order"):
                draw_combined_validation(source, root / "missing")

            reordered = document()
            reordered["models"][0], reordered["models"][1] = (
                reordered["models"][1],
                reordered["models"][0],
            )
            source = self.write_document(root, reordered)
            with self.assertRaisesRegex(ValueError, "fixed three-model order"):
                draw_combined_validation(source, root / "reordered")

        args = parse_args(["plot.json", "--output-dir", "figures", "--combined"])
        self.assertTrue(args.combined)
        self.assertIsNone(args.model_id)

    def test_model_selection_is_repeatable_and_defaults_to_all(self):
        args = parse_args(
            [
                "plot.json",
                "--output-dir",
                "figures",
                "--model-id",
                "Qwen3-8B",
                "--model-id",
                "Qwen3-4B-Instruct-2507",
            ]
        )
        self.assertEqual(args.model_id, ["Qwen3-8B", "Qwen3-4B-Instruct-2507"])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.write_document(root)
            outputs = draw_multimodel_validation(
                source,
                root / "selected",
                model_ids=["Qwen3-8B"],
            )
            self.assertEqual(list(outputs), ["Qwen3-8B"])
            self.assertEqual(len(list((root / "selected").iterdir())), 3)

            cli_output = root / "cli"
            status = main(
                [
                    str(source),
                    "--output-dir",
                    str(cli_output),
                    "--model-id",
                    "Qwen3-8B",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in cli_output.iterdir()},
                {
                    "sempic_interior_validation_Qwen3-8B.svg",
                    "sempic_interior_validation_Qwen3-8B.pdf",
                    "sempic_interior_validation_Qwen3-8B.png",
                },
            )

    def test_rejects_invalid_schema_shape_order_and_metrics(self):
        cases = []
        wrong_schema = document()
        wrong_schema["schema_name"] = "wrong"
        cases.append((wrong_schema, "version 1"))

        wrong_version = document()
        wrong_version["schema_version"] = 2
        cases.append((wrong_version, "version 1"))

        float_version = document()
        float_version["schema_version"] = 1.0
        cases.append((float_version, "version 1"))

        short_points = document()
        short_points["models"][0]["points"].pop()
        cases.append((short_points, "all four datasets in order"))

        wrong_order = document()
        wrong_order["models"][0]["points"][0]["dataset_id"] = "niah"
        cases.append((wrong_order, "all four datasets in order"))

        negative = document()
        negative["models"][0]["points"][0]["sempic_recovery"] = -0.01
        cases.append((negative, "finite and nonnegative"))

        non_finite = document()
        non_finite["models"][0]["points"][0]["kv_rint"] = math.inf
        cases.append((non_finite, "finite and nonnegative"))

        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate_plot_data(value)

    def test_rejects_shared_contract_violations_before_both_renderers(self):
        mutations = (
            (
                lambda value: value["models"][0]["points"][0]["sink_profile"][0].__setitem__(
                    "count", 20
                ),
                "Invalid sink profile bin",
            ),
            (
                lambda value: value["models"][0]["points"][0].__setitem__(
                    "kv_interior_ratio", 0.8
                ),
                "must be identical",
            ),
        )
        renderers = (draw_multimodel_validation, draw_combined_validation)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for mutation, message in mutations:
                for renderer in renderers:
                    with self.subTest(message=message, renderer=renderer.__name__):
                        value = document()
                        mutation(value)
                        source = self.write_document(root, value)
                        with self.assertRaisesRegex(ValueError, message):
                            renderer(source, root / f"invalid-{renderer.__name__}")

    def test_rejects_unknown_or_repeated_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self.write_document(root)
            with self.assertRaisesRegex(ValueError, "Unknown model_id"):
                draw_multimodel_validation(source, root / "unknown", model_ids=["missing"])
            with self.assertRaisesRegex(ValueError, "Repeated --model-id"):
                draw_multimodel_validation(
                    source,
                    root / "duplicate",
                    model_ids=["Qwen3-8B", "Qwen3-8B"],
                )


if __name__ == "__main__":
    unittest.main()
