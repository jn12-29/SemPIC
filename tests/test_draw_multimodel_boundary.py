import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_scripts.draw_multimodel_boundary import (
    COMBINED_MODEL_ORDER,
    COMBINED_MODEL_STYLES,
    DATASET_ORDER,
    load_plot_data,
    main,
    parse_args,
    plot_combined,
    plot_models,
    validate_plot_data,
)


def _point(dataset_id: str, offset: float = 0.0) -> dict:
    return {
        "dataset_id": dataset_id,
        "f1": {
            "full": 0.9,
            "no_cache": 0.1,
            "no_recompute": 0.2,
            "kvpacket": 0.2 + (0.05 + offset) * 0.7,
            "sempic": 0.83,
            "joint": 0.84,
        },
        "kv_recovery": 0.05 + offset,
        "sempic_recovery": 0.9,
        "f1_change": 0.63,
        "kv_pre_ratio": 0.2 + offset,
        "kv_interior_ratio": 0.9 + offset,
        "kv_rint": 0.9 + offset,
        "sempic_rint": 0.7,
        "sink_profile": [
            {
                "start": index / 20,
                "end": (index + 1) / 20,
                "mean": 2.0 if index < 2 else 1.0,
                "sem": 0.001,
                "count": 100,
            }
            for index in range(20)
        ],
        "sink_ratio": 2.0,
        "source_ids": {
            "f1": "f1_authority",
            "boundary_attention": "source",
            "interior_attention": "source",
            "sink_attention": "source",
        },
    }


def _payload() -> dict:
    return {
        "schema_name": "sempic.paper_multimodel_evidence",
        "schema_version": 1,
        "sources": [
            {
                "source_id": "f1_authority",
                "path": "/tmp/authority.csv",
                "sha256": "a" * 64,
            },
            {"source_id": "source", "path": "/tmp/source", "sha256": "b" * 64},
        ],
        "models": [
            {
                "model_id": "Qwen3-4B-Instruct-2507",
                "display_name": "Qwen3-4B",
                "points": [_point(dataset, 0.00) for dataset in DATASET_ORDER],
            },
            {
                "model_id": "Qwen3-8B",
                "display_name": "Qwen3-8B",
                "points": [_point(dataset, 0.01) for dataset in DATASET_ORDER],
            },
            {
                "model_id": "Llama-3.1-8B-Instruct",
                "display_name": "Llama-3.1-8B",
                "points": [_point(dataset, 0.02) for dataset in DATASET_ORDER],
            },
        ],
    }


class DrawMultimodelBoundaryTest(unittest.TestCase):
    def test_load_accepts_the_fixed_four_dataset_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "plot_data.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")

            data = load_plot_data(source)

        self.assertEqual(
            [point["dataset_id"] for point in data["models"][0]["points"]],
            list(DATASET_ORDER),
        )

    def test_schema_name_and_version_are_strict(self):
        for field, value in (
            ("schema_name", "wrong"),
            ("schema_version", 2),
            ("schema_version", 1.0),
            ("schema_version", True),
        ):
            with self.subTest(field=field, value=value):
                payload = _payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, "version 1"):
                    validate_plot_data(payload)

    def test_points_require_each_dataset_exactly_once(self):
        payload = _payload()
        payload["models"][0]["points"][0] = _point("hotpot_qa")
        with self.assertRaisesRegex(ValueError, "all four datasets in order"):
            validate_plot_data(payload)

        payload = _payload()
        payload["models"][0]["points"].pop()
        with self.assertRaisesRegex(ValueError, "all four datasets in order"):
            validate_plot_data(payload)

    def test_shared_contract_rejects_non_full100_profile_counts(self):
        payload = _payload()
        for model in payload["models"]:
            for point in model["points"]:
                for profile_bin in point["sink_profile"]:
                    profile_bin["count"] = 20

        with tempfile.TemporaryDirectory() as directory:
            for renderer in (plot_models, plot_combined):
                with self.subTest(renderer=renderer.__name__):
                    with self.assertRaisesRegex(
                        ValueError, "Invalid sink profile bin"
                    ):
                        renderer(payload, directory)

    def test_shared_contract_rejects_boundary_validation_rint_mismatch(self):
        payload = _payload()
        payload["models"][0]["points"][0]["kv_interior_ratio"] = 0.8

        with tempfile.TemporaryDirectory() as directory:
            for renderer in (plot_models, plot_combined):
                with self.subTest(renderer=renderer.__name__):
                    with self.assertRaisesRegex(ValueError, "must be identical"):
                        renderer(payload, directory)

    def test_values_must_be_finite_nonnegative_numbers(self):
        for value in (-0.1, float("inf"), float("nan"), True, "0.5", None):
            with self.subTest(value=value):
                payload = _payload()
                payload["models"][0]["points"][0]["kv_recovery"] = value
                with self.assertRaisesRegex(
                    ValueError, "numeric|finite and nonnegative"
                ):
                    validate_plot_data(payload)

    def test_plot_models_writes_one_tight_three_format_figure_per_model(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = plot_models(_payload(), directory)

            self.assertEqual(set(outputs), set(COMBINED_MODEL_ORDER))
            expected_names = {
                "boundary_motivation_Qwen3-4B-Instruct-2507.svg",
                "boundary_motivation_Qwen3-4B-Instruct-2507.pdf",
                "boundary_motivation_Qwen3-4B-Instruct-2507.png",
                "boundary_motivation_Qwen3-8B.svg",
                "boundary_motivation_Qwen3-8B.pdf",
                "boundary_motivation_Qwen3-8B.png",
                "boundary_motivation_Llama-3.1-8B-Instruct.svg",
                "boundary_motivation_Llama-3.1-8B-Instruct.pdf",
                "boundary_motivation_Llama-3.1-8B-Instruct.png",
            }
            self.assertEqual({path.name for paths in outputs.values() for path in paths}, expected_names)
            for path in (path for paths in outputs.values() for path in paths):
                self.assertGreater(path.stat().st_size, 100)
            svg = outputs["Qwen3-8B"][0].read_text(encoding="utf-8")
            self.assertIn("Qwen3-8B", svg)
            self.assertIn("F1 recovery", svg)
            self.assertIn("KV pre", svg)
            self.assertIn("KV interior", svg)
            self.assertEqual(svg.count('id="recovery-bar-'), 4)
            self.assertEqual(svg.count('id="pre-interior-arrow-'), 4)
            self.assertEqual(svg.count('id="pre-marker-'), 4)
            self.assertEqual(svg.count('id="interior-marker-'), 4)
            self.assertIn('id="unit-reference-line"', svg)
            self.assertIn("stroke-dasharray", svg)
            self.assertIn("<pattern", svg)

    def test_repeatable_model_id_cli_selects_requested_models(self):
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
                    "Llama-3.1-8B-Instruct",
                ]
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "boundary_motivation_Llama-3.1-8B-Instruct.svg",
                    "boundary_motivation_Llama-3.1-8B-Instruct.pdf",
                    "boundary_motivation_Llama-3.1-8B-Instruct.png",
                },
            )

        args = parse_args(
            [
                "plot.json",
                "--output-dir",
                "figures",
                "--model-id",
                "Llama-3.1-8B-Instruct",
                "--model-id",
                "Qwen3-8B",
            ]
        )
        self.assertEqual(args.model_id, ["Llama-3.1-8B-Instruct", "Qwen3-8B"])

    def test_combined_renderer_writes_three_formats_and_twelve_glyphs(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = plot_combined(_payload(), directory)

            self.assertEqual(
                {path.name for path in outputs},
                {
                    "boundary_motivation_all_models.svg",
                    "boundary_motivation_all_models.pdf",
                    "boundary_motivation_all_models.png",
                },
            )
            for path in outputs:
                self.assertGreater(path.stat().st_size, 100)
            svg = outputs[0].read_text(encoding="utf-8")
            self.assertEqual(svg.count('id="combined-recovery-bar-'), 12)
            self.assertEqual(svg.count('id="combined-pre-interior-arrow-'), 12)
            self.assertEqual(svg.count('id="combined-pre-marker-'), 12)
            self.assertEqual(svg.count('id="combined-interior-marker-'), 12)
            for label in ("Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B"):
                self.assertIn(label, svg)
            self.assertIn("KV F1 recovery", svg)
            self.assertIn("Pre", svg)
            self.assertIn("Interior", svg)
            self.assertIn('id="combined-unit-reference-line"', svg)
            self.assertIn("stroke-dasharray", svg)

    def test_combined_styles_redundantly_encode_three_models_in_grayscale(self):
        styles = [COMBINED_MODEL_STYLES[model_id] for model_id in COMBINED_MODEL_ORDER]
        self.assertEqual(len({style["marker"] for style in styles}), 3)
        self.assertEqual(len({style["hatch"] for style in styles}), 3)
        for style in styles:
            for field in ("bar_color", "line_color"):
                color = style[field].removeprefix("#")
                self.assertEqual(color[0:2], color[2:4])
                self.assertEqual(color[2:4], color[4:6])

    def test_combined_requires_exact_fixed_model_ids_and_order(self):
        payload = _payload()
        payload["models"][0], payload["models"][1] = (
            payload["models"][1],
            payload["models"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "fixed three-model order"):
                plot_combined(payload, directory)

    def test_combined_cli_writes_only_shared_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "plot_data.json"
            source.write_text(json.dumps(_payload()), encoding="utf-8")
            output_dir = root / "figures"

            status = main(
                [str(source), "--output-dir", str(output_dir), "--combined"]
            )

            self.assertEqual(status, 0)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "boundary_motivation_all_models.svg",
                    "boundary_motivation_all_models.pdf",
                    "boundary_motivation_all_models.png",
                },
            )

    def test_unknown_or_duplicate_model_selection_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unknown model_id"):
                plot_models(_payload(), directory, model_ids=["missing"])
            with self.assertRaisesRegex(ValueError, "must be unique"):
                plot_models(
                    _payload(),
                    directory,
                    model_ids=["Qwen3-8B", "Qwen3-8B"],
                )


if __name__ == "__main__":
    unittest.main()
