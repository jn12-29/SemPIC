import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_scripts.build_main_results_data import (
    DEFAULT_OUTPUT_FILE as DEFAULT_DATA_OUTPUT_FILE,
    build_and_write_plot_data,
    build_plot_data,
    discover_result_files,
    write_plot_data,
)
from plot_scripts.draw_main_results import (
    BOLD_LEGEND_SERIES,
    DEFAULT_OUTPUT_FILE as DEFAULT_FIGURE_OUTPUT_FILE,
    STYLE_CONFIG,
    arrange_legend_names,
    canonical_legend_name,
    display_dataset_name,
    display_series_name,
    plot_results,
)
from plot_scripts.main_results_data import (
    load_plot_data,
    overlay_plot_data,
    suffix_output_path,
    validate_plot_data,
)


def result_payload(
    *, dataset: str = "biography", method: str = "sempic", run_suffix: str | None = None
) -> dict:
    config = {
        "dataset": {"dataset_name": dataset},
        "cache_comb": {"method": method},
    }
    if run_suffix is not None:
        config["run_suffix"] = run_suffix
    return {
        "config": config,
        "result": {"f1": 0.8, "ttft": 0.1, "flops": 100.0},
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class MainResultsDataTests(unittest.TestCase):
    def test_default_output_directories(self):
        self.assertEqual(DEFAULT_DATA_OUTPUT_FILE, "plot_data/main_results.json")
        self.assertEqual(
            DEFAULT_FIGURE_OUTPUT_FILE, "plot_figs/gathered_results_plot.pdf"
        )

    def test_series_display_name_mapping(self):
        expected = {
            "a3": "A3",
            "cache_blend": "Cache Blend",
            "epic": "EPIC",
            "full_recompute": "Full Recompute",
            "kv_packet": "KVPacket",
            "kvpacket": "KVPacket",
            "no_cache": "No Cache",
            "no_recompute": "No Recompute",
            "rand_recompute": "Random Recompute",
            "sam_kv": "SAM-KV",
            "sempic": "SemPIC",
            "sempic_kvpacket": "Joint",
        }
        self.assertEqual(
            {name: display_series_name(name) for name in expected},
            expected,
        )
        self.assertEqual(display_series_name("custom_method"), "custom_method")

    def test_dataset_display_name_mapping(self):
        expected = {
            "biography": "Biography",
            "hotpot_qa": "HotpotQA",
            "musique": "MusiQue",
            "niah": "Needle-in-a-Haystack",
        }
        self.assertEqual(
            {name: display_dataset_name(name) for name in expected},
            expected,
        )
        self.assertEqual(display_dataset_name("custom_dataset"), "custom_dataset")

    def test_sempic_family_has_distinct_related_styles(self):
        self.assertEqual(
            STYLE_CONFIG["sempic"],
            {
                "color": "#0072B2",
                "marker": "*",
                "s": 180,
                "zorder": 10,
                "edgecolors": "black",
                "linewidths": 0.5,
            },
        )
        self.assertEqual(
            STYLE_CONFIG["kvpacket"],
            {
                "color": "#6F42C1",
                "marker": "D",
                "s": 125,
                "zorder": 8,
                "edgecolors": "black",
                "linewidths": 0.5,
            },
        )
        self.assertEqual(STYLE_CONFIG["kv_packet"], STYLE_CONFIG["kvpacket"])
        self.assertEqual(
            STYLE_CONFIG["sempic_kvpacket"],
            {
                "color": "#009E73",
                "marker": "P",
                "s": 165,
                "zorder": 9,
                "edgecolors": "black",
                "linewidths": 0.5,
            },
        )
        self.assertEqual(
            len(
                {
                    STYLE_CONFIG[name]["marker"]
                    for name in ("sempic", "kvpacket", "sempic_kvpacket")
                }
            ),
            3,
        )

    def test_legend_order_two_row_layout_and_bold_series(self):
        legend_names, column_count = arrange_legend_names(
            [
                "epic",
                "custom_method",
                "kvpacket",
                "sempic_kvpacket",
                "no_cache",
                "sempic",
            ]
        )

        self.assertEqual(column_count, 3)
        self.assertEqual(
            legend_names,
            [
                "sempic",
                "no_cache",
                "sempic_kvpacket",
                "epic",
                "kvpacket",
                "custom_method",
            ],
        )
        self.assertEqual(BOLD_LEGEND_SERIES, {"sempic", "sempic_kvpacket"})

    def test_single_legend_entry_uses_one_column(self):
        self.assertEqual(arrange_legend_names(["kvpacket"]), (["kvpacket"], 1))

    def test_legend_aliases_are_deduplicated_and_unknown_order_is_stable(self):
        legend_names, column_count = arrange_legend_names(
            ["unknown_b", "kv_packet", "unknown_a", "kvpacket", "sempic"]
        )

        self.assertEqual(canonical_legend_name("kv_packet"), "kvpacket")
        self.assertEqual(column_count, 2)
        self.assertEqual(
            legend_names,
            ["sempic", "unknown_b", "kvpacket", "unknown_a"],
        )

    def test_run_suffix_is_inserted_before_extension(self):
        self.assertEqual(
            suffix_output_path("plot_data/main_results.json", "paper-1"),
            Path("plot_data/main_results_paper-1.json"),
        )
        self.assertEqual(
            suffix_output_path("custom.figure.pdf", "qwen_3.4"),
            Path("custom.figure_qwen_3.4.pdf"),
        )
        self.assertEqual(
            suffix_output_path("plot.pdf", None),
            Path("plot.pdf"),
        )

    def test_invalid_run_suffix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_suffix"):
            suffix_output_path("plot.pdf", "../bad")

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "must be a file path"):
                suffix_output_path(temp_dir, None)

            suffixed_directory = Path(temp_dir) / "plot_paper.pdf"
            suffixed_directory.mkdir()
            with self.assertRaisesRegex(ValueError, "must be a file path"):
                suffix_output_path(Path(temp_dir) / "plot.pdf", "paper")

    def test_build_groups_points_and_preserves_input_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first_result.json"
            second = root / "second_result.json"
            third = root / "third_result.json"
            write_json(first, result_payload(run_suffix="rank-8"))
            write_json(second, result_payload())
            write_json(third, result_payload(dataset="niah", method="kvpacket"))

            data = build_plot_data([first, second, third])

            self.assertEqual(data["schema_version"], 1)
            self.assertEqual([item["name"] for item in data["datasets"]], ["biography", "niah"])
            points = data["datasets"][0]["series"][0]["points"]
            self.assertEqual([point["label"] for point in points], ["rank-8", "second"])
            self.assertEqual(points[0]["source_file"], str(first.resolve()))

    def test_later_plot_data_replaces_whole_matching_series(self):
        base = {
            "schema_version": 1,
            "title": "base title",
            "datasets": [
                {
                    "name": "biography",
                    "display_name": "Base Biography",
                    "series": [
                        {
                            "name": "sempic",
                            "color": "blue",
                            "marker": "o",
                            "points": [
                                {"label": "base-1", "f1": 0.7, "ttft": 0.2, "flops": 20.0},
                                {"label": "base-2", "f1": 0.8, "ttft": 0.3, "flops": 30.0},
                            ],
                        },
                        {
                            "name": "kvpacket",
                            "points": [
                                {"label": "keep", "f1": 0.6, "ttft": 0.1, "flops": 10.0}
                            ],
                        },
                    ],
                }
            ],
        }
        override = {
            "schema_version": 1,
            "title": "override title",
            "datasets": [
                {
                    "name": "biography",
                    "display_name": "Override Biography",
                    "series": [
                        {
                            "name": "sempic",
                            "color": "red",
                            "points": [
                                {"label": "override", "f1": 0.9, "ttft": 0.15, "flops": 15.0}
                            ],
                        },
                        {
                            "name": "new_method",
                            "points": [
                                {"label": "new", "f1": 0.5, "ttft": 0.4, "flops": 40.0}
                            ],
                        },
                    ],
                },
                {
                    "name": "niah",
                    "series": [
                        {
                            "name": "sempic",
                            "points": [
                                {"label": "niah", "f1": 0.75, "ttft": 0.25, "flops": 25.0}
                            ],
                        }
                    ],
                },
            ],
        }
        final_override = {
            "schema_version": 1,
            "datasets": [
                {
                    "name": "biography",
                    "series": [
                        {
                            "name": "sempic",
                            "color": "green",
                            "points": [
                                {"label": "final", "f1": 0.95, "ttft": 0.12, "flops": 12.0}
                            ],
                        }
                    ],
                }
            ],
        }

        merged = overlay_plot_data([base, override, final_override])

        self.assertEqual(merged["title"], "base title")
        self.assertEqual(
            [dataset["name"] for dataset in merged["datasets"]],
            ["biography", "niah"],
        )
        self.assertEqual(merged["datasets"][0]["display_name"], "Base Biography")
        series = merged["datasets"][0]["series"]
        self.assertEqual(
            [item["name"] for item in series],
            ["sempic", "kvpacket", "new_method"],
        )
        self.assertEqual(series[0]["color"], "green")
        self.assertNotIn("marker", series[0])
        self.assertEqual(
            [point["label"] for point in series[0]["points"]],
            ["final"],
        )
        self.assertEqual(series[1]["points"][0]["label"], "keep")

        final_override["datasets"][0]["series"][0]["points"][0]["label"] = "mutated"
        merged["datasets"][0]["series"][1]["points"][0]["label"] = "changed"
        self.assertEqual(series[0]["points"][0]["label"], "final")
        self.assertEqual(base["datasets"][0]["series"][1]["points"][0]["label"], "keep")

    def test_directory_is_recursive_sorted_and_overlapping_input_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "a_result.json"
            second = root / "nested" / "b_result.json"
            ignored = root / "nested" / "config.json"
            second.parent.mkdir()
            (root / "fake_result.json").mkdir()
            write_json(first, result_payload())
            write_json(second, result_payload())
            write_json(ignored, result_payload())

            discovered = discover_result_files([root, second])

            self.assertEqual(discovered, [first, second])
            points = build_plot_data([root, second])["datasets"][0]["series"][0]["points"]
            self.assertEqual(len(points), 2)

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "No result files found"):
                discover_result_files([temp_dir])

    def test_output_cannot_overwrite_input_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "source_result.json"
            write_json(result_file, result_payload())

            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                build_and_write_plot_data([result_file], result_file)

    def test_missing_or_nonfinite_metric_is_rejected_with_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "bad_result.json"
            payload = result_payload()
            del payload["result"]["ttft"]
            write_json(result_file, payload)

            with self.assertRaisesRegex(
                ValueError, rf"{result_file}.*result\.ttft must be a number"
            ):
                build_plot_data([result_file])

    def test_invalid_json_reports_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_file = Path(temp_dir) / "bad_result.json"
            result_file.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, rf"Invalid JSON in {result_file}"):
                build_plot_data([result_file])

    def test_plot_data_round_trip_and_render(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_file = root / "result.json"
            data_file = root / "nested" / "main_results.json"
            output_file = root / "plots" / "main_results.svg"
            write_json(result_file, result_payload())

            data = build_plot_data([result_file])
            series = data["datasets"][0]["series"][0]
            series.update(name="kvpacket", color="navy", marker="o")
            series["points"][0].update(label="manual-label", annotate=True)
            write_plot_data(data, data_file)
            loaded = load_plot_data(data_file)
            plot_results(loaded, output_file)

            self.assertTrue(output_file.is_file())
            svg = output_file.read_text(encoding="utf-8")
            self.assertIn("<svg", svg[:500])
            self.assertIn("manual-label", svg)
            self.assertIn("KVPacket", svg)
            self.assertIn("Biography", svg)
            self.assertNotIn("Better", svg)

    def test_schema_rejects_empty_dataset_collection(self):
        with self.assertRaisesRegex(ValueError, "at least one dataset"):
            validate_plot_data({"schema_version": 1, "datasets": []})

    def test_schema_version_must_be_integer_one(self):
        with self.assertRaisesRegex(ValueError, "schema_version must be 1"):
            validate_plot_data({"schema_version": True, "datasets": []})

    def test_schema_rejects_empty_series_and_points(self):
        with self.assertRaisesRegex(ValueError, "at least one series"):
            validate_plot_data(
                {"schema_version": 1, "datasets": [{"name": "niah", "series": []}]}
            )

        with self.assertRaisesRegex(ValueError, "at least one point"):
            validate_plot_data(
                {
                    "schema_version": 1,
                    "datasets": [
                        {
                            "name": "niah",
                            "series": [{"name": "sempic", "points": []}],
                        }
                    ],
                }
            )


if __name__ == "__main__":
    unittest.main()
