import csv
import json
import tempfile
import unittest
from pathlib import Path

from plot_scripts.boundary_motivation_data import (
    ATTENTION_METHODS,
    ATTENTION_REGIONS,
    BEHAVIOR_METHODS,
    TARGET_DATASETS,
    build_motivation_data,
    extract_authoritative_attention_rows,
    materialize_pinned_configs,
    record_bundle_files,
    write_bundle,
)
from plot_scripts.draw_boundary_motivation import draw_diagnostic, load_plot_data


def _config(model_path: str, dataset: str, method: str) -> dict:
    max_new_tokens = {"biography": 32, "hotpot_qa": 512, "musique": 256, "niah": 128}[dataset]
    return {
        "model": {
            "model_path": model_path,
            "dtype": "bfloat16",
            "device": "cuda:0",
            "generation_kwargs": {
                "max_new_tokens": max_new_tokens,
                "stop_strings": ["<|im_end|>"],
                "do_sample": False,
                "use_cache": True,
            },
        },
        "dataset": {
            "dataset_name": dataset,
            "num_samples": 100,
            "split": "test",
            "seed": 42,
            "template": "tokenizer_chat",
            "template_kwargs": {},
            "data_kwargs": {},
        },
        "cache_comb": {"method": method, "kwargs": {}},
        "packet_wrapper": {
            "path": f"./train_outputs/{model_path}/{dataset}/packet_wrapper.pt"
            if method == "kvpacket"
            else None
        },
        "lora": {"path": None},
        "compress": None,
        "quantization": None,
        "seed": 42,
        "logging": {"level": "INFO"},
        "debug_dump": {"enabled": False},
    }


def _write_source(root: Path, model_id: str) -> tuple[Path, Path, dict[tuple[str, str], dict]]:
    configs = {
        (dataset, method): _config(f"./models/{model_id}", dataset, method)
        for dataset in TARGET_DATASETS
        for method in BEHAVIOR_METHODS
    }
    manifest = root / f"{model_id}_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_config": {
                    "query_passes": [{"query_pass_id": "shifted_prediction"}]
                },
                "eval_configs": [
                    {
                        "source_config": f"eval_config/{model_id}/{dataset}/{method}.json",
                        "config": configs[(dataset, method)],
                    }
                    for dataset in TARGET_DATASETS
                    for method in BEHAVIOR_METHODS
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = root / f"{model_id}_summary.csv"
    fieldnames = (
        "model",
        "dataset",
        "query_pass",
        "metric",
        "metric_label",
        "view",
        "method",
        "facets",
        "layer",
        "position_bin",
        "query_head",
        "mean",
        "sem",
        "count",
    )
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset_index, dataset in enumerate(TARGET_DATASETS):
            for method in ATTENTION_METHODS:
                for region in ATTENTION_REGIONS:
                    vanilla = 0.01 if region == "prefix" else 0.001
                    ratio = 0.1 if region == "prefix" else 1.1
                    mean = vanilla if method == "vanilla_pic" else vanilla * ratio
                    writer.writerow(
                        {
                            "model": model_id,
                            "dataset": dataset,
                            "query_pass": "shifted_prediction",
                            "metric": "attention_absolute_deviation",
                            "metric_label": "Attention absolute deviation",
                            "view": "global_bar",
                            "method": method,
                            "facets": json.dumps(
                                {
                                    "attention_view": "raw",
                                    "edge_ratio": "0.1",
                                    "region": region,
                                }
                            ),
                            "mean": mean + dataset_index * 0.000001,
                            "sem": 0.00001,
                            "count": 100,
                        }
                    )
        writer.writerow(
            {
                "model": model_id,
                "dataset": "biography",
                "query_pass": "shifted_prediction",
                "metric": "attention_absolute_deviation",
                "metric_label": "distractor",
                "view": "layer_position_heatmap",
                "method": "kvpacket",
                "facets": json.dumps({"attention_view": "raw"}),
                "mean": 999,
                "sem": 0,
                "count": 100,
            }
        )
    return manifest, summary, configs


def _write_results(
    root: Path,
    model_id: str,
    configs: dict[tuple[str, str], dict],
    methods: tuple[str, ...],
) -> list[Path]:
    paths = []
    f1_by_method = {"full_recompute": 0.9, "no_recompute": 0.2, "kvpacket": 0.76}
    for dataset in TARGET_DATASETS:
        for method in methods:
            path = root / f"{model_id}_{dataset}_{method}_result.json"
            path.write_text(
                json.dumps(
                    {
                        "config": configs[(dataset, method)],
                        "result": {"f1": f1_by_method[method]},
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
    return paths


class BoundaryMotivationEvidenceTest(unittest.TestCase):
    def _fixture(self, root: Path, methods: tuple[str, ...]):
        sources = []
        results = []
        configs_by_model = {}
        for model_id in ("Qwen3-4B-Instruct-2507", "Qwen3-8B"):
            manifest, summary, configs = _write_source(root, model_id)
            sources.append((manifest, summary))
            configs_by_model[model_id] = configs
            results.extend(_write_results(root, model_id, configs, methods))
        return sources, results, configs_by_model

    def test_exact_global_bar_extraction_ignores_heatmap(self):
        with tempfile.TemporaryDirectory() as directory:
            _, summary, _ = _write_source(Path(directory), "Qwen3-4B-Instruct-2507")
            model_id, rows = extract_authoritative_attention_rows(summary)
            self.assertEqual(model_id, "Qwen3-4B-Instruct-2507")
            self.assertEqual(len(rows), 16)
            self.assertNotIn(999, {row["global_bar_mean"] for row in rows.values()})

    def test_duplicate_authoritative_row_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            _, summary, _ = _write_source(Path(directory), "Qwen3-8B")
            lines = summary.read_text(encoding="utf-8").splitlines()
            summary.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate authoritative row"):
                extract_authoritative_attention_rows(summary)

    def test_missing_behavior_keeps_all_eight_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, ("kvpacket",))
            data = build_motivation_data(sources, results, root)
            self.assertEqual(len(data["plot_rows"]), 8)
            for row in data["plot_rows"]:
                self.assertEqual(row["status"], "blocked")
                self.assertEqual(row["full_recompute_status"], "missing")
                self.assertEqual(row["kvpacket_status"], "matched")
                self.assertIsNone(row["f1_recovery_fraction"])
                self.assertEqual(row["prefix_attention_ratio_status"], "defined")

    def test_complete_data_has_method_ids_and_unclipped_formulas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, BEHAVIOR_METHODS)
            data = build_motivation_data(sources, results, root)
            self.assertEqual(len(data["behavior_measurements"]), 24)
            for row in data["plot_rows"]:
                self.assertEqual(row["status"], "pass")
                self.assertAlmostEqual(row["f1_residual_gap"], 0.14)
                self.assertAlmostEqual(row["f1_recovery_fraction"], 0.8)
                self.assertEqual(
                    len(
                        {
                            row["full_measurement_id"],
                            row["no_recompute_measurement_id"],
                            row["kvpacket_measurement_id"],
                        }
                    ),
                    3,
                )

    def test_identical_exact_results_are_recorded_as_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, BEHAVIOR_METHODS)
            duplicate = root / "duplicate_result.json"
            duplicate.write_text(results[0].read_text(encoding="utf-8"), encoding="utf-8")
            data = build_motivation_data(sources, [*results, duplicate], root)
            aliased = [
                measurement
                for measurement in data["behavior_measurements"]
                if measurement["result_path_aliases"]
            ]
            self.assertEqual(len(aliased), 1)

    def test_conflicting_exact_results_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, BEHAVIOR_METHODS)
            payload = json.loads(results[0].read_text(encoding="utf-8"))
            payload["result"]["f1"] = 0.123
            duplicate = root / "conflicting_result.json"
            duplicate.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ambiguous exact results"):
                build_motivation_data(sources, [*results, duplicate], root)

    def test_manifest_methods_must_share_behavior_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, ("kvpacket",))
            manifest_path = sources[0][0]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["eval_configs"][1]["config"]["model"]["model_path"] = "./models/wrong"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Summary model|Behavior fields differ"):
                build_motivation_data(sources, results, root)

    def test_summary_model_must_match_manifest_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, ("kvpacket",))
            summary_path = sources[0][1]
            text = summary_path.read_text(encoding="utf-8")
            summary_path.write_text(
                text.replace("Qwen3-4B-Instruct-2507", "WrongModel"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match manifest models"):
                build_motivation_data(sources, results, root)

    def test_nonpositive_recovery_denominator_is_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, BEHAVIOR_METHODS)
            for path in results:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["config"]["cache_comb"]["method"] == "full_recompute":
                    payload["result"]["f1"] = 0.1
                    path.write_text(json.dumps(payload), encoding="utf-8")
            data = build_motivation_data(sources, results, root)
            for row in data["plot_rows"]:
                self.assertIsNone(row["f1_recovery_fraction"])
                self.assertEqual(
                    row["f1_recovery_status"], "nonpositive_denominator"
                )

    def test_materializes_exactly_sixteen_full_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, ("kvpacket",))
            data = build_motivation_data(sources, results, root)
            pinned = materialize_pinned_configs(data["points"], root / "pinned")
            self.assertEqual(len(pinned), 16)
            for (_, _, method), path in pinned.items():
                self.assertIn(method, {"full_recompute", "no_recompute"})
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["cache_comb"]["method"], method)
                self.assertIn("stop_strings", payload["model"]["generation_kwargs"])

    def test_records_pinned_configs_in_bundle_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, ("kvpacket",))
            data = build_motivation_data(sources, results, root)
            bundle = root / "bundle"
            write_bundle(data, bundle)
            pinned = materialize_pinned_configs(data["points"], bundle / "resolved_configs")
            record_bundle_files(bundle, pinned.values())
            marker = json.loads(
                (bundle / ".boundary_motivation_bundle.json").read_text(encoding="utf-8")
            )
            recorded = [
                path for path in marker["generated_files"] if path.startswith("resolved_configs/")
            ]
            self.assertEqual(len(recorded), 16)
            write_bundle(data, bundle, overwrite=True)

    def test_bundle_guard_and_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources, results, _ = self._fixture(root, BEHAVIOR_METHODS)
            data = build_motivation_data(sources, results, root)
            output_dir = root / "bundle"
            outputs = write_bundle(data, output_dir)
            self.assertEqual(len(load_plot_data(outputs["plot_data"])), 8)
            with self.assertRaises(FileExistsError):
                write_bundle(data, output_dir)
            write_bundle(data, output_dir, overwrite=True)

            figures = draw_diagnostic(outputs["plot_data"], output_dir)
            self.assertEqual(set(figures), {"svg", "pdf", "png"})
            for path in figures.values():
                self.assertGreater(path.stat().st_size, 100)
            svg = figures["svg"].read_text(encoding="utf-8")
            self.assertIn("F1 recovery", svg)
            self.assertIn("Rpre → Rint", svg)
            write_bundle(data, output_dir, overwrite=True)
            with self.assertRaisesRegex(ValueError, "Unsafe output stem"):
                draw_diagnostic(
                    outputs["plot_data"], output_dir, stem="../escape"
                )


if __name__ == "__main__":
    unittest.main()
