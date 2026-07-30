import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

from plot_scripts.attention_sink_data import (
    EXPECTED_DATASETS,
    aggregate_partition,
    recovery_fraction,
    validate_plot_data,
)
from plot_scripts.build_attention_sink_data import (
    _load_sink_partition,
    build_plot_data,
    write_bundle,
)
from plot_scripts.draw_attention_sink import OUTPUT_STEM, plot_diagnostic
from sempic.attention_metrics.profile_identity import runtime_fingerprint
from sempic.attention_metrics.profiles import make_partition
from sempic.attention_metrics.processing import process_partitions
from sempic.attention_metrics.processed_storage import save_processed_metrics


MODEL_ID = "Qwen3-4B-Instruct-2507"
EVAL_SEED = 17
MODEL_CONFIG = {
    "model_path": "/fixture/Qwen3-4B-Instruct-2507",
    "dtype": "bfloat16",
    "device": "cuda:0",
    "generation_kwargs": {
        "max_new_tokens": 32,
        "stop_strings": ["<|im_end|>"],
        "do_sample": False,
        "use_cache": True,
    },
}
METHODS = ("full_recompute", "vanilla_pic", "kvpacket", "sempic")
RESULT_METHODS = ("full_recompute", "vanilla_pic", "sempic")


def method_record(method: str) -> dict:
    resolved = {
        "cache_comb": {"method": method, "kwargs": {}},
        "packet_wrapper": {"path": None},
        "lora": {"path": "/fixture/sempic/lora" if method == "sempic" else None},
        "compress": None,
        "quantization": None,
    }
    if method == "kvpacket":
        resolved["packet_wrapper"]["path"] = "/fixture/kvpacket/packet_wrapper.pt"
    return {
        "method_key": method,
        "runtime_fingerprint": runtime_fingerprint(MODEL_CONFIG, resolved),
        "resolved_method_config": resolved,
        "source_config": f"eval_config/{method}.json",
    }


def dataset_config(dataset_id: str) -> dict:
    return {
        "dataset_name": dataset_id,
        "num_samples": 2,
        "num_data_strs": 5,
        "num_shots": 0,
        "subset": "fixture",
        "split": "test",
        "seed": 42,
        "data_kwargs": {},
        "template": "tokenizer_chat",
        "template_kwargs": {},
    }


def raw_profile(leading: float, interior: float, suffix: float = 0.1) -> torch.Tensor:
    values = [leading] + [interior] * 8 + [suffix]
    return torch.tensor([values, values], dtype=torch.float32)


def chunk(chunk_id: str, sempic_leading: float, sempic_interior: float = 0.2) -> dict:
    token_length = 10
    layouts = {
        method: {
            "pic_start": 1,
            "pic_end": token_length + 1,
            "scope_start": 1,
            "scope_end": token_length + 1,
        }
        for method in METHODS
    }
    layouts["kvpacket"]["scope_start"] = 0
    layouts["kvpacket"]["scope_end"] = token_length + 2
    profile_shape = (2, token_length)
    retrieval_shape = (2, 2)
    attention_profile = {
        "full_recompute": {
            "raw": torch.full(profile_shape, 0.1, dtype=torch.float32),
            "chunk_conditional": torch.full(
                profile_shape, 1 / token_length, dtype=torch.float32
            ),
        },
        **{
            method: {
                "raw_absolute_error": torch.full(
                    profile_shape, 0.01, dtype=torch.float32
                ),
                "chunk_conditional_absolute_error": torch.full(
                    profile_shape, 0.01, dtype=torch.float32
                ),
            }
            for method in ("vanilla_pic", "kvpacket", "sempic")
        },
    }
    attention_profile["kvpacket"]["raw_absolute_error"].fill_(0.02)
    attention_profile["sempic"]["raw_absolute_error"].fill_(0.005)
    retrieval = {
        "full_recompute": {
            "query_count": 2,
            "reference_energy_sum": torch.ones(retrieval_shape),
            "scope_mass_sum": torch.ones(retrieval_shape),
        },
        **{
            method: {
                "query_count": 2,
                "squared_error_sum": torch.zeros(retrieval_shape),
                "reference_energy_sum": torch.ones(retrieval_shape),
                "cosine_distance_sum": torch.zeros(retrieval_shape),
                "cosine_valid_count": torch.full(
                    retrieval_shape, 2, dtype=torch.int64
                ),
                "absolute_mass_error_sum": torch.zeros(retrieval_shape),
                "full_scope_mass_sum": torch.ones(retrieval_shape),
                "candidate_scope_mass_sum": torch.ones(retrieval_shape),
            }
            for method in ("vanilla_pic", "kvpacket", "sempic")
        },
    }
    return {
        "chunk_id": chunk_id,
        "token_digest": f"digest:{chunk_id}",
        "token_length": token_length,
        "method_layouts": layouts,
        "reducer_outputs": {
            "attention_profile": attention_profile,
            "pic_retrieval": retrieval,
            "raw_attention_profile": {
                "full_recompute": {"raw": raw_profile(0.1, 0.1)},
                "vanilla_pic": {"raw": raw_profile(0.5, 0.1)},
                "kvpacket": {"raw": raw_profile(0.4, 0.15)},
                "sempic": {"raw": raw_profile(sempic_leading, sempic_interior)},
            }
        },
    }


def partition(dataset_id: str, *, zero_interior: bool = False) -> dict:
    records = [method_record(method) for method in METHODS]
    configured_paths = {
        MODEL_CONFIG["model_path"],
        "/fixture/sempic/lora",
        "/fixture/kvpacket/packet_wrapper.pt",
    }
    identity = {
        "model_config": MODEL_CONFIG,
        "dataset_config": dataset_config(dataset_id),
        "eval_seed": EVAL_SEED,
        "artifact_snapshots": [
            {
                "artifact_key": f"artifact-{index}",
                "canonical_path": path,
                "files": [],
            }
            for index, path in enumerate(sorted(configured_paths))
        ],
        "model_id": MODEL_ID,
        "dataset_id": dataset_id,
        "query_pass_id": "shifted_prediction",
        "query_spec": {
            "query_pass_id": "shifted_prediction",
            "kind": "gold_answer_shifted_prediction_queries",
            "reducers": [
                "attention_profile",
                "pic_retrieval",
                "raw_attention_profile",
            ],
        },
        "methods": records,
        "max_samples": None,
        "dataset_iteration": dataset_config(dataset_id),
    }
    interior = 0.0 if zero_interior else 0.2
    samples = [
        {
            "sample_index": 0,
            "sample_id": f"{dataset_id}:sample:0",
            "canonical_token_digest": "prompt:0",
            "query_target_digest": "answer:0",
            "query_count": 2,
            "chunks": [
                chunk("a", 0.8, interior),
                chunk("b", 0.2, interior),
            ],
        },
        {
            "sample_index": 1,
            "sample_id": f"{dataset_id}:sample:1",
            "canonical_token_digest": "prompt:1",
            "query_target_digest": "answer:1",
            "query_count": 2,
            "chunks": [chunk("c", 0.8, interior)],
        },
    ]
    return make_partition(
        partition_identity=identity,
        layer_count=2,
        query_head_count=2,
        samples=samples,
    )


def result_payload(dataset_id: str, method: str, f1: float) -> dict:
    config_method = "no_recompute" if method == "vanilla_pic" else method
    return {
        "config": {
            "model": copy.deepcopy(MODEL_CONFIG),
            "dataset": dataset_config(dataset_id),
            "cache_comb": {"method": config_method, "kwargs": {}},
            "packet_wrapper": {"path": None},
            "lora": {"path": "/fixture/sempic/lora" if method == "sempic" else None},
            "compress": None,
            "quantization": None,
            "seed": EVAL_SEED,
            "logging": {"level": "INFO"},
            "debug_dump": {"enabled": False},
        },
        "result": {"f1": f1},
    }


def write_results(root: Path, f1_values=None):
    f1_values = f1_values or {
        "full_recompute": 0.9,
        "vanilla_pic": 0.5,
        "sempic": 0.82,
    }
    result_map = {}
    paths = []
    for dataset_id in EXPECTED_DATASETS:
        for method in RESULT_METHODS:
            path = root / f"{dataset_id}_{method}_result.json"
            payload = result_payload(dataset_id, method, f1_values[method])
            path.write_text(json.dumps(payload), encoding="utf-8")
            result_map[(dataset_id, method)] = (path, payload)
            paths.append(path)
    return result_map, paths


def write_partitions(root: Path, partitions: dict[str, dict]) -> dict[str, Path]:
    paths = {}
    for dataset_id, value in partitions.items():
        path = root / f"{dataset_id}_shifted_prediction.pt"
        torch.save(value, path)
        paths[dataset_id] = path
    return paths


def write_processed(root: Path, partitions: dict[str, dict]):
    artifact = process_partitions(
        list(partitions.values()),
        {
            "position_mode": "normalized",
            "num_position_bins": 10,
            "edge_ratios": ["0.1"],
        },
    )
    path = save_processed_metrics(root / "metrics.pt", artifact)
    return artifact, path


def build_fixture(*, root: Path, partitions, partition_paths, results, num_position_bins=10):
    processed_metrics, processed_metrics_path = write_processed(root, partitions)
    return build_plot_data(
        model_id=MODEL_ID,
        partitions=partitions,
        partition_paths=partition_paths,
        results=results,
        processed_metrics=processed_metrics,
        processed_metrics_path=processed_metrics_path,
        num_position_bins=num_position_bins,
    )


class AttentionSinkEvidenceTests(unittest.TestCase):
    def test_region_aggregation_is_equal_chunk_layer_sample_and_s_is_ratio_of_aggregates(self):
        aggregate = aggregate_partition(partition("biography"), num_position_bins=10)

        sempic = aggregate["methods"]["sempic"]
        # sample 0: equal-chunk leading mean (0.8 + 0.2) / 2 = 0.5;
        # sample 1: 0.8. Equal-sample aggregate is 0.65; interior is 0.2.
        self.assertAlmostEqual(sempic["regions"]["leading"]["mean"], 0.65)
        self.assertAlmostEqual(sempic["regions"]["interior"]["mean"], 0.2)
        self.assertAlmostEqual(aggregate["sink_ratio"], 3.25)
        self.assertEqual(aggregate["sink_ratio_status"], "defined")
        self.assertEqual(
            [record["sample_id"] for record in sempic["sample_values"]],
            ["biography:sample:0", "biography:sample:1"],
        )
        self.assertEqual(len(sempic["positions"]), 10)

    def test_invalid_recovery_and_sink_denominators_are_null_with_status(self):
        recovery, recovery_status = recovery_fraction(0.4, 0.5, 0.6)
        sink = aggregate_partition(partition("biography", zero_interior=True))

        self.assertIsNone(recovery)
        self.assertEqual(recovery_status, "undefined_nonpositive_denominator")
        self.assertIsNone(sink["sink_ratio"])
        self.assertEqual(sink["sink_ratio_status"], "undefined_nonpositive_denominator")

    def test_build_retains_all_four_identities_and_exact_method_result_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            partition_paths = write_partitions(root, partitions)

            data, tables, provenance = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=partition_paths,
                results=results,
                num_position_bins=10,
            )

            self.assertIs(validate_plot_data(data), data)
            self.assertEqual([point["dataset_id"] for point in data["points"]], list(EXPECTED_DATASETS))
            self.assertTrue(all(point["status"] == "pass" for point in data["points"]))
            self.assertTrue(all(abs(point["recovery_fraction"] - 0.8) < 1e-12 for point in data["points"]))
            self.assertTrue(
                all(
                    [row["value"] for row in point["relative_interior_attention_errors"]]
                    == [2.0, 0.5]
                    for point in data["points"]
                )
            )
            self.assertEqual(len(tables["behavior"]), 12)
            self.assertEqual(len(tables["interior_errors"]), 12)
            self.assertEqual(len({row["result_path"] for row in tables["behavior"]}), 12)
            behavior_ids = {row["measurement_id"] for row in tables["behavior"]}
            provenance_ids = {row["measurement_id"] for row in provenance}
            self.assertLessEqual(behavior_ids, provenance_ids)

    def test_missing_dataset_remains_explicit_and_bundle_contains_required_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, result_paths = write_results(root)
            partitions = {
                dataset_id: partition(dataset_id)
                for dataset_id in EXPECTED_DATASETS
                if dataset_id != "niah"
            }
            partition_paths = write_partitions(root, partitions)
            data, tables, provenance = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=partition_paths,
                results=results,
                num_position_bins=10,
            )
            output_dir = root / "bundle"
            write_bundle(
                output_dir=output_dir,
                plot_data=data,
                tables=tables,
                provenance=provenance,
                input_paths=[*partition_paths.values(), *result_paths],
            )

            niah = data["points"][-1]
            self.assertEqual(niah["dataset_id"], "niah")
            self.assertEqual(niah["status"], "not-run")
            for relative in (
                "bundle_manifest.json",
                "schema.md",
                "completion_report.md",
                "data/candidate_inventory.csv",
                "data/behavior_measurements.csv",
                "data/position_measurements.csv",
                "data/region_measurements.csv",
                "data/interior_attention_error_measurements.csv",
                "data/resolved_configs.json",
                "data/plot_data.json",
                "data/statistics.json",
                "data/provenance.jsonl",
            ):
                self.assertTrue((output_dir / relative).is_file(), relative)
            report = (output_dir / "completion_report.md").read_text()
            self.assertIn("Export status: partial", report)
            self.assertIn("Overall evidence status: pending", report)
            manifest = json.loads((output_dir / "bundle_manifest.json").read_text())
            self.assertIn("schema.md", manifest["artifacts_at_export"])
            self.assertIn("completion_report.md", manifest["artifacts_at_export"])

    def test_result_generation_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            path, payload = results[("biography", "sempic")]
            payload["config"]["model"]["generation_kwargs"]["max_new_tokens"] = 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}

            with self.assertRaisesRegex(ValueError, "model config does not match"):
                build_fixture(
                    root=root,
                    partitions=partitions,
                    partition_paths=write_partitions(root, partitions),
                    results=results,
                    num_position_bins=10,
                )

    def test_top_level_seed_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            path, payload = results[("biography", "sempic")]
            payload["config"]["seed"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}

            with self.assertRaisesRegex(ValueError, "top-level seed"):
                build_fixture(
                    root=root,
                    partitions=partitions,
                    partition_paths=write_partitions(root, partitions),
                    results=results,
                    num_position_bins=10,
                )

    def test_short_partition_is_explicitly_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            partitions["biography"]["samples"] = partitions["biography"]["samples"][:1]

            data, _, _ = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=write_partitions(root, partitions),
                results=results,
                num_position_bins=10,
            )

            point = data["points"][0]
            self.assertEqual(point["status"], "blocked")
            self.assertIn("actual sample count 1", point["status_reason"])
            self.assertIn("frozen expected count 2", point["status_reason"])

    def test_legacy_partition_without_eval_seed_is_loaded_but_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {
                dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS
            }
            processed_metrics, processed_metrics_path = write_processed(root, partitions)
            partitions["biography"]["partition_identity"].pop("eval_seed")
            partition_paths = write_partitions(root, partitions)
            partitions["biography"] = _load_sink_partition(
                partition_paths["biography"]
            )

            data, _, _ = build_plot_data(
                model_id=MODEL_ID,
                partitions=partitions,
                partition_paths=partition_paths,
                results=results,
                processed_metrics=processed_metrics,
                processed_metrics_path=processed_metrics_path,
                num_position_bins=10,
            )

            point = data["points"][0]
            self.assertEqual(point["status"], "blocked")
            self.assertIn("does not record top-level eval_seed", point["status_reason"])

    def test_attention_provenance_separates_partition_from_quality_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            partition_paths = write_partitions(root, partitions)

            _, _, provenance = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=partition_paths,
                results=results,
                num_position_bins=10,
            )

            record = next(
                item
                for item in provenance
                if item["measurement_id"] == "position:biography:sempic:0"
            )
            partition_path = partition_paths["biography"]
            expected_hash = hashlib.sha256(partition_path.read_bytes()).hexdigest()
            quality_path = results[("biography", "sempic")][0]
            self.assertEqual(record["result_path"], str(partition_path.resolve()))
            self.assertEqual(record["source_artifact_sha256"], expected_hash)
            self.assertEqual(
                record["paired_quality_result_path"], str(quality_path.resolve())
            )
            self.assertNotEqual(record["result_path"], record["paired_quality_result_path"])
            self.assertIn("not recorded", record["source_hardware_and_runtime"])
            self.assertIn("CPU export", record["export_hardware_and_runtime"])
            relative_records = [
                item
                for item in provenance
                if ":relative_interior_attention_error:" in item["measurement_id"]
            ]
            self.assertEqual(len(relative_records), 8)
            self.assertTrue(
                all("R_int(method)" in item["metric_definition"] for item in relative_records)
            )
            self.assertTrue(
                all("C_int" not in item["metric_definition"] for item in relative_records)
            )
            recovery_records = [
                item
                for item in provenance
                if item["measurement_id"].endswith(":recovery_fraction")
            ]
            self.assertEqual(len(recovery_records), 4)
            self.assertTrue(
                all(
                    item["metric_definition"].startswith("Recovery=(")
                    for item in recovery_records
                )
            )
            self.assertTrue(
                all("R=(F1_" not in item["metric_definition"] for item in recovery_records)
            )

    def test_plot_schema_recomputes_derived_estimands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            data, _, _ = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=write_partitions(root, partitions),
                results=results,
                num_position_bins=10,
            )
            data["points"][0]["recovery_fraction"] = 0.1

            with self.assertRaisesRegex(ValueError, "does not recompute"):
                validate_plot_data(data)

    def test_relative_interior_error_rejects_tampering_and_ratio_sem(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            data, _, _ = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=write_partitions(root, partitions),
                results=results,
            )

            tampered = copy.deepcopy(data)
            tampered["points"][0]["relative_interior_attention_errors"][0]["value"] = 1.0
            with self.assertRaisesRegex(ValueError, "does not recompute"):
                validate_plot_data(tampered)

            extra_uncertainty = copy.deepcopy(data)
            extra_uncertainty["points"][0]["relative_interior_attention_errors"][0][
                "ratio_sem"
            ] = 0.1
            with self.assertRaisesRegex(ValueError, "invalid field set"):
                validate_plot_data(extra_uncertainty)

    def test_processed_source_fingerprint_and_count_must_match_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            partition_paths = write_partitions(root, partitions)
            processed, processed_path = write_processed(root, partitions)

            fingerprint_mismatch = copy.deepcopy(processed)
            fingerprint_mismatch["source_partitions"][0]["partition_fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "do not exactly match"):
                build_plot_data(
                    model_id=MODEL_ID,
                    partitions=partitions,
                    partition_paths=partition_paths,
                    results=results,
                    processed_metrics=fingerprint_mismatch,
                    processed_metrics_path=processed_path,
                )

            count_mismatch = copy.deepcopy(processed)
            record = next(
                item
                for item in count_mismatch["records"]
                if item["dataset_id"] == "biography"
                and item["method_key"] == "sempic"
                and item["metric_key"] == "attention_absolute_deviation"
                and item["view_key"] == "global_bar"
                and item["facets"]
                == {
                    "attention_view": "raw",
                    "edge_ratio": "0.1",
                    "region": "interior",
                }
            )
            record["count"] = torch.tensor(1, dtype=torch.int64)
            with self.assertRaisesRegex(ValueError, "does not match partition sample count"):
                build_plot_data(
                    model_id=MODEL_ID,
                    partitions=partitions,
                    partition_paths=partition_paths,
                    results=results,
                    processed_metrics=count_mismatch,
                    processed_metrics_path=processed_path,
                )

    def test_nonpositive_vanilla_error_exports_null_ratio_with_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            partition_paths = write_partitions(root, partitions)
            processed, processed_path = write_processed(root, partitions)
            record = next(
                item
                for item in processed["records"]
                if item["dataset_id"] == "biography"
                and item["method_key"] == "vanilla_pic"
                and item["metric_key"] == "attention_absolute_deviation"
                and item["view_key"] == "global_bar"
                and item["facets"]
                == {
                    "attention_view": "raw",
                    "edge_ratio": "0.1",
                    "region": "interior",
                }
            )
            record["mean"] = torch.tensor(0.0, dtype=torch.float64)
            data, _, _ = build_plot_data(
                model_id=MODEL_ID,
                partitions=partitions,
                partition_paths=partition_paths,
                results=results,
                processed_metrics=processed,
                processed_metrics_path=processed_path,
            )

            relative = data["points"][0]["relative_interior_attention_errors"]
            self.assertTrue(all(row["value"] is None for row in relative))
            self.assertTrue(
                all(row["status"] == "undefined_nonpositive_denominator" for row in relative)
            )

    def test_plotter_uses_plot_data_only_and_writes_three_diagnostic_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, _ = write_results(root)
            partitions = {dataset_id: partition(dataset_id) for dataset_id in EXPECTED_DATASETS}
            data, _, _ = build_fixture(
                root=root,
                partitions=partitions,
                partition_paths=write_partitions(root, partitions),
                results=results,
                num_position_bins=10,
            )
            plot_data_path = root / "plot_data.json"
            plot_data_path.write_text(json.dumps(data), encoding="utf-8")
            outputs = plot_diagnostic(
                data,
                root / "figures",
                plot_data_path=plot_data_path,
            )

            self.assertEqual(
                {path.name for path in outputs},
                {f"{OUTPUT_STEM}.svg", f"{OUTPUT_STEM}.pdf", f"{OUTPUT_STEM}.png"},
            )
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            status = json.loads(
                (root / "figures" / f"{OUTPUT_STEM}_status.json").read_text()
            )
            self.assertEqual(status["status"], "generated_unverified")
            self.assertEqual(status["plot_data_path"], str(plot_data_path.resolve()))
            self.assertEqual(
                status["plot_data_sha256"],
                hashlib.sha256(plot_data_path.read_bytes()).hexdigest(),
            )
            svg = (root / "figures" / f"{OUTPUT_STEM}.svg").read_text(encoding="utf-8")
            for text in (
                "Biography",
                "HotpotQA",
                "MuSiQue",
                "NIAH",
                "Sink",
                "Pre÷Int.",
                "ΔF1",
            ):
                self.assertIn(text, svg)


if __name__ == "__main__":
    unittest.main()
