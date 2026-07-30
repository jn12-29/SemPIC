import json
import math
import tempfile
import unittest
from copy import deepcopy
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from sempic.eval_dashboard.aggregate import (
    build_metric_table,
    build_run_leaderboard,
    checkpoint_options,
    frame_to_csv,
    frame_to_latex,
    frame_to_markdown,
    records_to_frame,
    summarize_results,
    summarize_results_with_provenance,
)
from sempic.eval_dashboard.checkpoint import derive_checkpoint_metadata
from sempic.eval_dashboard.charts import (
    ExactChartTooLarge,
    algorithm_color,
    build_cross_dataset_figure,
    build_run_metric_figure,
    build_run_tradeoff_figure,
    series_color,
)
from sempic.eval_dashboard.loader import (
    discover_result_directories,
    load_result_directories,
)
from sempic.eval_dashboard.metrics import (
    metric_is_lower_better,
    metric_label,
    metric_numeric,
    metric_options,
    metric_unit,
)
from sempic.eval_dashboard.schema import normalize_result
from sempic.eval_dashboard.query import QuerySpec, apply_query
from sempic.eval_dashboard.state import (
    SelectionState,
    reconcile_multiselect,
    transition_directory_selection,
)
from sempic.eval_dashboard.views.common import render_exports


def result_payload(*, seven_metrics=False, method="kv_packet"):
    result = {
        "precision": 0.5,
        "recall": 0.25,
        "f1": 1 / 3,
        "ttft": 0.1,
        "flops": 100.0,
    }
    if seven_metrics:
        result.update(num_orig_tokens=1000, num_wrapped_tokens=1100)
    return {
        "config": {
            "model": {
                "model_path": "/models/example/model",
                "dtype": "bfloat16",
                "device": "cuda:0",
                "generation_kwargs": {"max_new_tokens": 32, "do_sample": False},
            },
            "dataset": {
                "dataset_name": "biography",
                "num_samples": 100,
                "subset": "10k",
                "split": "test",
                "seed": 42,
                "data_kwargs": {"question_type": "QA"},
            },
            "cache_comb": {"method": method, "kwargs": {}},
            "packet_wrapper": "./wrapper.pt",
            "compress": None,
            "quantization": None,
            "seed": 42,
        },
        "result": result,
    }


def write_result(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def exact_chart_frame(count: int) -> pd.DataFrame:
    indexes = range(count)
    return pd.DataFrame(
        {
            "source_path": [f"/results/{index:05d}.json" for index in indexes],
            "run_label": [f"run-{index:05d}" for index in indexes],
            "checkpoint_id": [f"checkpoint-{index:05d}" for index in indexes],
            "checkpoint_label": [f"checkpoint-{index:05d}" for index in indexes],
            "method": ["sempic"] * count,
            "method_label": ["SemPIC"] * count,
            "series_id": ["series"] * count,
            "comparison_id": ["comparison"] * count,
            "comparison_label": ["Model · Dataset"] * count,
            "dataset_name": [f"dataset-{index % 2}" for index in indexes],
            "benchmark_label": [f"Dataset {index % 2}" for index in indexes],
            "metric.f1": [0.5 + index / max(count, 1) / 10 for index in indexes],
            "metric.ttft_mean": [0.1 + index / max(count, 1) for index in indexes],
        }
    )


class EvalDashboardLoaderTests(unittest.TestCase):
    def test_overlapping_roots_and_source_paths_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_dir = root / "nested" / "eval_results"
            source = result_dir / "run_result.json"
            write_result(source, result_payload())

            discovery = discover_result_directories([root, root / "nested", root])
            loaded = load_result_directories(
                [result_dir, result_dir, result_dir / ".." / "eval_results"]
            )

            self.assertEqual(discovery.directories, (result_dir.resolve(),))
            self.assertEqual(len(loaded.records), 1)
            self.assertEqual(loaded.records[0].source_path, source.resolve())

    def test_bad_empty_and_metricless_files_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            (result_dir / "bad_result.json").write_text("{", encoding="utf-8")
            write_result(result_dir / "empty_result.json", {})
            metricless = result_payload()
            metricless["result"] = {"note": "not measured", "ok": True}
            write_result(result_dir / "metricless_result.json", metricless)
            write_result(result_dir / "good_result.json", result_payload())

            loaded = load_result_directories([result_dir])

            self.assertEqual([record.run_label for record in loaded.records], ["good"])
            self.assertEqual(len(loaded.warnings), 3)
            self.assertTrue(any("bad_result.json" in warning for warning in loaded.warnings))
            self.assertTrue(any("empty_result.json" in warning for warning in loaded.warnings))
            self.assertTrue(any("no valid numeric metrics" in warning for warning in loaded.warnings))

    def test_unresolvable_source_and_oversized_integer_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            (result_dir / "loop_result.json").symlink_to("loop_result.json")
            oversized = result_payload()
            oversized["result"] = {"oversized": 10**400}
            write_result(result_dir / "oversized_result.json", oversized)
            write_result(result_dir / "good_result.json", result_payload())

            loaded = load_result_directories([result_dir])
            frame = records_to_frame(loaded.records)

            self.assertEqual([record.run_label for record in loaded.records], ["good"])
            self.assertEqual(len(frame), 1)
            self.assertTrue(any("loop_result.json" in item for item in loaded.warnings))
            self.assertTrue(any("oversized_result.json" in item for item in loaded.warnings))

    def test_nonfinite_config_is_rejected_but_invalid_result_metric_is_omitted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir)
            invalid_config = result_payload()
            invalid_config["config"]["model"]["generation_kwargs"]["temperature"] = math.inf
            write_result(result_dir / "config_result.json", invalid_config)
            degraded_result = result_payload()
            degraded_result["result"].update(
                extra=math.nan,
                description="raw",
                numeric_string="3.5",
            )
            write_result(result_dir / "degraded_result.json", degraded_result)

            loaded = load_result_directories([result_dir])

            self.assertEqual(len(loaded.records), 1)
            self.assertNotIn("extra", loaded.records[0].metrics)
            self.assertNotIn("numeric_string", loaded.records[0].metrics)
            self.assertIn("description", loaded.records[0].result)
            self.assertTrue(any("config is not canonical JSON" in item for item in loaded.warnings))
            self.assertTrue(any("metric 'extra'" in item for item in loaded.warnings))

    def test_legacy_five_and_current_seven_metrics_keep_missing_values_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            legacy_dir = root / "legacy"
            current_dir = root / "current"
            legacy = result_payload()
            current = result_payload(seven_metrics=True)
            current["config"]["packet_wrapper"] = {"path": "./wrapper.pt"}
            current["config"]["lora"] = None
            write_result(legacy_dir / "run_result.json", legacy)
            write_result(current_dir / "run_result.json", current)

            frame = records_to_frame(
                load_result_directories([legacy_dir, current_dir]).records
            )
            summary = summarize_results(frame)

            self.assertEqual(len(frame), 2)
            legacy_row = frame.loc[frame["source_path"].str.contains("/legacy/")].iloc[0]
            self.assertTrue(pd.isna(legacy_row["metric.num_orig_tokens"]))
            self.assertEqual(
                frame.loc[
                    frame["source_path"].str.contains("/current/"),
                    "metric.num_wrapped_tokens",
                ].iloc[0],
                1100,
            )
            counts = summary.set_index("metric")["count"]
            self.assertEqual(counts["metric.f1"], 2)
            self.assertEqual(counts["metric.num_orig_tokens"], 1)


class EvalDashboardIdentityTests(unittest.TestCase):
    def _normalize(self, path: Path, payload):
        return normalize_result(path, payload, modified_at=1.0)[0]

    def test_kv_alias_and_legacy_artifact_shape_share_series_identity(self):
        legacy = result_payload(method="kv_packet")
        current = result_payload(method="kvpacket")
        current["config"]["packet_wrapper"] = {"path": "./wrapper.pt"}

        first = self._normalize(Path("/tmp/a/run_result.json"), legacy)
        second = self._normalize(Path("/tmp/b/run_result.json"), current)

        self.assertEqual(first.method, "kvpacket")
        self.assertEqual(first.series_id, second.series_id)
        self.assertNotEqual(first.method_raw, second.method_raw)

    def test_seed_and_device_changes_aggregate_but_conditions_separate(self):
        base = result_payload()
        repeated = deepcopy(base)
        repeated["config"]["seed"] = 7
        repeated["config"]["dataset"]["seed"] = 9
        repeated["config"]["model"]["device"] = "cuda:3"
        different_dataset = deepcopy(base)
        different_dataset["config"]["dataset"]["subset"] = "100k"
        different_generation = deepcopy(base)
        different_generation["config"]["model"]["generation_kwargs"]["max_new_tokens"] = 64
        different_dtype = deepcopy(base)
        different_dtype["config"]["model"]["dtype"] = "float32"

        records = [
            self._normalize(Path("/tmp/a/run_result.json"), base),
            self._normalize(Path("/tmp/b/run_result.json"), repeated),
            self._normalize(Path("/tmp/c/run_result.json"), different_dataset),
            self._normalize(Path("/tmp/d/run_result.json"), different_generation),
            self._normalize(Path("/tmp/e/run_result.json"), different_dtype),
        ]

        self.assertEqual(records[0].comparison_id, records[1].comparison_id)
        self.assertEqual(records[0].series_id, records[1].series_id)
        self.assertEqual(len({record.comparison_id for record in records}), 4)
        summary = summarize_results(records_to_frame(records[:2]))
        f1 = summary.loc[summary["metric"] == "metric.f1"].iloc[0]
        self.assertEqual(f1["count"], 2)
        self.assertAlmostEqual(f1["mean"], 1 / 3)
        self.assertEqual(f1["std"], 0.0)

    def test_algorithm_conditions_and_run_label_separate_series(self):
        base = result_payload()
        other_wrapper = deepcopy(base)
        other_wrapper["config"]["packet_wrapper"] = "./other.pt"

        first = self._normalize(Path("/tmp/a/run_result.json"), base)
        artifact = self._normalize(Path("/tmp/b/run_result.json"), other_wrapper)
        label = self._normalize(Path("/tmp/c/alternate_result.json"), base)

        self.assertEqual(first.comparison_id, artifact.comparison_id)
        self.assertNotEqual(first.series_id, artifact.series_id)
        self.assertNotEqual(first.series_id, label.series_id)


class EvalDashboardCheckpointTests(unittest.TestCase):
    def _payload(
        self,
        *,
        dataset="biography",
        packet_path=(
            "./train_outputs/model/biography/kvpacket/"
            "20260722_161500/packet_wrapper.pt"
        ),
        lora_path=None,
        method="kv_packet",
        f1=0.5,
    ):
        payload = result_payload(method=method)
        payload["config"]["dataset"]["dataset_name"] = dataset
        payload["config"]["packet_wrapper"] = {"path": packet_path}
        payload["config"]["lora"] = {"path": lora_path}
        payload["result"]["f1"] = f1
        return payload

    def _record(self, path, **kwargs):
        return normalize_result(path, self._payload(**kwargs), modified_at=1.0)[0]

    def test_checkpoint_scope_path_boundaries_and_exact_raw_identity(self):
        matched = self._record("/tmp/matched/run_result.json")
        cross = self._record(
            "/tmp/cross/run_result.json",
            dataset="niah",
        )
        windows = self._record(
            "/tmp/windows/run_result.json",
            packet_path=(
                r"C:\repo\train_outputs\model\biography\kvpacket\20260722_161500\packet_wrapper.pt"
            ),
        )
        windows_posix = self._record(
            "/tmp/windows-posix/run_result.json",
            packet_path=(
                "C:/repo/train_outputs/model/biography/kvpacket/"
                "20260722_161500/packet_wrapper.pt"
            ),
        )
        direct = self._record(
            "/tmp/direct/run_result.json",
            packet_path="./train_outputs/model/biography/kvpacket/packet_wrapper.pt",
        )
        timestamped = self._record(
            "/tmp/timestamped/run_result.json",
            packet_path=(
                "./train_outputs/model/biography/kvpacket/"
                "20260722_161500_trial/packet_wrapper.pt"
            ),
        )
        legacy = self._record(
            "/tmp/legacy/run_result.json",
            packet_path="./packet_wrapper/model/biography/8_8.pt",
        )
        unrelated = self._record(
            "/tmp/unrelated/run_result.json",
            packet_path="./artifacts/model/biography/8_8.pt",
        )
        repeated_marker = self._record(
            "/tmp/repeated/run_result.json",
            packet_path=(
                "./train_outputs/model/biography/archive/"
                "train_outputs/model/biography/kvpacket/packet_wrapper.pt"
            ),
        )

        self.assertEqual(matched.checkpoint_scope, "matched")
        self.assertEqual(cross.checkpoint_scope, "cross_dataset")
        self.assertEqual(windows.checkpoint_source_dataset, "biography")
        self.assertEqual(windows.checkpoint_scope, "matched")
        self.assertEqual(direct.checkpoint_scope, "matched")
        self.assertEqual(timestamped.checkpoint_scope, "matched")
        self.assertEqual(legacy.checkpoint_source_dataset, "biography")
        self.assertEqual(legacy.checkpoint_scope, "matched")
        self.assertNotEqual(windows.checkpoint_id, windows_posix.checkpoint_id)
        self.assertEqual(windows.algorithm_variant_id, windows_posix.algorithm_variant_id)
        self.assertEqual(unrelated.checkpoint_scope, "unresolved")
        self.assertEqual(repeated_marker.checkpoint_scope, "unresolved")

    def test_run_selectors_preserve_checkpoint_identity_and_normalize_variant(self):
        paths = (
            (
                "./train_outputs/model/biography/kvpacket/"
                "20260722_161500/packet_wrapper.pt"
            ),
            (
                "./train_outputs/model/biography/kvpacket/"
                "20260722_161500_trial_1/packet_wrapper.pt"
            ),
            "./train_outputs/model/biography/kvpacket/packet_wrapper.pt",
        )
        records = [
            self._record(f"/tmp/selector-{index}/run_result.json", packet_path=path)
            for index, path in enumerate(paths)
        ]

        self.assertEqual(len({record.checkpoint_id for record in records}), len(paths))
        self.assertEqual(len({record.algorithm_variant_id for record in records}), 1)
        self.assertEqual(len({record.algorithm_variant_label for record in records}), 1)

    def test_exact_latest_is_not_normalized_as_run_selector(self):
        latest = self._record(
            "/tmp/latest/run_result.json",
            packet_path=(
                "./train_outputs/model/biography/kvpacket/latest/packet_wrapper.pt"
            ),
        )
        concrete = self._record(
            "/tmp/concrete/run_result.json",
            packet_path=(
                "./train_outputs/model/biography/kvpacket/"
                "20260722_161500/packet_wrapper.pt"
            ),
        )

        self.assertNotEqual(latest.algorithm_variant_id, concrete.algorithm_variant_id)
        self.assertIn("latest/packet_wrapper.pt", latest.algorithm_variant_label)

    def test_joint_paths_agree_and_conflicts_do_not(self):
        joint = self._record(
            "/tmp/joint/run_result.json",
            method="sempic_kvpacket",
            packet_path=(
                "./train_outputs/model/biography/"
                "joint/"
                "20260722_161500_joint/packet_wrapper.pt"
            ),
            lora_path=(
                "./train_outputs/model/biography/"
                "joint/"
                "20260722_161500_joint/lora"
            ),
        )
        conflict = self._record(
            "/tmp/conflict/run_result.json",
            method="sempic_kvpacket",
            packet_path=(
                "./train_outputs/model/biography/"
                "joint/packet_wrapper.pt"
            ),
            lora_path=(
                "./train_outputs/model/niah/"
                "joint/lora"
            ),
        )
        legacy_joint = self._record(
            "/tmp/legacy-joint/run_result.json",
            method="sempic_kvpacket",
            packet_path="./packet_wrapper/model/biography_joint/8_8.pt",
            lora_path="./lora_cache/model/biography/rank8/lora_kv_cache",
        )
        independent = self._record(
            "/tmp/independent/run_result.json",
            method="sempic_kvpacket",
            packet_path=(
                "./train_outputs/model/biography/kvpacket/"
                "20260722_161500_packet/packet_wrapper.pt"
            ),
            lora_path=(
                "./train_outputs/model/biography/"
                "sempic/20260722_162000_lora/lora"
            ),
        )

        self.assertEqual(joint.checkpoint_source_dataset, "biography")
        self.assertEqual(joint.checkpoint_scope, "matched")
        self.assertEqual(legacy_joint.checkpoint_source_dataset, "biography")
        self.assertEqual(legacy_joint.checkpoint_scope, "matched")
        self.assertEqual(independent.checkpoint_scope, "matched")
        self.assertNotEqual(joint.algorithm_variant_id, independent.algorithm_variant_id)
        self.assertNotEqual(joint.algorithm_variant_label, independent.algorithm_variant_label)
        self.assertEqual(conflict.checkpoint_scope, "unresolved")
        self.assertIsNone(conflict.checkpoint_source_dataset)

    def test_dataset_templates_align_and_baseline_run_labels_remain_distinct(self):
        biography = self._record("/tmp/a/run_result.json")
        musique = self._record(
            "/tmp/b/run_result.json",
            dataset="musique",
            packet_path="./train_outputs/model/musique/kvpacket/packet_wrapper.pt",
        )
        baseline = self._payload(packet_path=None)
        first = normalize_result(
            "/tmp/base/run_result.json", baseline, modified_at=1.0
        )[0]
        second = normalize_result(
            "/tmp/base/alternate_result.json", baseline, modified_at=1.0
        )[0]

        self.assertNotEqual(biography.checkpoint_id, musique.checkpoint_id)
        self.assertEqual(biography.algorithm_variant_id, musique.algorithm_variant_id)
        self.assertEqual(first.checkpoint_scope, "none")
        self.assertNotEqual(first.algorithm_variant_id, second.algorithm_variant_id)
        self.assertEqual(first.algorithm_variant_label, "run")
        self.assertEqual(second.algorithm_variant_label, "alternate")

    def test_checkpoint_digest_payload_uses_exact_present_raw_paths(self):
        config = self._payload()["config"]
        metadata = derive_checkpoint_metadata(
            config, "biography", "kvpacket", "Kvpacket", "run"
        )
        alias = deepcopy(config)
        alias["packet_wrapper"]["path"] = alias["packet_wrapper"]["path"].replace(
            "./", ""
        )
        changed = derive_checkpoint_metadata(
            alias, "biography", "kvpacket", "Kvpacket", "run"
        )

        self.assertRegex(metadata.checkpoint_id or "", r"^[0-9a-f]{64}$")
        self.assertNotEqual(metadata.checkpoint_id, changed.checkpoint_id)


class EvalDashboardSmartTableTests(unittest.TestCase):
    def _record(
        self,
        path,
        *,
        dataset="biography",
        packet_path=None,
        f1=0.5,
    ):
        payload = result_payload()
        payload["config"]["dataset"]["dataset_name"] = dataset
        payload["config"]["packet_wrapper"] = {"path": packet_path}
        payload["result"]["f1"] = f1
        return normalize_result(path, payload, modified_at=1.0)[0]

    def test_matched_mode_excludes_cross_and_unresolved_with_contract_counts(self):
        records = [
            self._record("/tmp/base/base_result.json"),
            self._record(
                "/tmp/matched/matched_result.json",
                packet_path="./train_outputs/model/biography/kvpacket/packet_wrapper.pt",
            ),
            self._record(
                "/tmp/cross/cross_result.json",
                packet_path="./train_outputs/model/niah/kvpacket/packet_wrapper.pt",
            ),
            self._record(
                "/tmp/unresolved/run_result.json",
                packet_path="./unknown/8_8.pt",
            ),
        ]
        frame = records_to_frame(records)
        result = build_metric_table(frame, "metric.f1", mode="dataset_matched")

        self.assertEqual(result.included_rows, 2)
        self.assertEqual(result.excluded_cross_dataset, 1)
        self.assertEqual(result.excluded_unresolved, 1)
        self.assertEqual(result.table.columns[0], "Comparison")
        self.assertIsInstance(result.table.index, pd.RangeIndex)

    def test_shared_checkpoint_selects_exact_id_across_eval_datasets(self):
        path = "./train_outputs/model/biography/kvpacket/packet_wrapper.pt"
        records = [
            self._record("/tmp/a/run_result.json", packet_path=path, f1=0.4),
            self._record(
                "/tmp/b/run_result.json",
                dataset="niah",
                packet_path=path,
                f1=0.6,
            ),
            self._record(
                "/tmp/c/run_result.json",
                dataset="musique",
                packet_path="./train_outputs/model/musique/kvpacket/packet_wrapper.pt",
                f1=0.8,
            ),
        ]
        frame = records_to_frame(records)
        checkpoint_id = records[0].checkpoint_id
        selected = build_metric_table(
            frame,
            "metric.f1",
            mode="shared_checkpoint",
            checkpoint_id=checkpoint_id,
        )
        missing = build_metric_table(
            frame, "metric.f1", mode="shared_checkpoint", checkpoint_id="missing"
        )
        unset = build_metric_table(frame, "metric.f1", mode="shared_checkpoint")

        self.assertEqual(selected.included_rows, 2)
        self.assertEqual(len(selected.table), 2)
        self.assertEqual(list(missing.table.columns), ["Comparison"])
        self.assertEqual(list(unset.table.columns), ["Comparison"])

    def test_exact_runs_is_safe_default_and_keeps_checkpoints_separate(self):
        records = [
            self._record(
                "/tmp/a/run_result.json",
                packet_path=(
                    "./train_outputs/model/biography/kvpacket/"
                    "20260724_010000_b32-e15/packet_wrapper.pt"
                ),
                f1=0.9,
            ),
            self._record(
                "/tmp/b/run_result.json",
                packet_path=(
                    "./train_outputs/model/biography/kvpacket/"
                    "20260724_020000_b8-e5/packet_wrapper.pt"
                ),
                f1=0.5,
            ),
        ]

        result = build_metric_table(records_to_frame(records), "metric.f1")

        self.assertEqual(result.aggregation_label, "Exact observations")
        self.assertEqual(result.estimator, "None")
        self.assertEqual(result.included_rows, 2)
        self.assertEqual(len(result.included_checkpoints), 2)
        self.assertEqual(result.table["Value"].tolist(), [0.9, 0.5])
        self.assertEqual(result.provenance["included"].tolist(), [True, True])

    def test_rollup_provenance_discloses_checkpoint_membership_and_exclusions(self):
        records = [
            self._record(
                "/tmp/matched/matched_result.json",
                packet_path="./train_outputs/model/biography/kvpacket/run-a/packet_wrapper.pt",
                f1=0.9,
            ),
            self._record(
                "/tmp/cross/cross_result.json",
                packet_path="./train_outputs/model/niah/kvpacket/run-b/packet_wrapper.pt",
                f1=0.4,
            ),
        ]

        result = build_metric_table(
            records_to_frame(records), "metric.f1", mode="dataset_matched"
        )

        self.assertEqual(result.included_rows, 1)
        self.assertEqual(len(result.included_checkpoints), 1)
        reasons = dict(zip(result.provenance["run_label"], result.provenance["reason"]))
        self.assertEqual(reasons["matched"], "included")
        self.assertEqual(reasons["cross"], "cross_dataset_checkpoint")

    def test_table_membership_and_provenance_use_the_same_eligible_rows(self):
        matched_path = (
            "./train_outputs/model/biography/kvpacket/run-a/packet_wrapper.pt"
        )
        records = [
            self._record(
                "/tmp/matched/a_result.json", packet_path=matched_path, f1=0.9
            ),
            self._record(
                "/tmp/matched/b_result.json", packet_path=matched_path, f1=0.7
            ),
            self._record(
                "/tmp/cross/c_result.json",
                packet_path=(
                    "./train_outputs/model/niah/kvpacket/run-b/packet_wrapper.pt"
                ),
                f1=0.4,
            ),
        ]
        frame = records_to_frame(records)
        frame.loc[frame["run_label"] == "b", "metric.f1"] = math.nan
        checkpoint_id = records[0].checkpoint_id

        cases = (
            (
                "exact_runs",
                None,
                {"/tmp/matched/a_result.json", "/tmp/cross/c_result.json"},
                [0.9, 0.4],
            ),
            ("dataset_matched", None, {"/tmp/matched/a_result.json"}, [0.9]),
            (
                "shared_checkpoint",
                checkpoint_id,
                {"/tmp/matched/a_result.json"},
                [0.9],
            ),
        )
        for mode, selected_checkpoint, expected_paths, expected_values in cases:
            with self.subTest(mode=mode):
                result = build_metric_table(
                    frame,
                    "metric.f1",
                    mode=mode,
                    checkpoint_id=selected_checkpoint,
                )
                provenance_paths = {
                    path
                    for path in result.provenance.loc[
                        result.provenance["included"], "source_path"
                    ]
                }
                self.assertEqual(result.included_rows, len(provenance_paths))
                self.assertEqual(
                    {Path(path).as_posix() for path in provenance_paths},
                    expected_paths,
                )
                self.assertEqual(
                    set(result.included_checkpoints),
                    {
                        checkpoint
                        for checkpoint in result.provenance.loc[
                            result.provenance["included"], "checkpoint_id"
                        ]
                        if isinstance(checkpoint, str) and checkpoint
                    },
                )
                if mode == "exact_runs":
                    self.assertEqual(set(result.table["Result path"]), provenance_paths)
                    table_values = result.table["Value"].tolist()
                else:
                    table_values = (
                        result.table.drop(columns="Comparison")
                        .stack()
                        .astype(float)
                        .tolist()
                    )
                self.assertEqual(table_values, expected_values)

    def test_metric_table_uses_logical_ttft_fallback_for_table_and_provenance(self):
        record = self._record(
            "/tmp/matched/run_result.json",
            packet_path=(
                "./train_outputs/model/biography/kvpacket/"
                "run-a/packet_wrapper.pt"
            ),
        )
        frame = records_to_frame([record])
        frame["metric.ttft_mean"] = math.nan
        frame["metric.ttft"] = 0.2

        result = build_metric_table(
            frame,
            "metric.ttft_mean",
            mode="dataset_matched",
        )

        self.assertEqual(result.included_rows, 1)
        self.assertEqual(
            result.table.drop(columns="Comparison").stack().astype(float).tolist(),
            [0.2],
        )
        self.assertEqual(
            result.provenance.loc[
                result.provenance["included"], "metric_value"
            ].tolist(),
            [0.2],
        )

    def test_all_aggregation_modes_have_empty_membership_contracts(self):
        empty = records_to_frame([])
        cases = (
            (
                "exact_runs",
                None,
                [
                    "Comparison",
                    "Method",
                    "Run label",
                    "Checkpoint",
                    "Value",
                    "Result path",
                ],
            ),
            ("dataset_matched", None, ["Comparison"]),
            ("shared_checkpoint", "checkpoint", ["Comparison"]),
        )

        for mode, checkpoint_id, expected_columns in cases:
            with self.subTest(mode=mode):
                result = build_metric_table(
                    empty,
                    "metric.f1",
                    mode=mode,
                    checkpoint_id=checkpoint_id,
                )

                self.assertTrue(result.table.empty)
                self.assertEqual(result.table.columns.tolist(), expected_columns)
                self.assertTrue(result.provenance.empty)
                self.assertEqual(result.included_rows, 0)
                self.assertEqual(result.included_checkpoints, ())
                self.assertEqual(result.excluded_cross_dataset, 0)
                self.assertEqual(result.excluded_unresolved, 0)

    def test_membership_reason_priority_precedes_missing_metric(self):
        matched_path = (
            "./train_outputs/model/biography/kvpacket/run-a/packet_wrapper.pt"
        )
        records = [
            self._record(
                "/tmp/matched/matched_result.json",
                packet_path=matched_path,
                f1=0.9,
            ),
            self._record(
                "/tmp/cross/cross_result.json",
                packet_path=(
                    "./train_outputs/model/niah/kvpacket/run-b/packet_wrapper.pt"
                ),
                f1=0.4,
            ),
        ]
        frame = records_to_frame(records)
        frame["metric.f1"] = math.nan

        exact = build_metric_table(frame, "metric.f1", mode="exact_runs")
        matched = build_metric_table(frame, "metric.f1", mode="dataset_matched")
        shared = build_metric_table(
            frame,
            "metric.f1",
            mode="shared_checkpoint",
            checkpoint_id=records[0].checkpoint_id,
        )
        missing_metric = build_metric_table(
            frame, "metric.not_present", mode="dataset_matched"
        )

        self.assertEqual(set(exact.provenance["reason"]), {"metric_missing_or_non_numeric"})
        matched_reasons = dict(
            zip(matched.provenance["run_label"], matched.provenance["reason"])
        )
        self.assertEqual(matched_reasons["matched"], "metric_missing_or_non_numeric")
        self.assertEqual(matched_reasons["cross"], "cross_dataset_checkpoint")
        shared_reasons = dict(
            zip(shared.provenance["run_label"], shared.provenance["reason"])
        )
        self.assertEqual(shared_reasons["matched"], "metric_missing_or_non_numeric")
        self.assertEqual(shared_reasons["cross"], "checkpoint_mismatch")
        missing_reasons = dict(
            zip(
                missing_metric.provenance["run_label"],
                missing_metric.provenance["reason"],
            )
        )
        self.assertEqual(
            missing_reasons["matched"], "metric_missing_or_non_numeric"
        )
        self.assertEqual(missing_reasons["cross"], "cross_dataset_checkpoint")
        self.assertEqual(missing_metric.included_rows, 0)

    def test_checkpoint_options_are_unique_disambiguated_and_sorted(self):
        frame = pd.DataFrame(
            [
                {
                    "checkpoint_id": "aaaaaaaa1" + "0" * 55,
                    "checkpoint_label": "Same",
                    "method": "kvpacket",
                    "method_label": "Kvpacket",
                    "checkpoint_source_dataset": "biography",
                },
                {
                    "checkpoint_id": "aaaaaaaa0" + "0" * 55,
                    "checkpoint_label": "Same",
                    "method": "kvpacket",
                    "method_label": "Kvpacket",
                    "checkpoint_source_dataset": "biography",
                },
                {
                    "checkpoint_id": "bbbbbbbb" + "0" * 56,
                    "checkpoint_label": "Earlier",
                    "method": "sempic",
                    "method_label": "Sempic",
                    "checkpoint_source_dataset": "niah",
                },
                {
                    "checkpoint_id": "cccccccc" + "0" * 56,
                    "checkpoint_label": "Same [aaaaaaaa0]",
                    "method": "sempic",
                    "method_label": "Sempic",
                    "checkpoint_source_dataset": "musique",
                },
            ]
        )
        options = checkpoint_options(frame)

        self.assertEqual(
            list(options.columns),
            [
                "checkpoint_id",
                "checkpoint_label",
                "method",
                "method_label",
                "checkpoint_source_dataset",
            ],
        )
        self.assertEqual(options.iloc[0]["checkpoint_label"], "Earlier")
        self.assertEqual(options["checkpoint_id"].nunique(), 4)
        self.assertEqual(options["checkpoint_label"].nunique(), 4)
        self.assertTrue(
            all(label.endswith("]") for label in options.iloc[1:]["checkpoint_label"])
        )
        self.assertTrue(any("[aaaaaaaa0]" in label for label in options["checkpoint_label"]))

    def test_comparison_is_exported_as_an_ordinary_first_column(self):
        record = self._record("/tmp/base/base_result.json")
        table = build_metric_table(records_to_frame([record]), "metric.f1").table

        self.assertEqual(table.columns[0], "Comparison")
        self.assertIn("Comparison,", frame_to_csv(table))
        self.assertIn("| Comparison", frame_to_markdown(table))
        self.assertIn("Comparison", frame_to_latex(table))


class EvalDashboardRunLeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            [
                {
                    "comparison_id": "comparison-a",
                    "algorithm_variant_id": "shared-variant",
                    "method_label": "Kvpacket",
                    "run_label": "run-low",
                    "checkpoint_label": "checkpoint-low",
                    "metric.f1": 0.4,
                    "metric.ttft": 0.3,
                    "metric.ttft_mean": 0.2,
                    "metric.ttft_p50": 0.15,
                    "metric.ttft_p99": 0.35,
                    "metric.flops": 20.0,
                    "source_path": "/results/b.json",
                },
                {
                    "comparison_id": "comparison-a",
                    "algorithm_variant_id": "shared-variant",
                    "method_label": "Kvpacket",
                    "run_label": "run-high",
                    "checkpoint_label": "checkpoint-high",
                    "metric.f1": 0.8,
                    "metric.ttft": 0.1,
                    "metric.ttft_mean": math.nan,
                    "metric.ttft_p50": 0.08,
                    "metric.ttft_p99": 0.25,
                    "metric.flops": 10.0,
                    "source_path": "/results/a.json",
                },
                {
                    "comparison_id": "comparison-b",
                    "algorithm_variant_id": "other-variant",
                    "method_label": "Sempic",
                    "run_label": "run-missing",
                    "checkpoint_label": None,
                    "metric.f1": math.nan,
                    "metric.ttft": math.nan,
                    "metric.flops": math.nan,
                    "source_path": "/results/c.json",
                },
            ]
        )
        self.frame.index = [10, 20, 30]

    def test_preserves_distinct_checkpoints_with_the_same_algorithm_variant(self):
        duplicate_identity = self.frame.iloc[[0]].copy()
        duplicate_identity["source_path"] = "/results/d.json"
        frame = pd.concat([self.frame, duplicate_identity], ignore_index=True)
        leaderboard = build_run_leaderboard(frame)

        self.assertEqual(len(leaderboard), len(frame))
        self.assertEqual(
            set(leaderboard.loc[leaderboard["Method"] == "Kvpacket", "Checkpoint"]),
            {"checkpoint-low", "checkpoint-high"},
        )
        self.assertEqual((leaderboard["Run label"] == "run-low").sum(), 2)
        self.assertEqual(leaderboard.iloc[-1]["Checkpoint"], "")

    def test_ttft_mean_prefers_mean_then_falls_back_and_preserves_percentiles(self):
        leaderboard = build_run_leaderboard(self.frame).set_index("Run label")

        self.assertEqual(leaderboard.loc["run-low", "TTFT Mean"], 0.2)
        self.assertEqual(leaderboard.loc["run-high", "TTFT Mean"], 0.1)
        self.assertEqual(leaderboard.loc["run-low", "TTFT P50"], 0.15)
        self.assertEqual(leaderboard.loc["run-high", "TTFT P99"], 0.25)

    def test_legacy_ttft_only_frame_supports_mean_display_and_sort_alias(self):
        legacy = self.frame.drop(columns="metric.ttft_mean")

        leaderboard = build_run_leaderboard(
            legacy,
            comparison_id="comparison-a",
            sort_metric="metric.ttft",
        )

        self.assertEqual(leaderboard["Run label"].tolist(), ["run-high", "run-low"])
        self.assertEqual(leaderboard["TTFT Mean"].tolist(), [0.1, 0.3])

    def test_comparison_filter_is_exact(self):
        leaderboard = build_run_leaderboard(
            self.frame,
            comparison_id="comparison-a",
        )

        self.assertEqual(set(leaderboard["Run label"]), {"run-low", "run-high"})
        self.assertNotIn("run-missing", leaderboard["Run label"].tolist())

    def test_sort_defaults_override_missing_and_path_tie_breaker(self):
        tied = self.frame.copy()
        tied.loc[tied.index[0], "metric.f1"] = 0.8

        quality = build_run_leaderboard(tied)
        self.assertEqual(
            quality["Result path"].tolist(),
            ["/results/a.json", "/results/b.json", "/results/c.json"],
        )

        latency = build_run_leaderboard(self.frame, sort_metric="metric.ttft_mean")
        self.assertEqual(
            latency["Run label"].tolist(),
            ["run-high", "run-low", "run-missing"],
        )

        flops = build_run_leaderboard(self.frame, sort_metric="metric.flops")
        self.assertEqual(
            flops["Run label"].tolist(),
            ["run-high", "run-low", "run-missing"],
        )

        reversed_latency = build_run_leaderboard(
            self.frame,
            sort_metric="metric.ttft_p50",
            ascending=False,
        )
        self.assertEqual(
            reversed_latency["Run label"].tolist(),
            ["run-low", "run-high", "run-missing"],
        )

    def test_empty_input_has_fixed_columns(self):
        leaderboard = build_run_leaderboard(
            pd.DataFrame(),
            comparison_id="comparison-a",
        )

        self.assertEqual(
            leaderboard.columns.tolist(),
            [
                "Method",
                "Run label",
                "Checkpoint",
                "F1",
                "TTFT Mean",
                "TTFT P50",
                "TTFT P99",
                "FLOPs",
                "Result path",
            ],
        )
        self.assertTrue(leaderboard.empty)


class EvalDashboardMetricTests(unittest.TestCase):
    def test_labels_units_and_directions_share_one_contract(self):
        self.assertEqual(metric_label("metric.f1"), "F1")
        self.assertEqual(metric_label("metric.ttft_p99"), "TTFT P99 (s)")
        self.assertEqual(metric_label("metric.custom_score"), "Custom Score")
        self.assertIsNone(metric_unit("metric.f1"))
        self.assertEqual(metric_unit("metric.ttft_mean"), "s")
        self.assertFalse(metric_is_lower_better("metric.f1"))
        self.assertTrue(metric_is_lower_better("metric.ttft_mean"))
        self.assertTrue(metric_is_lower_better("metric.flops"))

    def test_ttft_mean_numeric_falls_back_per_row_without_filling_missing_with_zero(self):
        frame = pd.DataFrame(
            {
                "metric.ttft_mean": [math.nan, 0.2, "bad", math.nan],
                "metric.ttft": [0.1, 0.3, 0.4, math.nan],
            },
            index=[10, 20, 30, 40],
        )

        values = metric_numeric(frame, "metric.ttft_mean")
        missing = metric_numeric(frame, "metric.not_present")

        self.assertEqual(values.loc[[10, 20, 30]].tolist(), [0.1, 0.2, 0.4])
        self.assertTrue(pd.isna(values.loc[40]))
        self.assertEqual(values.index.tolist(), frame.index.tolist())
        self.assertTrue(missing.isna().all())

    def test_metric_options_canonicalize_legacy_ttft_and_keep_numeric_extras(self):
        frame = pd.DataFrame(
            {
                "metric.ttft": [0.1],
                "metric.ttft_mean": [math.nan],
                "metric.f1": [0.5],
                "metric.custom_score": [3.0],
                "run_label": ["run"],
            }
        )

        options = metric_options(frame)

        self.assertEqual(options.count("metric.ttft_mean"), 1)
        self.assertNotIn("metric.ttft", options)
        self.assertIn("metric.f1", options)
        self.assertIn("metric.custom_score", options)


class EvalDashboardAggregateTests(unittest.TestCase):
    def test_extreme_finite_values_use_stable_sample_statistics(self):
        positive = result_payload()
        negative = result_payload()
        positive["result"] = {"extreme": 1e308}
        negative["result"] = {"extreme": -1e308}
        records = [
            normalize_result("/tmp/a/run_result.json", positive, modified_at=1.0)[0],
            normalize_result("/tmp/b/run_result.json", negative, modified_at=1.0)[0],
        ]

        summary = summarize_results(records_to_frame(records))
        extreme = summary.loc[summary["metric"] == "metric.extreme"].iloc[0]

        self.assertEqual(extreme["mean"], 0.0)
        self.assertTrue(math.isfinite(extreme["std"]))
        self.assertAlmostEqual(extreme["std"] / 1e308, math.sqrt(2))

        second_positive = normalize_result(
            "/tmp/c/run_result.json", positive, modified_at=1.0
        )[0]
        same_sign = summarize_results(records_to_frame([records[0], second_positive]))
        same_sign_extreme = same_sign.loc[
            same_sign["metric"] == "metric.extreme"
        ].iloc[0]
        self.assertEqual(same_sign_extreme["mean"], 1e308)

    def test_unknown_metrics_are_prefixed_and_single_sample_std_is_missing(self):
        payload = result_payload()
        payload["result"]["custom_score"] = 4
        record = normalize_result("/tmp/run_result.json", payload, modified_at=1.0)[0]

        frame = records_to_frame([record])
        summary = summarize_results(frame)
        custom = summary.loc[summary["metric"] == "metric.custom_score"].iloc[0]

        self.assertIn("metric.custom_score", frame.columns)
        self.assertEqual(custom["mean"], 4)
        self.assertEqual(custom["count"], 1)
        self.assertTrue(pd.isna(custom["std"]))

    def test_exports_use_the_given_snapshot(self):
        payload = result_payload()
        record = normalize_result("/tmp/run_result.json", payload, modified_at=1.0)[0]
        snapshot = pd.DataFrame([{"name": "only", "value": 3}])

        self.assertIn("name,value", frame_to_csv(snapshot))
        self.assertIn("| name", frame_to_markdown(snapshot))
        self.assertIn("\\begin{tabular}", frame_to_latex(snapshot))
        self.assertNotIn("precision", frame_to_csv(snapshot))
        self.assertIn("config", frame_to_markdown(records_to_frame([record])))

    def test_auditable_summary_discloses_estimator_and_source_membership(self):
        first_payload = result_payload()
        first_payload["result"].update(ttft=0.1, ttft_mean=0.2)
        second_payload = result_payload()
        second_payload["result"].update(ttft=0.3, ttft_mean=0.4)
        records = [
            normalize_result(
                "/tmp/repeat-a/run_result.json",
                first_payload,
                modified_at=1.0,
            )[0],
            normalize_result(
                "/tmp/repeat-b/run_result.json",
                second_payload,
                modified_at=2.0,
            )[0],
        ]

        summary = summarize_results_with_provenance(records_to_frame(records))
        f1 = summary.loc[summary["metric"] == "metric.f1"].iloc[0]

        self.assertEqual(f1["count"], 2)
        self.assertEqual(f1["source_count"], 2)
        self.assertEqual(f1["checkpoint_count"], 1)
        self.assertEqual(
            f1["estimator"],
            "Arithmetic mean; sample standard deviation (n-1)",
        )
        self.assertIn("repeat-a", f1["source_paths"])
        self.assertIn("repeat-b", f1["source_paths"])
        raw_ttft = summary.loc[summary["metric"] == "metric.ttft"].iloc[0]
        mean_ttft = summary.loc[summary["metric"] == "metric.ttft_mean"].iloc[0]
        self.assertAlmostEqual(raw_ttft["mean"], 0.2)
        self.assertAlmostEqual(mean_ttft["mean"], 0.3)


class EvalDashboardExportTests(unittest.TestCase):
    def test_only_prepared_format_is_built_from_lazy_frame_factory(self):
        serializers = {
            "CSV": "frame_to_csv",
            "Markdown": "frame_to_markdown",
            "LaTeX": "frame_to_latex",
        }
        for selected_format, selected_serializer in serializers.items():
            with self.subTest(selected_format=selected_format):
                frame_factory = Mock(
                    return_value=pd.DataFrame([{"name": "run", "value": 1}])
                )
                serializer_mocks = {
                    name: Mock(return_value=f"prepared-{name}")
                    for name in serializers.values()
                }
                with ExitStack() as stack:
                    for name, mock in serializer_mocks.items():
                        stack.enter_context(
                            patch(
                                f"sempic.eval_dashboard.views.common.{name}", mock
                            )
                        )
                    stack.enter_context(
                        patch(
                            "sempic.eval_dashboard.views.common.st.selectbox",
                            return_value=selected_format,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "sempic.eval_dashboard.views.common.st.button",
                            side_effect=[False, True],
                        )
                    )
                    download_button = stack.enter_context(
                        patch(
                            "sempic.eval_dashboard.views.common.st.download_button"
                        )
                    )
                    render_exports(
                        frame_factory,
                        stem="snapshot",
                        key_prefix="export.test",
                    )
                    self.assertEqual(frame_factory.call_count, 0)
                    self.assertFalse(download_button.called)

                    render_exports(
                        frame_factory,
                        stem="snapshot",
                        key_prefix="export.test",
                    )

                self.assertEqual(frame_factory.call_count, 1)
                for name, serializer in serializer_mocks.items():
                    self.assertEqual(
                        serializer.call_count,
                        int(name == selected_serializer),
                    )
                download_button.assert_called_once()
                _, kwargs = download_button.call_args
                self.assertEqual(kwargs["on_click"], "ignore")
                self.assertEqual(kwargs["data"], f"prepared-{selected_serializer}")


class EvalDashboardQueryTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "run_label": ["keep-one", "keep-two", "drop-one"],
                "source_path": [
                    "/results/a/first_result.json",
                    "/results/b/second_result.json",
                    "/results/a/third_result.json",
                ],
                "score": pd.Series([1, 2, 3], index=[10, 20, 30], dtype="int64"),
            },
            index=[10, 20, 30],
        )

    def test_patterns_search_case_sensitively_and_support_inline_flags_and_groups(self):
        frame = pd.DataFrame(
            {
                "run_label": ["prefix-B32-suffix", "prefix-b16-suffix"],
                "source_path": ["/results/first.json", "/results/second.json"],
            }
        )

        result = apply_query(
            frame, QuerySpec(run_label_pattern="(b32|b16)")
        )
        self.assertEqual(result.errors, ())
        self.assertEqual(result.frame["run_label"].tolist(), ["prefix-b16-suffix"])

        insensitive = apply_query(
            frame, QuerySpec(run_label_pattern="(?i)(b32|b16)")
        )
        self.assertEqual(insensitive.errors, ())
        self.assertEqual(
            insensitive.frame["run_label"].tolist(), frame["run_label"].tolist()
        )

        sensitive_path = apply_query(
            frame, QuerySpec(source_path_pattern=r"(FIRST|second)")
        )
        self.assertEqual(sensitive_path.errors, ())
        self.assertEqual(
            sensitive_path.frame["source_path"].tolist(), ["/results/second.json"]
        )

        insensitive_path = apply_query(
            frame, QuerySpec(source_path_pattern=r"(?i)(FIRST|second)")
        )
        self.assertEqual(insensitive_path.errors, ())
        self.assertEqual(
            insensitive_path.frame["source_path"].tolist(),
            frame["source_path"].tolist(),
        )

    def test_run_label_and_source_path_patterns_are_combined_with_and(self):
        result = apply_query(
            self.frame,
            QuerySpec(run_label_pattern="keep", source_path_pattern=r"/a/"),
        )

        self.assertEqual(result.errors, ())
        self.assertEqual(result.frame.index.tolist(), [10])

        unmatched = apply_query(
            self.frame, QuerySpec(run_label_pattern="missing")
        )
        self.assertEqual(unmatched.errors, ())
        self.assertTrue(unmatched.frame.empty)

    def test_none_and_empty_patterns_are_noops_but_whitespace_is_not_stripped(self):
        unchanged = apply_query(
            self.frame,
            QuerySpec(run_label_pattern=None, source_path_pattern=""),
        )
        self.assertEqual(unchanged.errors, ())
        pd.testing.assert_frame_equal(unchanged.frame, self.frame)

        frame = self.frame.copy()
        frame.loc[10, "run_label"] = "keep one"
        filtered = apply_query(frame, QuerySpec(run_label_pattern=" "))
        self.assertEqual(filtered.errors, ())
        self.assertEqual(filtered.frame.index.tolist(), [10])

    def test_only_string_values_participate_in_matching(self):
        frame = pd.DataFrame(
            {
                "run_label": [None, math.nan, pd.NA, 123, "nan"],
                "source_path": ["/kept/0", "/kept/1", "/kept/2", "/kept/3", "/kept/4"],
            }
        )

        filtered = apply_query(
            frame, QuerySpec(run_label_pattern=r"nan|123")
        )
        self.assertEqual(filtered.errors, ())
        self.assertEqual(filtered.frame.index.tolist(), [4])

        frame["source_path"] = [None, math.nan, pd.NA, 123, "/kept/4"]
        filtered = apply_query(
            frame, QuerySpec(source_path_pattern=r"nan|123|kept")
        )
        self.assertEqual(filtered.errors, ())
        self.assertEqual(filtered.frame.index.tolist(), [4])

    def test_invalid_patterns_return_empty_copy_and_report_each_field(self):
        result = apply_query(
            self.frame,
            QuerySpec(run_label_pattern="(", source_path_pattern="["),
        )

        pd.testing.assert_frame_equal(result.frame, self.frame.iloc[0:0])
        self.assertIsInstance(result.errors, tuple)
        self.assertEqual(len(result.errors), 2)
        self.assertTrue(any("Run label" in error for error in result.errors))
        self.assertTrue(any("Result path" in error for error in result.errors))

    def test_invalid_pattern_is_reported_before_filtering_even_for_empty_input(self):
        result = apply_query(
            self.frame,
            QuerySpec(run_label_pattern="does-not-match", source_path_pattern="("),
        )
        self.assertTrue(result.frame.empty)
        self.assertIsInstance(result.errors, tuple)
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(any("Result path" in error for error in result.errors))

        empty = self.frame.iloc[0:0]
        empty_result = apply_query(empty, QuerySpec(run_label_pattern="("))
        pd.testing.assert_frame_equal(empty_result.frame, empty)
        self.assertIsNot(empty_result.frame, empty)
        self.assertIsInstance(empty_result.errors, tuple)
        self.assertEqual(len(empty_result.errors), 1)
        self.assertTrue(any("Run label" in error for error in empty_result.errors))

    def test_regex_repetition_overflow_is_reported(self):
        result = apply_query(
            self.frame,
            QuerySpec(run_label_pattern="a{9999999999999999999999999999999}"),
        )

        self.assertTrue(result.frame.empty)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Run label", result.errors[0])

    def test_typed_query_combines_exact_scope_and_regex(self):
        frame = self.frame.assign(
            model_name=["Qwen", "Qwen", "Llama"],
            method=["kvpacket", "sempic", "kvpacket"],
            dataset_name=["biography", "biography", "niah"],
            benchmark_label=["biography", "biography", "niah"],
        )

        result = apply_query(
            frame,
            QuerySpec(
                models=("Qwen",),
                methods=("kvpacket",),
                run_label_pattern="keep",
            ),
        )

        self.assertEqual(result.errors, ())
        self.assertEqual(result.frame.index.tolist(), [10])


class EvalDashboardSelectionStateTests(unittest.TestCase):
    def test_custom_selection_does_not_widen_after_options_shrink_and_return(self):
        customized = reconcile_multiselect(
            ["a", "b"],
            None,
            action="customize",
            user_selection=["a"],
        )
        shrunk = reconcile_multiselect(["a"], customized)
        restored = reconcile_multiselect(["a", "b"], shrunk)

        self.assertEqual(restored, SelectionState(selected=("a",), follow_all=False))

    def test_follow_all_tracks_new_options_but_clear_remains_empty(self):
        initial = reconcile_multiselect(["a"], None)
        followed = reconcile_multiselect(["a", "b"], initial)
        cleared = reconcile_multiselect(["a", "b"], followed, action="clear")
        refreshed = reconcile_multiselect(["a", "b", "c"], cleared)

        self.assertEqual(followed.selected, ("a", "b"))
        self.assertTrue(followed.follow_all)
        self.assertEqual(refreshed.selected, ())
        self.assertFalse(refreshed.follow_all)

    def test_directory_selection_resolves_paths_and_preserves_selection_intent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first"
            second = root / "second"
            third = root / "third"
            first.mkdir()
            second.mkdir()
            third.mkdir()

            initial = transition_directory_selection([first, first, second], None)
            shrunk = transition_directory_selection([first], initial)
            restored = transition_directory_selection([first, second], shrunk)
            customized = transition_directory_selection(
                [first, second],
                restored,
                action="customize",
                user_selection=[second],
            )
            custom_shrunk = transition_directory_selection([first], customized)
            custom_restored = transition_directory_selection(
                [first, second], custom_shrunk
            )
            selected_all = transition_directory_selection(
                [first, second, third], custom_restored, action="select_all"
            )
            cleared = transition_directory_selection(
                [first, second, third], selected_all, action="clear"
            )

            resolved_first = str(first.resolve())
            resolved_second = str(second.resolve())
            resolved_third = str(third.resolve())
            self.assertEqual(initial.selected, (resolved_first, resolved_second))
            self.assertTrue(initial.follow_all)
            self.assertEqual(shrunk.selected, (resolved_first,))
            self.assertTrue(shrunk.follow_all)
            self.assertEqual(restored.selected, (resolved_first, resolved_second))
            self.assertTrue(restored.follow_all)
            self.assertEqual(customized.selected, (resolved_second,))
            self.assertFalse(customized.follow_all)
            self.assertEqual(custom_shrunk.selected, ())
            self.assertEqual(custom_restored.selected, ())
            self.assertEqual(
                selected_all.selected,
                (resolved_first, resolved_second, resolved_third),
            )
            self.assertTrue(selected_all.follow_all)
            self.assertEqual(cleared.selected, ())
            self.assertFalse(cleared.follow_all)


class EvalDashboardColorTests(unittest.TestCase):
    def test_algorithm_and_series_colors_are_distinct_and_stable(self):
        methods = (
            "full_recompute",
            "no_cache",
            "no_recompute",
            "kvpacket",
            "sempic",
            "sempic_kvpacket",
            "a3",
            "cache_blend",
            "epic",
            "rand_recompute",
            "sam_kv",
        )
        colors = [algorithm_color(method) for method in methods]

        self.assertEqual(len(colors), len(set(colors)))
        self.assertEqual(algorithm_color("kv_packet"), algorithm_color("kvpacket"))
        self.assertNotEqual(algorithm_color("foo-bar"), algorithm_color("foo_bar"))
        self.assertEqual(
            series_color("sempic", "series-a"),
            series_color("sempic", "series-a"),
        )
        self.assertNotEqual(
            series_color("sempic", "series-a"),
            series_color("sempic", "series-b"),
        )
        neutral = series_color("rand_recompute", "series-a")
        self.assertEqual(neutral[1:3], neutral[3:5])
        self.assertEqual(neutral[3:5], neutral[5:7])

    def test_series_colors_match_across_exact_chart_types(self):
        first = normalize_result(
            "/tmp/colors/a_result.json",
            result_payload(),
            modified_at=1.0,
        )[0]
        second_payload = result_payload(method="sempic")
        second_payload["config"]["lora"] = {"path": "./adapter"}
        second = normalize_result(
            "/tmp/colors/b_result.json",
            second_payload,
            modified_at=1.0,
        )[0]
        frame = records_to_frame([first, second])
        expected = {
            series_color(str(row.method), str(row.series_id))
            for row in frame[["method", "series_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        }

        figures = (
            build_run_metric_figure(frame, "metric.f1"),
            build_run_tradeoff_figure(frame, "metric.ttft_mean"),
        )
        for figure in figures:
            self.assertIsNotNone(figure)
            observed = {
                color
                for trace in figure.data
                if isinstance(trace.marker.color, (list, tuple))
                for color in trace.marker.color
            }
            self.assertEqual(observed, expected)

    def test_exact_figures_keep_one_marker_per_numeric_observation(self):
        first = normalize_result(
            "/tmp/exact/a_result.json", result_payload(), modified_at=1.0
        )[0]
        second_payload = result_payload(method="sempic")
        second_payload["config"]["lora"] = {"path": "./adapter"}
        second_payload["config"]["dataset"]["dataset_name"] = "niah"
        second = normalize_result(
            "/tmp/exact/b_result.json", second_payload, modified_at=1.0
        )[0]
        frame = records_to_frame([first, second])

        run_figure = build_run_metric_figure(frame, "metric.f1")
        cross_figure = build_cross_dataset_figure(frame, "metric.f1")

        self.assertIsNotNone(run_figure)
        self.assertIsNotNone(cross_figure)
        self.assertEqual(sum(len(trace.x) for trace in run_figure.data), 2)
        self.assertEqual(sum(len(trace.y) for trace in cross_figure.data), 2)
        self.assertIsNone(build_run_metric_figure(frame, "metric.missing"))
        tradeoff = build_run_tradeoff_figure(frame, "metric.ttft_mean")
        self.assertIsNotNone(tradeoff)
        self.assertEqual(sum(len(trace.x) for trace in tradeoff.data), 2)

        repeated = records_to_frame(
            [
                normalize_result(
                    "/tmp/repeat-a/run_result.json",
                    result_payload(),
                    modified_at=1.0,
                )[0],
                normalize_result(
                    "/tmp/repeat-b/run_result.json",
                    result_payload(),
                    modified_at=2.0,
                )[0],
            ]
        )
        repeated_tradeoff = build_run_tradeoff_figure(
            repeated, "metric.ttft_mean"
        )
        self.assertIsNotNone(repeated_tradeoff)
        self.assertEqual(sum(len(trace.x) for trace in repeated_tradeoff.data), 2)


class EvalDashboardExactChartScaleTests(unittest.TestCase):
    @staticmethod
    def _figures(frame: pd.DataFrame):
        return (
            build_run_metric_figure(frame, "metric.f1"),
            build_cross_dataset_figure(frame, "metric.f1"),
            build_run_tradeoff_figure(frame, "metric.ttft_mean"),
        )

    def test_svg_and_webgl_thresholds_are_exact(self):
        for count, expected_type in ((500, "scatter"), (501, "scattergl")):
            with self.subTest(count=count):
                figures = self._figures(exact_chart_frame(count))
                for figure in figures:
                    self.assertIsNotNone(figure)
                    self.assertTrue(figure.data)
                    self.assertTrue(
                        all(trace.type == expected_type for trace in figure.data)
                    )

    def test_ranked_axis_has_at_most_thirty_labels(self):
        figure = build_run_metric_figure(exact_chart_frame(501), "metric.f1")

        self.assertIsNotNone(figure)
        self.assertLessEqual(len(figure.layout.yaxis.tickvals), 30)
        self.assertEqual(
            len(figure.layout.yaxis.tickvals), len(figure.layout.yaxis.ticktext)
        )

    def test_five_thousand_one_is_rejected(self):
        oversized = exact_chart_frame(5_001)
        builders = (
            lambda: build_run_metric_figure(oversized, "metric.f1"),
            lambda: build_cross_dataset_figure(oversized, "metric.f1"),
            lambda: build_run_tradeoff_figure(oversized, "metric.ttft_mean"),
        )
        for builder in builders:
            with self.assertRaises(ExactChartTooLarge) as context:
                builder()
            self.assertEqual(context.exception.count, 5_001)
            self.assertEqual(context.exception.limit, 5_000)

    def test_size_guard_counts_only_metric_eligible_observations(self):
        frame = exact_chart_frame(5_001)
        frame["metric.f1"] = math.nan

        self.assertIsNone(build_run_metric_figure(frame, "metric.f1"))
        self.assertIsNone(build_cross_dataset_figure(frame, "metric.f1"))
        self.assertIsNone(build_run_tradeoff_figure(frame, "metric.ttft_mean"))

    def test_rendering_thresholds_count_eligible_not_raw_observations(self):
        cases = ((501, 500, "scatter"), (5_001, 5_000, "scattergl"))
        for raw_count, eligible_count, expected_type in cases:
            with self.subTest(raw_count=raw_count, eligible_count=eligible_count):
                frame = exact_chart_frame(raw_count)
                frame.loc[eligible_count:, "metric.f1"] = math.nan

                figures = self._figures(frame)

                for figure in figures:
                    self.assertIsNotNone(figure)
                    self.assertEqual(
                        sum(len(trace.x) for trace in figure.data), eligible_count
                    )
                    self.assertTrue(
                        all(trace.type == expected_type for trace in figure.data)
                    )


class EvalDashboardAppTests(unittest.TestCase):
    def test_oversized_exact_charts_ask_user_to_narrow_scope(self):
        view_calls = (
            (
                "from sempic.eval_dashboard.views.experiment import render_experiment",
                "render_experiment",
            ),
            (
                "from sempic.eval_dashboard.views.cross_dataset import render_cross_dataset",
                "render_cross_dataset",
            ),
        )
        for import_line, function_name in view_calls:
            with self.subTest(view=function_name):
                app = AppTest.from_string(
                    "from tests.test_eval_dashboard import exact_chart_frame\n"
                    f"{import_line}\n"
                    f"{function_name}(exact_chart_frame(5001))\n",
                    default_timeout=30,
                ).run()

                self.assertFalse(app.exception)
                warning_text = " ".join(item.value for item in app.warning)
                self.assertIn("5,000", warning_text)
                self.assertIn("narrow", warning_text.lower())
                self.assertFalse(app.get("plotly_chart"))

    def test_four_workflows_render_without_exceptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = result_payload()
            second = result_payload(method="sempic")
            second["config"]["dataset"]["dataset_name"] = "niah"
            second["config"]["lora"] = {"path": "./adapter"}
            write_result(root / "biography" / "eval_results" / "first_result.json", first)
            write_result(root / "niah" / "eval_results" / "second_result.json", second)
            app = AppTest.from_string(
                "from sempic.eval_dashboard.app import render_dashboard\n"
                f"render_dashboard([{str(root)!r}])\n",
                default_timeout=30,
            ).run()

            self.assertFalse(app.exception)
            self.assertEqual(
                app.segmented_control(key="eval_dashboard.workflow").value, "Runs"
            )
            for workflow in ("Experiment", "Cross-dataset", "Audit", "Runs"):
                app.segmented_control(key="eval_dashboard.workflow").set_value(
                    workflow
                ).run()
                self.assertFalse(app.exception, workflow)
                self.assertIn(workflow, [item.value for item in app.subheader])

            app.button(key="eval_dashboard.directories.clear").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(
                any("No result directories are selected" in item.value for item in app.info)
            )
            app.button(key="eval_dashboard.directories.all").click().run()
            self.assertFalse(app.exception)
            self.assertIn("Runs", [item.value for item in app.subheader])

            app.text_input(key="eval_dashboard.runs.run_regex").set_value("(").run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            app.text_input(key="eval_dashboard.runs.run_regex").set_value("").run()

            app.segmented_control(key="eval_dashboard.workflow").set_value(
                "Cross-dataset"
            ).run()
            app.selectbox(key="eval_dashboard.cross_dataset.mode").set_value(
                "Dataset-matched algorithm rollup"
            ).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            self.assertTrue(
                any("Estimator:" in item.value for item in app.caption)
            )
            provenance_frames = [
                item.value
                for item in app.dataframe
                if "reason" in item.value.columns
            ]
            self.assertEqual(len(provenance_frames), 1)
            self.assertFalse(provenance_frames[0]["included"].any())
            self.assertTrue(
                set(provenance_frames[0]["reason"]) == {"unresolved_checkpoint"}
            )

            app.selectbox(key="eval_dashboard.cross_dataset.mode").set_value(
                "Shared checkpoint"
            ).run()
            self.assertFalse(app.exception)
            self.assertTrue(
                app.selectbox(key="eval_dashboard.cross_dataset.checkpoint")
            )

            app.button(key="eval_dashboard.scope.models.clear").click().run()
            app.segmented_control(key="eval_dashboard.workflow").set_value("Audit").run()
            self.assertFalse(app.exception)
            self.assertIn("Audit", [item.value for item in app.subheader])

    def test_mixed_stable_and_history_roots_are_rejected_before_scanning(self):
        app = AppTest.from_string(
            "from sempic.eval_dashboard.app import render_dashboard\n"
            "render_dashboard(['eval_config', 'eval_outputs'])\n",
            default_timeout=30,
        ).run()

        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertIn("cannot be scanned together", app.error[0].value)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "eval_config").mkdir()
            (root / "eval_outputs").mkdir()
            parent_app = AppTest.from_string(
                "from sempic.eval_dashboard.app import render_dashboard\n"
                f"render_dashboard([{str(root)!r}])\n",
                default_timeout=30,
            ).run()

        self.assertFalse(parent_app.exception)
        self.assertTrue(parent_app.error)
        self.assertIn("cannot be scanned together", parent_app.error[0].value)


if __name__ == "__main__":
    unittest.main()
