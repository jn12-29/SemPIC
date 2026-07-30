import copy
from pathlib import Path
import tempfile
import unittest

import torch

from sempic.attention_metrics.processed_storage import (
    load_processed_metrics,
    save_processed_metrics,
    validate_processed_metrics,
)
from sempic.attention_metrics.processing import normalize_processing_config
from sempic.attention_metrics.profile_identity import fingerprint
from sempic.attention_visualization.attention_reports import (
    plan_attention_report_pages,
    validate_attention_report_records,
)
from sempic.attention_visualization.common import display_scale
from sempic.attention_visualization.processed import plot_processed_metrics
from sempic.attention_visualization.retrieval_reports import (
    plan_retrieval_report_pages,
    validate_retrieval_report_records,
)


class ProcessedFrontendTest(unittest.TestCase):
    def setUp(self):
        self.artifact = _artifact()

    def test_strict_validation_and_round_trip(self):
        self.assertIs(validate_processed_metrics(self.artifact), self.artifact)
        with tempfile.TemporaryDirectory() as directory:
            path = save_processed_metrics(Path(directory) / "metrics.pt", self.artifact)
            loaded = load_processed_metrics(path)
        self.assertEqual(loaded["metric_specs"], self.artifact["metric_specs"])
        self.assertEqual(len(loaded["records"]), len(self.artifact["records"]))

    def test_rejects_shape_identity_and_estimate_violations(self):
        mutations = []
        bad_axes = copy.deepcopy(self.artifact)
        bad_axes["records"][0]["axes"] = ["query_head", "layer"]
        mutations.append(bad_axes)
        bad_coordinate = copy.deepcopy(self.artifact)
        bad_coordinate["records"][0]["coordinates"]["layer"] = [0]
        mutations.append(bad_coordinate)
        bad_source = copy.deepcopy(self.artifact)
        bad_source["records"][0]["query_pass_id"] = "missing"
        mutations.append(bad_source)
        bad_dtype = copy.deepcopy(self.artifact)
        bad_dtype["records"][0]["mean"] = bad_dtype["records"][0]["mean"].float()
        mutations.append(bad_dtype)
        negative = copy.deepcopy(self.artifact)
        negative["records"][0]["mean"][0, 0] = -0.1
        mutations.append(negative)
        missing_spec = copy.deepcopy(self.artifact)
        del missing_spec["metric_specs"]["attention_absolute_deviation"]
        mutations.append(missing_spec)
        duplicate = copy.deepcopy(self.artifact)
        duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
        mutations.append(duplicate)
        for invalid in mutations:
            with self.subTest(keys=invalid.keys()), self.assertRaises(ValueError):
                validate_processed_metrics(invalid)

    def test_rejects_model_id_that_can_escape_output_root(self):
        invalid = copy.deepcopy(self.artifact)
        for source in invalid["source_partitions"]:
            source["model_id"] = "../escape"
        for record in invalid["records"]:
            record["model_id"] = "../escape"

        with self.assertRaisesRegex(ValueError, "canonical path ID"):
            validate_processed_metrics(invalid)

    def test_report_plans_are_complete_ordered_and_input_order_independent(self):
        records = self.artifact["records"]
        expected_attention = [
            (query, view)
            for query in ("terminal_query", "gold_answer")
            for view in ("raw", "chunk_conditional")
        ]
        for candidate in (records, list(reversed(records))):
            attention = plan_attention_report_pages(candidate)
            retrieval = plan_retrieval_report_pages(candidate)
            self.assertEqual(attention["attention_maps"], expected_attention)
            self.assertEqual(attention["attention_error_structure"], expected_attention)
            self.assertTrue(all(
                pages == ["terminal_query", "gold_answer"]
                for pages in retrieval.values()
            ))

    def test_plotting_writes_only_five_model_level_pdf_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = save_processed_metrics(root / "metrics.pt", self.artifact)
            outputs = plot_processed_metrics(metrics, root / "figures")
            expected = {
                "attention_maps.pdf",
                "attention_error_structure.pdf",
                "retrieval_nrmse.pdf",
                "retrieval_cosine_distance.pdf",
                "retrieval_mass_error.pdf",
            }
            self.assertEqual({path.name for path in outputs}, expected)
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(path.parent == root / "figures/qwen3_4b" for path in outputs))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in outputs))
            self.assertEqual(list((root / "figures").rglob("*.png")), [])
            self.assertEqual(list((root / "figures").rglob("*.csv")), [])
            self.assertEqual(len(list((root / "figures").rglob("*.pdf"))), 5)

    def test_report_validation_fails_on_missing_required_records(self):
        attention_missing = copy.deepcopy(self.artifact)
        _remove_record(
            attention_missing,
            metric="attention_absolute_deviation", view="layer_curve",
            query="terminal_query", dataset="biography", method="kvpacket",
            facets={"attention_view": "raw", "edge_ratio": "0.1", "region": "prefix"},
        )
        with self.assertRaisesRegex(ValueError, "Expected one attention report record"):
            validate_attention_report_records(
                attention_missing["records"], attention_missing["processing_config"]
            )

        retrieval_missing = copy.deepcopy(self.artifact)
        _remove_record(
            retrieval_missing, metric="retrieval_nrmse", view="global_bar",
            query="gold_answer", dataset="niah", method="kvpacket", facets={},
        )
        with self.assertRaisesRegex(ValueError, "Missing retrieval report record"):
            validate_retrieval_report_records(retrieval_missing["records"])

    def test_plotting_preflights_all_reports_before_writing(self):
        invalid = copy.deepcopy(self.artifact)
        _remove_record(
            invalid, metric="retrieval_nrmse", view="global_bar",
            query="gold_answer", dataset="niah", method="kvpacket", facets={},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = save_processed_metrics(root / "metrics.pt", invalid)
            with self.assertRaisesRegex(ValueError, "Missing retrieval report record"):
                plot_processed_metrics(metrics, root / "figures")
            self.assertEqual(list((root / "figures").rglob("*.pdf")), [])

    def test_plotting_preflights_every_model_before_writing(self):
        invalid = copy.deepcopy(self.artifact)
        second_sources = copy.deepcopy(invalid["source_partitions"])
        for index, source in enumerate(second_sources, start=16):
            source["model_id"] = "qwen3_8b"
            source["partition_fingerprint"] = f"{index:064x}"
        second_records = copy.deepcopy(invalid["records"])
        for record in second_records:
            record["model_id"] = "qwen3_8b"
        invalid["source_partitions"].extend(second_sources)
        invalid["records"].extend(second_records)
        invalid["records"] = [
            record for record in invalid["records"]
            if not (
                record["model_id"] == "qwen3_8b"
                and record["metric_key"] == "retrieval_nrmse"
                and record["view_key"] == "global_bar"
                and record["query_pass_id"] == "gold_answer"
                and record["dataset_id"] == "niah"
                and record["method_key"] == "kvpacket"
                and record["facets"] == {}
            )
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = save_processed_metrics(root / "metrics.pt", invalid)
            with self.assertRaisesRegex(ValueError, "Missing retrieval report record"):
                plot_processed_metrics(metrics, root / "figures")
            self.assertEqual(list((root / "figures").rglob("*.pdf")), [])

    def test_report_validation_rejects_coordinate_mismatch(self):
        invalid = copy.deepcopy(self.artifact)
        record = next(
            item for item in invalid["records"]
            if item["metric_key"] == "retrieval_nrmse"
            and item["view_key"] == "layer_curve"
        )
        record["coordinates"]["layer"] = [0, 3]
        validate_processed_metrics(invalid)
        with self.assertRaisesRegex(ValueError, "mismatched layer"):
            validate_retrieval_report_records(invalid["records"])

    def test_display_scale_uses_clipping_and_extreme_symlog(self):
        values = torch.tensor(
            [0.1] * 900 + [100.0] * 100 + [1000.0], dtype=torch.float64
        ).numpy()
        linear = display_scale([values])
        nonlinear = display_scale([values], allow_symlog=True)
        self.assertLess(linear.display_max, linear.true_max)
        self.assertTrue(linear.clipped)
        self.assertFalse(linear.nonlinear)
        self.assertTrue(nonlinear.nonlinear)

        sparse_extreme = torch.tensor(
            [2.0] * 990 + [1000.0] * 10, dtype=torch.float64
        ).numpy()
        self.assertTrue(display_scale([sparse_extreme], allow_symlog=True).nonlinear)


def _artifact():
    config = normalize_processing_config({
        "position_mode": "normalized",
        "num_position_bins": 3,
        "edge_ratios": ["0.1", "0.2"],
    })
    metric_specs = {
        "attention_profile": _spec("Full attention profile", "Attention probability"),
        "attention_absolute_deviation": _spec("Attention deviation", "Absolute deviation"),
        "retrieval_nrmse": _spec("Retrieval NRMSE", "NRMSE"),
        "retrieval_cosine_distance": _spec("Cosine distance", "Cosine distance"),
        "attention_mass_error": _spec("Attention mass error", "Absolute mass error"),
    }
    records = []
    methods = {"biography": ("kvpacket", "sempic"), "niah": ("kvpacket",)}
    for query_index, query in enumerate(("terminal_query", "gold_answer"), start=1):
        for dataset, candidates in methods.items():
            dataset_scale = query_index + (dataset == "niah")
            for attention_view in ("raw", "chunk_conditional"):
                facets = {"attention_view": attention_view, "position_mode": "normalized"}
                records.append(_record(
                    query, dataset, "attention_profile", "layer_position_heatmap",
                    "full_recompute", ["layer", "position_bin"],
                    {"layer": [0, 2], "position_bin": [0.1, 0.5, 0.9]},
                    0.05 * dataset_scale, facets,
                ))
                for method_index, method in enumerate(candidates, start=1):
                    scale = float(dataset_scale * method_index)
                    records.append(_record(
                        query, dataset, "attention_absolute_deviation",
                        "layer_position_heatmap", method,
                        ["layer", "position_bin"],
                        {"layer": [0, 2], "position_bin": [0.1, 0.5, 0.9]},
                        scale, facets,
                    ))
                    for ratio in config["edge_ratios"]:
                        for region in ("prefix", "interior", "suffix"):
                            region_facets = {
                                "attention_view": attention_view,
                                "edge_ratio": ratio,
                                "region": region,
                            }
                            records.append(_record(
                                query, dataset, "attention_absolute_deviation",
                                "layer_curve", method, ["layer"], {"layer": [0, 2]},
                                scale, region_facets,
                            ))
                            records.append(_record(
                                query, dataset, "attention_absolute_deviation",
                                "global_bar", method, [], {}, scale, region_facets,
                            ))
            for method_index, method in enumerate(candidates, start=1):
                scale = float(dataset_scale * method_index)
                for metric in (
                    "retrieval_nrmse", "retrieval_cosine_distance", "attention_mass_error"
                ):
                    records.extend((
                        _record(
                            query, dataset, metric, "layer_head_heatmap", method,
                            ["layer", "query_head"],
                            {"layer": [0, 2], "query_head": [0, 1]}, scale, {},
                        ),
                        _record(
                            query, dataset, metric, "layer_curve", method,
                            ["layer"], {"layer": [0, 2]}, scale, {},
                        ),
                        _record(
                            query, dataset, metric, "global_bar", method,
                            [], {}, scale, {},
                        ),
                    ))
    sources = []
    fingerprint_chars = iter("abcd")
    for query in ("terminal_query", "gold_answer"):
        for dataset in methods:
            sources.append({
                "partition_fingerprint": next(fingerprint_chars) * 64,
                "model_id": "qwen3_4b",
                "dataset_id": dataset,
                "query_pass_id": query,
            })
    return {
        "processing_config": config,
        "processing_fingerprint": fingerprint(config),
        "source_partitions": sources,
        "metric_specs": metric_specs,
        "records": records,
    }


def _spec(label, value_label):
    return {"label": label, "value_label": value_label, "axis_policy": "nonnegative_auto"}


def _record(query, dataset, metric, view, method, axes, coordinates, scale, facets):
    shape = tuple(len(coordinates[axis]) for axis in axes)
    if shape:
        mean = torch.arange(1, _numel(shape) + 1, dtype=torch.float64).reshape(shape)
        mean = mean * scale
    else:
        mean = torch.tensor(scale, dtype=torch.float64)
    return {
        "model_id": "qwen3_4b",
        "dataset_id": dataset,
        "query_pass_id": query,
        "metric_key": metric,
        "view_key": view,
        "method_key": method,
        "facets": dict(facets),
        "axes": list(axes),
        "coordinates": dict(coordinates),
        "mean": mean,
        "sem": torch.full(shape, 0.1, dtype=torch.float64),
        "count": torch.full(shape, 2, dtype=torch.int64),
    }


def _remove_record(artifact, *, metric, view, query, dataset, method, facets):
    artifact["records"] = [
        record for record in artifact["records"]
        if not (
            record["metric_key"] == metric
            and record["view_key"] == view
            and record["query_pass_id"] == query
            and record["dataset_id"] == dataset
            and record["method_key"] == method
            and record["facets"] == facets
        )
    ]


def _numel(shape):
    result = 1
    for size in shape:
        result *= size
    return result


if __name__ == "__main__":
    unittest.main()
