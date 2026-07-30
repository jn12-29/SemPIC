import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from plot_scripts.export_attention_global_bar_summary import export_summary, global_bar_rows


class AttentionSummaryExportTests(unittest.TestCase):
    @staticmethod
    def _minimal_payload():
        return {
            "metric_specs": {"metric": {"label": "Metric"}},
            "records": [
                {
                    "model_id": "model",
                    "dataset_id": "dataset",
                    "query_pass_id": "shifted_prediction",
                    "metric_key": "metric",
                    "view_key": "global_bar",
                    "method_key": "method",
                    "facets": {},
                    "axes": [],
                    "coordinates": {},
                    "mean": 1.0,
                    "sem": 0.1,
                    "count": 20,
                }
            ],
        }

    def test_exports_only_scalar_global_bar_records(self):
        payload = {
            "metric_specs": {
                "attention_absolute_deviation": {"label": "Attention absolute deviation"}
            },
            "records": [
                {
                    "model_id": "model",
                    "dataset_id": "dataset",
                    "query_pass_id": "shifted_prediction",
                    "metric_key": "attention_absolute_deviation",
                    "view_key": "layer_curve",
                    "method_key": "vanilla_pic",
                    "facets": {},
                    "axes": ["layer"],
                    "coordinates": {"layer": [0]},
                    "mean": torch.tensor([1.0]),
                    "sem": torch.tensor([0.1]),
                    "count": torch.tensor([20]),
                },
                {
                    "model_id": "model",
                    "dataset_id": "dataset",
                    "query_pass_id": "shifted_prediction",
                    "metric_key": "attention_absolute_deviation",
                    "view_key": "global_bar",
                    "method_key": "vanilla_pic",
                    "facets": {"region": "prefix", "edge_ratio": "0.1"},
                    "axes": [],
                    "coordinates": {},
                    "mean": torch.tensor(0.25),
                    "sem": torch.tensor(0.05),
                    "count": torch.tensor(20),
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.pt"
            output = root / "summary.csv"
            torch.save(payload, metrics)
            export_summary(metrics, output, overwrite=False)
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["view"], "global_bar")
        self.assertEqual(rows[0]["count"], "20")
        self.assertEqual(rows[0]["layer"], "")
        self.assertEqual(rows[0]["facets"], '{"edge_ratio":"0.1","region":"prefix"}')
        self.assertEqual(rows[0]["mean"], "0.25")
        self.assertEqual(rows[0]["sem"], "0.05000000074505806")

    def test_refuses_to_replace_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.pt"
            output = root / "summary.csv"
            torch.save({"metric_specs": {}, "records": []}, metrics)
            output.write_text("existing")
            with self.assertRaises(FileExistsError):
                export_summary(metrics, output, overwrite=False)
            self.assertEqual(output.read_text(), "existing")
            self.assertEqual(list(root.glob(".summary.csv.*")), [])

    def test_rejects_values_the_consumer_cannot_load(self):
        base_record = {
            "model_id": "model",
            "dataset_id": "dataset",
            "query_pass_id": "shifted_prediction",
            "metric_key": "metric",
            "view_key": "global_bar",
            "method_key": "method",
            "facets": {},
            "axes": [],
            "coordinates": {},
            "mean": 1.0,
            "sem": 0.1,
            "count": 20,
        }
        for field, value in (("mean", math.nan), ("sem", math.inf), ("count", 0)):
            with self.subTest(field=field):
                record = dict(base_record)
                record[field] = value
                with self.assertRaises(ValueError):
                    global_bar_rows(
                        {"metric_specs": {"metric": {"label": "Metric"}}, "records": [record]}
                    )

    def test_no_clobber_when_target_appears_during_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.pt"
            output = root / "summary.csv"
            torch.save(self._minimal_payload(), metrics)

            def create_competing_target(_source, target):
                Path(target).write_text("competitor")
                raise FileExistsError

            with mock.patch(
                "plot_scripts.export_attention_global_bar_summary.os.link",
                side_effect=create_competing_target,
            ):
                with self.assertRaises(FileExistsError):
                    export_summary(metrics, output, overwrite=False)
            self.assertEqual(output.read_text(), "competitor")
            self.assertEqual(list(root.glob(".summary.csv.*")), [])

    def test_cleans_unique_temporary_file_after_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.pt"
            output = root / "summary.csv"
            torch.save(self._minimal_payload(), metrics)
            with mock.patch.object(csv.DictWriter, "writerows", side_effect=OSError("write failed")):
                with self.assertRaises(OSError):
                    export_summary(metrics, output, overwrite=False)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".summary.csv.*")), [])


if __name__ == "__main__":
    unittest.main()
