import json
import multiprocessing
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sempic.attention_metrics.analysis_pipeline import (
    allocate_analysis_variant,
    plot_attention_variant,
    process_attention_run,
)


def processing_config() -> dict:
    return {
        "position_mode": "auto",
        "num_position_bins": 8,
        "edge_ratios": ["0.2"],
    }


def allocate_worker(run_dir: str, timestamp: str, start, results) -> None:
    start.wait()
    try:
        variant = allocate_analysis_variant(
            run_dir,
            now=datetime.strptime(timestamp, "%Y%m%d_%H%M%S"),
        )
        results.put(("ok", variant.name))
    except Exception as error:
        results.put(("error", repr(error)))


class AttentionVariantTests(unittest.TestCase):
    def test_auto_variants_use_global_sequence_across_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            first = allocate_analysis_variant(
                run_dir, now=datetime(2026, 7, 23, 12, 0, 0)
            )
            second = allocate_analysis_variant(
                run_dir, now=datetime(2026, 7, 23, 12, 1, 0)
            )
        self.assertEqual(first.name, "20260723_120000-0")
        self.assertEqual(second.name, "20260723_120100-1")

    def test_concurrent_auto_variants_have_unique_global_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=allocate_worker,
                    args=(directory, f"20260723_1200{index:02d}", start, results),
                )
                for index in range(8)
            ]
            for process in processes:
                process.start()
            start.set()
            records = [results.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

        errors = [payload for status, payload in records if status == "error"]
        self.assertEqual(errors, [])
        names = [payload for status, payload in records if status == "ok"]
        indices = sorted(int(name.rsplit("-", 1)[1]) for name in names)
        self.assertEqual(indices, list(range(8)))

    def test_manual_suffix_and_collisions_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            now = datetime(2026, 7, 23, 12, 0, 0)
            first = allocate_analysis_variant(run_dir, suffix="paper", now=now)
            second = allocate_analysis_variant(run_dir, suffix="paper", now=now)
            third = allocate_analysis_variant(run_dir, suffix="paper", now=now)
            with self.assertRaisesRegex(ValueError, "non-digit"):
                allocate_analysis_variant(run_dir, suffix="7", now=now)
        self.assertEqual(first.name, "20260723_120000-paper")
        self.assertEqual(second.name, "20260723_120000-paper-0")
        self.assertEqual(third.name, "20260723_120000-paper-1")

    def test_processing_creates_complete_isolated_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            statistics = run_dir / "statistics" / "model" / "dataset"
            statistics.mkdir(parents=True)
            partition = statistics / "query_pass.pt"
            partition.write_bytes(b"statistics")

            def fake_process(paths, *, processing_config, metrics_path):
                self.assertEqual(paths, [partition])
                Path(metrics_path).write_bytes(b"metrics")
                return Path(metrics_path)

            def fake_plot(metrics_path, output_dir):
                self.assertEqual(Path(metrics_path).read_bytes(), b"metrics")
                Path(output_dir).mkdir(parents=True)
                (Path(output_dir) / "figure.pdf").write_bytes(b"figure")
                return [Path(output_dir) / "figure.pdf"]

            with (
                patch(
                    "sempic.attention_metrics.analysis_pipeline.process_profile_partitions",
                    side_effect=fake_process,
                ),
                patch(
                    "sempic.attention_metrics.analysis_pipeline.plot_processed_metrics",
                    side_effect=fake_plot,
                ),
            ):
                first = process_attention_run(
                    run_dir,
                    processing_config=processing_config(),
                    now=datetime(2026, 7, 23, 12, 0, 0),
                )
                first_snapshot = {
                    path.relative_to(first): path.read_bytes()
                    for path in first.rglob("*")
                    if path.is_file()
                }
                second = process_attention_run(
                    run_dir,
                    processing_config=processing_config(),
                    suffix="middle",
                    now=datetime(2026, 7, 23, 12, 1, 0),
                )

            self.assertEqual(first.name, "20260723_120000-0")
            self.assertEqual(second.name, "20260723_120100-middle")
            self.assertEqual(partition.read_bytes(), b"statistics")
            self.assertEqual(
                {
                    path.relative_to(first): path.read_bytes()
                    for path in first.rglob("*")
                    if path.is_file()
                },
                first_snapshot,
            )
            self.assertEqual(
                json.loads((second / "processing_config.json").read_text()),
                processing_config(),
            )
            for variant in (first, second):
                self.assertTrue((variant / "metrics.pt").is_file())
                self.assertFalse((variant / "summary.csv").exists())
                self.assertTrue((variant / "figures" / "figure.pdf").is_file())

    def test_replot_uses_exact_variant_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "selected"
            sibling = root / "sibling"
            selected.mkdir()
            sibling.mkdir()
            metrics = selected / "metrics.pt"
            metrics.write_bytes(b"metrics")
            protected = sibling / "figure.pdf"
            protected.write_bytes(b"keep")
            with patch(
                "sempic.attention_metrics.analysis_pipeline.plot_processed_metrics"
            ) as plot:
                result = plot_attention_variant(selected)
            self.assertEqual(result, selected)
            plot.assert_called_once_with(metrics, selected / "figures")
            self.assertEqual(protected.read_bytes(), b"keep")

    def test_failed_processing_preserves_sibling_and_consumes_number(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            statistics = run_dir / "statistics" / "model" / "dataset"
            statistics.mkdir(parents=True)
            (statistics / "query_pass.pt").write_bytes(b"statistics")
            sibling = allocate_analysis_variant(
                run_dir, now=datetime(2026, 7, 23, 12, 0, 0)
            )
            protected = sibling / "metrics.pt"
            protected.write_bytes(b"keep")
            with patch(
                "sempic.attention_metrics.analysis_pipeline.process_profile_partitions",
                side_effect=RuntimeError("processing failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "processing failed"):
                    process_attention_run(
                        run_dir,
                        processing_config=processing_config(),
                        now=datetime(2026, 7, 23, 12, 1, 0),
                    )
            retry = allocate_analysis_variant(
                run_dir, now=datetime(2026, 7, 23, 12, 2, 0)
            )

            self.assertEqual(protected.read_bytes(), b"keep")
            self.assertTrue(
                (run_dir / "analysis" / "20260723_120100-1" / "processing_config.json").is_file()
            )
            self.assertEqual(retry.name, "20260723_120200-2")


if __name__ == "__main__":
    unittest.main()
