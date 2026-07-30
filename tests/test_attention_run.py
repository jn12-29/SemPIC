import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sempic.attention_metrics.attention_run import (
    _execute_run,
    create_attention_run,
    load_run_record,
    resume_attention_run,
)


ROOT = Path(__file__).resolve().parents[1]


def eval_config(model: Path, method: str) -> dict:
    return {
        "model": {
            "model_path": str(model),
            "dtype": "float32",
            "device": "cpu",
            "generation_kwargs": {},
        },
        "dataset": {"dataset_name": "fixture", "split": "test", "seed": 1},
        "seed": 1,
        "cache_comb": {
            "method": "no_recompute" if method == "vanilla_pic" else method,
            "kwargs": {},
        },
        "packet_wrapper": {"path": None},
        "lora": {"path": None},
        "compress": None,
        "quantization": None,
    }


def processing_config() -> dict:
    return {
        "position_mode": "auto",
        "num_position_bins": 8,
        "edge_ratios": ["0.2"],
    }


class AttentionRunTests(unittest.TestCase):
    def test_execute_run_collects_then_creates_automatic_variant(self):
        record = {
            "max_samples": 3,
            "processing_config": processing_config(),
        }
        loaded = [("full.json", {"cache_comb": {"method": "full_recompute"}})]
        analysis = object()
        with (
            patch(
                "sempic.attention_metrics.attention_run._restore_inputs",
                return_value=(loaded, analysis),
            ),
            patch(
                "sempic.attention_metrics.attention_run.collect_profile_run_loaded"
            ) as collect,
            patch(
                "sempic.attention_metrics.attention_run.process_attention_run"
            ) as process,
        ):
            process.side_effect = lambda *args, **kwargs: self.assertTrue(
                collect.called
            )
            _execute_run(Path("run"), record)
        collect.assert_called_once_with(
            loaded,
            analysis=analysis,
            run_dir=Path("run"),
            max_samples=3,
        )
        process.assert_called_once_with(
            Path("run"),
            processing_config=record["processing_config"],
        )

    def test_successful_resume_executes_processing_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            loaded = [
                (f"{method}.json", eval_config(model, method))
                for method in ("full_recompute", "vanilla_pic", "kvpacket", "sempic")
            ]
            with (
                patch(
                    "sempic.attention_metrics.attention_run.load_analysis_configs",
                    return_value=loaded,
                ),
                patch("sempic.attention_metrics.attention_run._execute_run"),
            ):
                run_dir = create_attention_run(
                    ["unused.json"],
                    analysis_config_path=ROOT / "attention_config/gold_answer.json",
                    processing_config_path=ROOT / "attention_config/processing_default.json",
                    processing_config=processing_config(),
                    output_dir=root / "attention_results",
                    run_name=None,
                    max_samples=1,
                )
            record = load_run_record(run_dir / "config.json")
            with patch(
                "sempic.attention_metrics.attention_run._execute_run"
            ) as execute:
                resumed = resume_attention_run(run_dir)

            self.assertEqual(resumed, run_dir.resolve())
            execute.assert_called_once_with(run_dir.resolve(), record)

    def test_resume_accepts_historical_png_pdf_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            loaded = [
                (f"{method}.json", eval_config(model, method))
                for method in ("full_recompute", "vanilla_pic")
            ]
            with (
                patch(
                    "sempic.attention_metrics.attention_run.load_analysis_configs",
                    return_value=loaded,
                ),
                patch("sempic.attention_metrics.attention_run._execute_run"),
            ):
                run_dir = create_attention_run(
                    ["unused.json"],
                    analysis_config_path=ROOT / "attention_config/gold_answer.json",
                    processing_config_path=ROOT / "attention_config/processing_default.json",
                    processing_config=processing_config(),
                    output_dir=root / "attention_results",
                    run_name=None,
                    max_samples=1,
                )
            config_path = run_dir / "config.json"
            record = json.loads(config_path.read_text())
            record["figure_formats"] = ["png", "pdf"]
            config_path.write_text(json.dumps(record))

            with patch(
                "sempic.attention_metrics.attention_run._execute_run"
            ) as execute:
                resumed = resume_attention_run(run_dir)

            self.assertEqual(resumed, run_dir.resolve())
            execute.assert_called_once_with(run_dir.resolve(), record)

    def test_new_run_persists_complete_config_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            loaded = [
                (f"{method}.json", eval_config(model, method))
                for method in ("full_recompute", "vanilla_pic", "kvpacket", "sempic")
            ]
            with (
                patch(
                    "sempic.attention_metrics.attention_run.load_analysis_configs",
                    return_value=loaded,
                ),
                patch("sempic.attention_metrics.attention_run._execute_run") as execute,
            ):
                run_dir = create_attention_run(
                    ["unused.json"],
                    analysis_config_path=ROOT / "attention_config/gold_answer.json",
                    processing_config_path=ROOT / "attention_config/processing_default.json",
                    processing_config=processing_config(),
                    output_dir=root / "attention_results",
                    run_name="paper",
                    max_samples=3,
                )
            record = load_run_record(run_dir / "config.json")
        self.assertIn("gold_answer_paper", run_dir.name)
        self.assertEqual(record["max_samples"], 3)
        self.assertEqual(record["figure_formats"], ["pdf"])
        self.assertEqual(len(record["tasks"]), 1)
        self.assertEqual(len(record["eval_configs"]), 4)
        execute.assert_called_once()

    def test_new_run_freezes_artifact_symlink_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            target = root / "adapter-v1"
            model.mkdir()
            target.mkdir()
            (model / "config.json").write_text("{}")
            (target / "adapter.json").write_text("{}")
            alias = root / "latest"
            alias.symlink_to(target.name)
            full = eval_config(model, "full_recompute")
            sempic = eval_config(model, "sempic")
            sempic["lora"]["path"] = str(alias)
            with (
                patch(
                    "sempic.attention_metrics.attention_run.load_analysis_configs",
                    return_value=[("full.json", full), ("sempic.json", sempic)],
                ),
                patch("sempic.attention_metrics.attention_run._execute_run"),
            ):
                run_dir = create_attention_run(
                    ["unused.json"],
                    analysis_config_path=ROOT / "attention_config/gold_answer.json",
                    processing_config_path=ROOT / "attention_config/processing_default.json",
                    processing_config=processing_config(),
                    output_dir=root / "attention_results",
                    run_name=None,
                    max_samples=1,
                )
            record = load_run_record(run_dir / "config.json")
        saved = next(
            item["config"]
            for item in record["eval_configs"]
            if item["config"]["cache_comb"]["method"] == "sempic"
        )
        self.assertEqual(saved["lora"]["path"], str(target.resolve()))

    def test_resume_does_not_rescan_large_artifact_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            model_file = model / "config.json"
            model_file.write_text("{}")
            loaded = [
                (f"{method}.json", eval_config(model, method))
                for method in ("full_recompute", "vanilla_pic", "kvpacket", "sempic")
            ]
            with (
                patch(
                    "sempic.attention_metrics.attention_run.load_analysis_configs",
                    return_value=loaded,
                ),
                patch("sempic.attention_metrics.attention_run._execute_run"),
            ):
                run_dir = create_attention_run(
                    ["unused.json"],
                    analysis_config_path=ROOT / "attention_config/gold_answer.json",
                    processing_config_path=ROOT / "attention_config/processing_default.json",
                    processing_config=processing_config(),
                    output_dir=root / "attention_results",
                    run_name=None,
                    max_samples=1,
                )
            model_file.write_text(json.dumps({"changed": True}))
            with patch("sempic.attention_metrics.attention_run._execute_run") as execute:
                resume_attention_run(run_dir)
            execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
