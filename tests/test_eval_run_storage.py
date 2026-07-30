import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_eval as run_eval_entry
from sempic.utils.run_storage import atomic_write_json as real_atomic_write_json


RESULT = {
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "ttft": 0.1,
    "flops": 1.0,
    "num_orig_tokens": 10,
    "num_wrapped_tokens": 8,
}


def write_eval_config(path: Path, *, run_suffix: str | None = None) -> None:
    config = {
        "model": {
            "model_path": "unused",
            "dtype": "float32",
            "device": "cpu",
            "generation_kwargs": {},
        },
        "dataset": {
            "dataset_name": "biography",
            "num_samples": 1,
            "num_data_strs": 1,
            "num_shots": 0,
            "subset": "1k",
            "split": "test",
            "seed": 0,
        },
        "cache_comb": {"method": "epic", "kwargs": {"recompute_tokens": 30}},
        "seed": 0,
    }
    if run_suffix is not None:
        config["run_suffix"] = run_suffix
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config), encoding="utf-8")


class EvalRunStorageTests(unittest.TestCase):
    def _run_patches(self):
        resources = SimpleNamespace(
            model=object(),
            tokenizer=object(),
            packet_wrapper=None,
        )
        return (
            mock.patch.object(run_eval_entry, "load_eval_resources", return_value=resources),
            mock.patch.object(run_eval_entry, "build_eval_generator", return_value=iter(())),
            mock.patch.object(run_eval_entry, "build_generation_config", return_value=object()),
            mock.patch.object(run_eval_entry, "build_compressor", return_value=(None, False)),
            mock.patch.object(run_eval_entry, "run_eval", return_value=RESULT),
            mock.patch.object(run_eval_entry, "SupportedModel", object),
        )

    def test_config_scope_preserves_repo_parent_and_external_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            internal = config_root / "model name" / "cross_domain" / "source" / "run.json"
            external = root / "custom configs" / "trial.json"

            self.assertEqual(
                run_eval_entry.eval_config_scope(internal, config_root),
                Path("model_name/cross_domain/source"),
            )
            self.assertEqual(
                run_eval_entry.eval_config_scope(external, config_root),
                Path("_external/custom_configs/trial"),
            )

    def test_cli_skip_creates_only_invocation_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            config_path = config_root / "model" / "dataset" / "stale.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{", encoding="utf-8")
            stable_path = run_eval_entry.stable_eval_result_path(config_path)
            stable_path.parent.mkdir()
            stable_path.write_text('{"existing": true}', encoding="utf-8")
            output_root = root / "eval_outputs"

            with mock.patch.object(run_eval_entry, "create_eval_resource_cache", return_value={}):
                exit_code = run_eval_entry.main(
                    [str(config_path), "--run-suffix", "batch"],
                    eval_output_root=output_root,
                    eval_config_root=config_root,
                )

            self.assertEqual(exit_code, 0)
            invocations = list((output_root / "_invocations").iterdir())
            self.assertEqual(len(invocations), 1)
            self.assertRegex(invocations[0].name, r"^\d{8}_\d{6}_batch$")
            self.assertIn(
                "Skipping existing evaluation for config:",
                (invocations[0] / "run.log").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.loads((invocations[0] / "cli_args.json").read_text())["run_suffix"],
                "batch",
            )
            self.assertEqual(
                [path for path in output_root.rglob("*") if path.is_dir()],
                [output_root / "_invocations", invocations[0]],
            )

    def test_missing_configs_marks_invocation_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "eval_outputs"

            exit_code = run_eval_entry.main(
                [str(Path(tmpdir) / "missing.json")],
                eval_output_root=output_root,
                eval_config_root=Path(tmpdir) / "eval_config",
            )

            invocation = next((output_root / "_invocations").iterdir())
            run_log = (invocation / "run.log").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 1)
            self.assertIn("Failed run_eval invocation", run_log)
            self.assertNotIn("Completed run_eval invocation", run_log)

    def test_success_writes_immutable_run_and_stable_copy_with_cli_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            config_path = config_root / "qwen" / "biography" / "epic_30.json"
            write_eval_config(config_path, run_suffix="from-config")
            output_root = root / "eval_outputs"
            patches = self._run_patches()

            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                run_eval_entry.run_one_config(
                    str(config_path),
                    {},
                    {},
                    overwrite=True,
                    cli_args={"run_suffix": "from-cli"},
                    eval_output_root=output_root,
                    eval_config_root=config_root,
                    run_suffix_override="from-cli",
                )

            method_root = output_root / "qwen" / "biography" / "epic"
            run_dirs = list(method_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertTrue(
                re.fullmatch(r"\d{8}_\d{6}_epic_30_from-cli", run_dir.name)
            )
            canonical_path = run_dir / "epic_30_result.json"
            stable_path = run_eval_entry.stable_eval_result_path(config_path)
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            stable = json.loads(stable_path.read_text(encoding="utf-8"))
            self.assertEqual(canonical, stable)
            self.assertEqual(canonical["config"]["run_suffix"], "from-cli")
            self.assertEqual(
                json.loads((run_dir / "eval_config.json").read_text(encoding="utf-8")),
                json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8")),
            )
            self.assertTrue((run_dir / "run.log").is_file())
            self.assertTrue((run_dir / "cli_args.json").is_file())
            self.assertFalse((run_dir / "debug").exists())

    def test_runtime_failure_keeps_previous_stable_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            config_path = config_root / "qwen" / "biography" / "epic_30.json"
            write_eval_config(config_path)
            stable_path = run_eval_entry.stable_eval_result_path(config_path)
            stable_path.parent.mkdir()
            stable_path.write_text('{"old": true}', encoding="utf-8")
            output_root = root / "eval_outputs"

            with mock.patch.object(
                run_eval_entry,
                "load_eval_resources",
                side_effect=RuntimeError("resource failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "resource failure"):
                    run_eval_entry.run_one_config(
                        str(config_path),
                        {},
                        {},
                        overwrite=True,
                        eval_output_root=output_root,
                        eval_config_root=config_root,
                    )

            self.assertEqual(json.loads(stable_path.read_text()), {"old": True})
            run_dir = next((output_root / "qwen" / "biography" / "epic").iterdir())
            self.assertFalse((run_dir / "epic_30_result.json").exists())
            self.assertIn("resource failure", (run_dir / "run.log").read_text())

    def test_overwrite_creates_a_new_immutable_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            config_path = config_root / "qwen" / "biography" / "epic_30.json"
            write_eval_config(config_path)
            output_root = root / "eval_outputs"
            patches = self._run_patches()
            second_result = dict(RESULT, f1=0.75)

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4] as run_eval_mock,
                patches[5],
            ):
                run_eval_mock.side_effect = [RESULT, second_result]
                for _ in range(2):
                    run_eval_entry.run_one_config(
                        str(config_path),
                        {},
                        {},
                        overwrite=True,
                        eval_output_root=output_root,
                        eval_config_root=config_root,
                    )

            method_root = output_root / "qwen" / "biography" / "epic"
            canonical_paths = sorted(method_root.glob("*/epic_30_result.json"))
            self.assertEqual(len(canonical_paths), 2)
            historical_f1 = {
                json.loads(path.read_text())["result"]["f1"]
                for path in canonical_paths
            }
            self.assertEqual(historical_f1, {0.5, 0.75})
            stable_path = run_eval_entry.stable_eval_result_path(config_path)
            self.assertEqual(json.loads(stable_path.read_text())["result"]["f1"], 0.75)

    def test_publication_failure_preserves_canonical_and_previous_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "eval_config"
            config_path = config_root / "qwen" / "biography" / "epic_30.json"
            write_eval_config(config_path)
            stable_path = run_eval_entry.stable_eval_result_path(config_path)
            stable_path.parent.mkdir()
            stable_path.write_text('{"old": true}', encoding="utf-8")
            output_root = root / "eval_outputs"
            patches = self._run_patches()

            def publish_or_fail(path, value):
                if Path(path) == stable_path:
                    raise OSError("publish failure")
                real_atomic_write_json(path, value)

            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
                mock.patch.object(
                    run_eval_entry,
                    "atomic_write_json",
                    side_effect=publish_or_fail,
                ),
            ):
                with self.assertRaisesRegex(OSError, "publish failure"):
                    run_eval_entry.run_one_config(
                        str(config_path),
                        {},
                        {},
                        overwrite=True,
                        eval_output_root=output_root,
                        eval_config_root=config_root,
                    )

            self.assertEqual(json.loads(stable_path.read_text()), {"old": True})
            run_dir = next((output_root / "qwen" / "biography" / "epic").iterdir())
            canonical_path = run_dir / "epic_30_result.json"
            self.assertEqual(json.loads(canonical_path.read_text())["result"], RESULT)
            self.assertIn("publish failure", (run_dir / "run.log").read_text())


if __name__ == "__main__":
    unittest.main()
