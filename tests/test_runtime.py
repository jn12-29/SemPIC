import argparse
import json
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import datetime
from io import StringIO
from pathlib import Path

import torch
from unittest.mock import MagicMock

import run_eval as run_eval_entry
from sempic.cache import KVCache
from sempic.cache_comb import PrefillResult
from sempic.evaluation import preparation as eval_preparation
from sempic.evaluation import runtime as eval_runtime
from sempic.prompt import ContextBlock, Inline, PromptSequence
from sempic.utils.config import load_config_file
from sempic.utils.generate import get_answers
from sempic.utils.run_storage import (
    allocate_run_dir,
    atomic_write_json,
    compose_run_suffix,
    select_existing_run_dir,
    validate_run_suffix,
)
from sempic.utils.runtime import (
    DebugRecorder,
    RuntimeContext,
    add_runtime_cli_args,
    apply_runtime_overrides,
    debug_recording_scope,
    kv_cache_summary,
    load_debug_dump_config,
    runtime_overrides_from_args,
)


class RuntimeTests(unittest.TestCase):
    def test_ttft_summary_reports_warmed_distribution(self):
        summary = run_eval_entry._summarize_ttft([0.1, 0.2, 0.3, 0.4])

        self.assertAlmostEqual(summary["ttft"], 0.25)
        self.assertEqual(summary["ttft"], summary["ttft_mean"])
        self.assertAlmostEqual(summary["ttft_p50"], 0.25)
        self.assertAlmostEqual(summary["ttft_p90"], 0.37)
        self.assertAlmostEqual(summary["ttft_p99"], 0.397)
        self.assertEqual(summary["ttft_min"], 0.1)
        self.assertEqual(summary["ttft_max"], 0.4)
        self.assertAlmostEqual(summary["ttft_std"], 0.11180339887498948)
        self.assertEqual(summary["ttft_count"], 4)

        empty = run_eval_entry._summarize_ttft([])
        self.assertEqual(empty["ttft_count"], 0)
        self.assertTrue(all(value == 0.0 for key, value in empty.items() if key != "ttft_count"))

        single = run_eval_entry._summarize_ttft([0.125])
        self.assertEqual(single["ttft_std"], 0.0)
        self.assertEqual(single["ttft_p99"], 0.125)

    def test_eval_artifact_paths_require_concrete_repository_relative_paths(self):
        base_config = {
            "model": {"model_path": "unused"},
            "dataset": {
                "dataset_name": "niah",
                "num_samples": 1,
                "num_data_strs": 1,
                "num_shots": 0,
                "subset": "default",
                "split": "test",
                "seed": 0,
            },
            "cache_comb": {"method": "sempic_kvpacket", "kwargs": {}},
            "packet_wrapper": {
                "path": "./artifacts/run-1/packet_wrapper.pt",
            },
            "lora": {"path": "./artifacts/run-1/lora"},
            "seed": 0,
        }

        loaded = eval_runtime.load_eval_config(base_config)
        self.assertEqual(
            loaded["packet_wrapper"]["path"],
            "./artifacts/run-1/packet_wrapper.pt",
        )
        self.assertEqual(loaded["lora"]["path"], "./artifacts/run-1/lora")
        latest_named_run = {
            **base_config,
            "lora": {"path": "./artifacts/latest-run/lora"},
        }
        self.assertEqual(
            eval_runtime.load_eval_config(latest_named_run)["lora"]["path"],
            "./artifacts/latest-run/lora",
        )

        invalid_paths = (
            "artifacts/run-1/lora",
            "/tmp/run-1/lora",
            "./artifacts/../run-1/lora",
            "./artifacts/latest/lora",
        )
        for field_name in ("packet_wrapper", "lora"):
            for artifact_path in invalid_paths:
                with self.subTest(field_name=field_name, artifact_path=artifact_path):
                    invalid = {
                        **base_config,
                        field_name: {"path": artifact_path},
                    }
                    with self.assertRaisesRegex(ValueError, rf"{field_name}\.path"):
                        eval_runtime.load_eval_config(invalid)

    def test_eval_artifact_paths_do_not_require_existing_files(self):
        config = {
            "model": {"model_path": "unused"},
            "dataset": {
                "dataset_name": "niah",
                "num_samples": 1,
                "num_data_strs": 1,
                "num_shots": 0,
                "subset": "default",
                "split": "test",
                "seed": 0,
            },
            "cache_comb": {"method": "kvpacket", "kwargs": {}},
            "packet_wrapper": {"path": "./does-not-exist/packet_wrapper.pt"},
            "seed": 0,
        }

        loaded = eval_runtime.load_eval_config(config)

        self.assertEqual(
            loaded["packet_wrapper"]["path"],
            "./does-not-exist/packet_wrapper.pt",
        )

    def test_eval_result_snapshots_preserve_configured_artifact_paths(self):
        result = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ttft": 0.0,
            "flops": 0.0,
            "num_orig_tokens": 0,
            "num_wrapped_tokens": 0,
        }
        packet_path = "./artifacts/run-1/packet_wrapper.pt"
        lora_path = "./artifacts/run-1/lora"
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "joint.json"
            with config_path.open("w") as config_file:
                json.dump({
                    "model": {"model_path": "unused", "device": "cpu"},
                    "dataset": {
                        "dataset_name": "niah",
                        "num_samples": 1,
                        "num_data_strs": 1,
                        "num_shots": 0,
                        "subset": "default",
                        "split": "test",
                        "seed": 0,
                    },
                    "cache_comb": {"method": "sempic_kvpacket", "kwargs": {}},
                    "packet_wrapper": {"path": packet_path},
                    "lora": {"path": lora_path},
                    "seed": 0,
                }, config_file)

            resources = MagicMock(
                model=object(),
                tokenizer=object(),
                packet_wrapper=object(),
                lora_adapter_name="eval-adapter",
            )
            eval_output_root = Path(tmpdir) / "eval_outputs"
            with (
                unittest.mock.patch.object(
                    run_eval_entry,
                    "load_eval_resources",
                    return_value=resources,
                ),
                unittest.mock.patch.object(
                    run_eval_entry,
                    "build_eval_generator",
                    return_value=iter([]),
                ),
                unittest.mock.patch.object(
                    run_eval_entry,
                    "build_generation_config",
                    return_value=None,
                ),
                unittest.mock.patch.object(
                    run_eval_entry,
                    "build_compressor",
                    return_value=(None, False),
                ),
                unittest.mock.patch.object(run_eval_entry, "run_eval", return_value=result),
                unittest.mock.patch.object(run_eval_entry, "SupportedModel", object),
            ):
                run_eval_entry.run_one_config(
                    str(config_path),
                    eval_runtime.create_eval_resource_cache(),
                    {},
                    overwrite=True,
                    eval_output_root=eval_output_root,
                    eval_config_root=Path(tmpdir) / "eval_config",
                )

            snapshot_path = next(eval_output_root.rglob("eval_config.json"))
            run_dir = snapshot_path.parent
            persisted_paths = (
                snapshot_path,
                run_dir / "resolved_config.json",
                run_dir / "joint_result.json",
                Path(tmpdir) / "eval_results" / "joint_result.json",
            )
            for persisted_path in persisted_paths:
                with self.subTest(persisted_path=persisted_path):
                    with persisted_path.open() as persisted_file:
                        payload = json.load(persisted_file)
                    config = payload.get("config", payload)
                    self.assertEqual(config["packet_wrapper"]["path"], packet_path)
                    self.assertEqual(config["lora"]["path"], lora_path)

    def test_runtime_defaults_and_cli_overrides(self):
        raw_config = {
            "logging": {"log_dir": "./base_logs", "level": "WARNING"},
            "debug_dump": {"enabled": False, "sample_limit": 3},
        }
        overrides = {
            "debug_enabled": True,
            "log_level": "DEBUG",
            "debug_sample_limit": 7,
        }

        resolved = apply_runtime_overrides(raw_config, overrides)

        self.assertEqual(resolved["logging"], {"level": "DEBUG"})
        self.assertTrue(resolved["debug_dump"]["enabled"])
        self.assertEqual(resolved["debug_dump"]["sample_limit"], 7)
        self.assertFalse(resolved["debug_dump"]["save_token_ids"])
        self.assertFalse(resolved["debug_dump"]["save_tensor_values"])

        no_debug = apply_runtime_overrides(
            {"debug_dump": {"enabled": True}},
            {"debug_enabled": False},
        )
        self.assertFalse(no_debug["debug_dump"]["enabled"])

    def test_debug_false_in_child_config_overrides_inherited_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = os.path.join(tmpdir, "_default.json")
            config_path = os.path.join(tmpdir, "child.json")
            with open(default_path, "w") as f:
                json.dump({"debug_dump": {"enabled": True, "sample_limit": 5}}, f)
            with open(config_path, "w") as f:
                json.dump({"debug_dump": {"enabled": False}}, f)

            config = load_config_file(config_path, default_config_file="_default.json")
            debug_config = load_debug_dump_config(config)

        self.assertFalse(debug_config["enabled"])
        self.assertEqual(debug_config["sample_limit"], 5)

    def test_runtime_cli_debug_tristate(self):
        parser = argparse.ArgumentParser()
        add_runtime_cli_args(parser)

        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertNotIn("--log-dir", option_strings)
        self.assertIsNone(runtime_overrides_from_args(parser.parse_args([]))["debug_enabled"])
        self.assertTrue(runtime_overrides_from_args(parser.parse_args(["--debug"]))["debug_enabled"])
        self.assertFalse(runtime_overrides_from_args(parser.parse_args(["--no-debug"]))["debug_enabled"])

    def test_runtime_context_writes_run_files_and_debug_shape_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "example.json")
            with open(config_file, "w") as f:
                json.dump({}, f)
            resolved_config = apply_runtime_overrides(
                {"logging": {"log_dir": tmpdir}, "debug_dump": {"enabled": True}},
            )
            allocated_run_dir = allocate_run_dir(Path(tmpdir) / "runs", "example")
            self.assertFalse((allocated_run_dir / "debug").exists())
            stderr = StringIO()
            with redirect_stderr(stderr):
                with RuntimeContext(
                    entrypoint="run_eval",
                    run_dir=allocated_run_dir,
                    config_file=config_file,
                    resolved_config=resolved_config,
                    config_snapshot_name="eval_config.json",
                    cli_args={"debug_enabled": True},
                ) as runtime_context:
                    self.assertFalse(runtime_context.debug_dir.exists())
                    logging.getLogger("sempic.test_runtime").info("package logger message")
                    runtime_context.debug_recorder.record_json(
                        "tensor",
                        {"tensor": torch.tensor([[1, 2], [3, 4]])},
                    )
                    run_dir = str(runtime_context.run_dir)

            tensor_path = os.path.join(run_dir, "debug", "tensor.json")
            with open(tensor_path) as f:
                artifact = json.load(f)

            self.assertTrue(os.path.exists(os.path.join(run_dir, "run.log")))
            with open(os.path.join(run_dir, "run.log")) as f:
                self.assertIn("package logger message", f.read())
            self.assertIn("package logger message", stderr.getvalue())
            self.assertTrue(os.path.exists(os.path.join(run_dir, "resolved_config.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "eval_config.json")))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "cli_args.json")))
            self.assertEqual(artifact["tensor"]["shape"], [2, 2])
            self.assertNotIn("values", artifact["tensor"])

    def test_runtime_context_invocation_and_snapshot_invariant(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = allocate_run_dir(tmpdir, "invocation")
            with RuntimeContext(
                entrypoint="run_eval",
                run_dir=run_dir,
                config_file=None,
                resolved_config=None,
                config_snapshot_name=None,
                cli_args={"overwrite": False, "log_level": "DEBUG"},
            ) as runtime_context:
                self.assertFalse((run_dir / "debug").exists())
                self.assertEqual(runtime_context.logger.level, logging.DEBUG)

            self.assertTrue((run_dir / "run.log").is_file())
            self.assertTrue((run_dir / "cli_args.json").is_file())
            self.assertFalse((run_dir / "resolved_config.json").exists())

            with self.assertRaisesRegex(ValueError, "must be all set or all None"):
                RuntimeContext(
                    entrypoint="run_eval",
                    run_dir=run_dir,
                    config_file="config.json",
                    resolved_config=None,
                    config_snapshot_name=None,
                    cli_args={},
                )

    def test_runtime_context_restores_handlers_when_snapshot_write_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = allocate_run_dir(tmpdir, "broken")
            package_logger = logging.getLogger("sempic")
            handlers_before = tuple(package_logger.handlers)
            context = RuntimeContext(
                entrypoint="run_eval",
                run_dir=run_dir,
                config_file="config.json",
                resolved_config=apply_runtime_overrides({}),
                config_snapshot_name="eval_config.json",
                cli_args={},
            )

            with unittest.mock.patch.object(
                context,
                "write_json",
                side_effect=OSError("snapshot failure"),
            ):
                with self.assertRaisesRegex(OSError, "snapshot failure"):
                    context.__enter__()

            self.assertEqual(tuple(package_logger.handlers), handlers_before)

    def test_run_storage_suffix_allocation_and_collision(self):
        self.assertEqual(
            compose_run_suffix("epic config", user_suffix="paper-1"),
            "epic_config_paper-1",
        )
        self.assertIsNone(compose_run_suffix())
        validate_run_suffix("r8.qkvo-try_1")
        with self.assertRaisesRegex(ValueError, "run_suffix"):
            validate_run_suffix("../bad")
        with self.assertRaisesRegex(ValueError, "run_suffix"):
            compose_run_suffix("epic", user_suffix="bad suffix")

        with tempfile.TemporaryDirectory() as tmpdir:
            now = datetime(2026, 7, 22, 18, 30, 0)
            first = allocate_run_dir(tmpdir, "epic", now=now)
            second = allocate_run_dir(tmpdir, "epic", now=now)

            self.assertEqual(first.name, "20260722_183000_epic")
            self.assertEqual(second.name, "20260722_183000_epic_1")

    def test_run_storage_existing_atomic_json_and_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "runs"
            run_dir = allocate_run_dir(root, None, now=datetime(2026, 7, 22, 19, 0, 0))
            self.assertEqual(select_existing_run_dir(root, run_dir), run_dir.resolve())

            latest = root / "latest"
            latest.symlink_to(run_dir.name, target_is_directory=True)
            self.assertTrue(latest.is_symlink())
            self.assertEqual(os.readlink(latest), run_dir.name)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                select_existing_run_dir(root, latest)

            alias_parent = root / "alias"
            alias_parent.symlink_to(".", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "direct child"):
                select_existing_run_dir(root, alias_parent / run_dir.name)

            json_path = run_dir / "result.json"
            atomic_write_json(json_path, {"value": 1})
            with open(json_path) as f:
                self.assertEqual(json.load(f), {"value": 1})
            with self.assertRaises(TypeError):
                atomic_write_json(json_path, {"value": object()})
            with open(json_path) as f:
                self.assertEqual(json.load(f), {"value": 1})

    def test_debug_context_records_get_answers_without_changing_return(self):
        class FakeTokenizer:
            eos_token_id = 2

            def decode(self, tokens, skip_special_tokens=True):
                del skip_special_tokens
                return " ".join(str(int(token)) for token in tokens)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DebugRecorder(
                tmpdir,
                {
                    "enabled": True,
                    "sample_limit": 2,
                    "save_token_ids": False,
                    "save_tensor_values": False,
                },
            )
            generated = torch.tensor([[9, 8, 5, 6, 2, 7]])
            input_ids = torch.tensor([[9, 8]])

            with debug_recording_scope(recorder):
                answers = get_answers(generated, input_ids, FakeTokenizer())

            with open(os.path.join(tmpdir, "generation_answers.json")) as f:
                artifact = json.load(f)

        self.assertEqual(answers, ["5 6"])
        self.assertEqual(artifact["generated_tokens"]["shape"], [1, 4])
        self.assertNotIn("values", artifact["generated_tokens"])

    def test_kv_cache_summary_does_not_consolidate_chunks(self):
        cache = KVCache.create_dummy(
            num_layers=1,
            batch_size=1,
            num_heads=1,
            seq_len=2,
            key_head_dim=3,
            value_head_dim=3,
            device=torch.device("cpu"),
        )
        extra = KVCache.create_dummy(
            num_layers=1,
            batch_size=1,
            num_heads=1,
            seq_len=1,
            key_head_dim=3,
            value_head_dim=3,
            device=torch.device("cpu"),
        )
        cache._cache[0].extend(extra._cache[0])

        summary = kv_cache_summary(cache)

        self.assertEqual(len(cache._cache[0]), 2)
        self.assertEqual(len(summary["layers"]["0"]), 2)

    def test_eval_builds_executor_before_iterating_samples(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, texts, **kwargs):
                del kwargs
                if isinstance(texts, str):
                    texts = [texts]
                return {"input_ids": [[1] * len(text) for text in texts]}

        class FakeBody:
            def embed_tokens(self, input_ids):
                return torch.zeros((input_ids.size(0), input_ids.size(1), 1))

        class FakeModel:
            device = torch.device("cpu")

        lifecycle = []

        def eval_entries():
            self.assertEqual(lifecycle, ["executor_ready"])
            yield {
                "query": "q",
                "answer": "a",
                "prompt": PromptSequence((
                    Inline("p"),
                    ContextBlock("doc"),
                    Inline("prompt"),
                )),
            }

        class FakeTTFTTimer:
            def __init__(self, device):
                self.device = device

            def start(self):
                lifecycle.append("timer_started")

        def fake_build_executor(method_name, model):
            self.assertEqual(method_name, "no_recompute")
            self.assertIsInstance(model, FakeModel)
            lifecycle.append("executor_ready")
            return fake_cache_comb_func

        def fake_cache_comb_func(**kwargs):
            lifecycle.append("formal_prefill")
            self.assertEqual(kwargs["prompt"].input_ids.numel(), 10)
            self.assertEqual(list(kwargs["prepared_kvs"]), [1])
            return PrefillResult(
                logits=torch.zeros((1, 1, 3)),
                past_key_values=MagicMock(),
                generation_input_ids=torch.tensor([[1]]),
                position_ids=torch.tensor([[0]]),
                attention_mask=torch.ones((1, 1), dtype=torch.long),
                flops=0,
            )

        def fake_warmup(**kwargs):
            lifecycle.append("warmup")
            self.assertEqual(kwargs["prompt"].input_ids.numel(), 10)
            self.assertEqual(list(kwargs["prepared_kvs"]), [1])

        fake_cache_comb_func.warmup = fake_warmup

        def fake_generate_from_prefill(**kwargs):
            lifecycle.append("first_token")
            self.assertIsInstance(kwargs["ttft_timer"], FakeTTFTTimer)
            return torch.tensor([[1, 2]]), 0.0

        with (
            unittest.mock.patch.object(eval_preparation, "get_causal_lm_body", return_value=FakeBody()),
            unittest.mock.patch.object(
                eval_preparation,
                "get_kv_caches",
                return_value=[KVCache.create_dummy(1, 1, 1, 1, 1, 1)],
            ),
            unittest.mock.patch.object(
                run_eval_entry,
                "build_cache_comb_executor",
                side_effect=fake_build_executor,
            ),
            unittest.mock.patch.object(
                run_eval_entry,
                "generate_from_prefill",
                side_effect=fake_generate_from_prefill,
            ),
            unittest.mock.patch.object(run_eval_entry, "TTFTTimer", FakeTTFTTimer),
            unittest.mock.patch.object(
                run_eval_entry,
                "get_answers",
                return_value=["a"],
            ),
        ):
            result = run_eval_entry.run_eval(
                model=FakeModel(), # type: ignore[arg-type]
                tokenizer=FakeTokenizer(), # type: ignore[arg-type]
                eval_generator=eval_entries(),
                cache_comb_method="no_recompute",
                cache_comb_kwargs={},
            )

        self.assertEqual(
            lifecycle,
            [
                "executor_ready",
                "warmup",
                "timer_started",
                "formal_prefill",
                "first_token",
            ],
        )
        self.assertEqual(result["f1"], 1.0)

    def test_eval_entry_passes_tokenizer_to_eval_generator(self):
        tokenizer = object()
        model = object()
        model_key = ("unused", "float32", "cpu")
        result = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ttft": 0.0,
            "flops": 0.0,
            "num_orig_tokens": 0,
            "num_wrapped_tokens": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "eval.json")
            with open(config_path, "w") as f:
                json.dump({
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
                        "template": "tokenizer_chat",
                        "template_kwargs": {"enable_thinking": True},
                    },
                    "cache_comb": {"method": "no_cache", "kwargs": {}},
                    "seed": 0,
                    "logging": {"log_dir": tmpdir},
                    "debug_dump": {"enabled": False},
                }, f)

            with (
                unittest.mock.patch.object(
                    eval_runtime,
                    "get_ret_eval_generator",
                    return_value=iter([]),
                ) as get_generator_mock,
                unittest.mock.patch.object(
                    run_eval_entry,
                    "build_cache_comb_executor",
                    return_value=lambda **kwargs: kwargs,
                ),
                unittest.mock.patch.object(
                    run_eval_entry,
                    "run_eval",
                    return_value=result,
                ),
                unittest.mock.patch.object(run_eval_entry, "SupportedModel", object),
            ):
                run_eval_entry.run_one_config(
                    config_path,
                    {
                        "packet_wrapper": {},
                        "model": {model_key: model},
                        "tokenizer": {model_key: tokenizer},
                        "lora_adapter": {},
                    },
                    {},
                    overwrite=True,
                    eval_output_root=os.path.join(tmpdir, "eval_outputs"),
                    eval_config_root=os.path.join(tmpdir, "eval_config"),
                )

        self.assertIs(get_generator_mock.call_args.kwargs["tokenizer"], tokenizer)

    def test_eval_model_cache_shares_base_across_lora_paths(self):
        result = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "ttft": 0.0,
            "flops": 0.0,
            "num_orig_tokens": 0,
            "num_wrapped_tokens": 0,
        }
        load_calls = []

        class FakeModel:
            def __init__(self, idx):
                self.idx = idx
                self.generation_config = type("GenerationConfigStub", (), {"pad_token_id": None})()

        class FakeTokenizer:
            padding_side = "left"
            pad_token_id = 0
            eos_token_id = 2

        def fake_from_pretrained(*args, **kwargs):
            del args, kwargs
            model = FakeModel(len(load_calls))
            load_calls.append(model)
            return model

        with tempfile.TemporaryDirectory() as tmpdir:
            config_paths = []
            for idx, lora_path in enumerate(("./adapter/a", "./adapter/b")):
                config_path = os.path.join(tmpdir, f"eval_{idx}.json")
                with open(config_path, "w") as f:
                    json.dump({
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
                        "cache_comb": {"method": "sempic", "kwargs": {}},
                        "lora": {"path": lora_path},
                        "seed": 0,
                        "logging": {"log_dir": tmpdir},
                    }, f)
                config_paths.append(config_path)

            with (
                unittest.mock.patch.object(eval_runtime.AutoModelForCausalLM, "from_pretrained", side_effect=fake_from_pretrained),
                unittest.mock.patch.object(eval_runtime.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()),
                unittest.mock.patch.object(eval_runtime, "load_lora_adapter_for_eval"),
                unittest.mock.patch.object(eval_runtime, "get_ret_eval_generator", return_value=iter([])),
                unittest.mock.patch.object(run_eval_entry, "build_cache_comb_executor", return_value=lambda **kwargs: kwargs),
                unittest.mock.patch.object(run_eval_entry, "run_eval", return_value=result),
                unittest.mock.patch.object(run_eval_entry, "SupportedModel", object),
            ):
                eval_cache = {
                    "packet_wrapper": {},
                    "model": {},
                    "tokenizer": {},
                    "lora_adapter": {},
                }
                for config_path in config_paths:
                    run_eval_entry.run_one_config(
                        config_path,
                        eval_cache,
                        {},
                        overwrite=True,
                        eval_output_root=os.path.join(tmpdir, "eval_outputs"),
                        eval_config_root=os.path.join(tmpdir, "eval_config"),
                    )

        self.assertEqual(len(load_calls), 1)


if __name__ == "__main__":
    unittest.main()
