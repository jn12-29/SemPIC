import json
import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import GenerationConfig

import run_train
from sempic.packet_wrapper import PacketWrapper
from sempic.prompt import TokenSpan, TokenizedPrompt
from sempic.utils.generate import GenerationCache
from sempic.utils.generation_cache import StreamingGenerationCacheWriter
from sempic.utils.train import (
    TrainAttentionRuntime,
    configure_trainable_parameters,
    create_train_attention_runtime,
    load_train_config,
    train_components,
)
from sempic.utils.train_checkpoint import (
    load_training_checkpoint,
    restore_training_state,
    save_training_checkpoint,
)


class TinyTrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.ones(1))
        self.base_weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(hidden_size=1)

    def forward(self):
        raise AssertionError("not used")


def make_sample(token_id: int = 1):
    return {
        "semantic_key": hashlib.sha256(f"sample-{token_id}".encode()).hexdigest(),
        "prompt": TokenizedPrompt(
            input_ids=torch.tensor([token_id], dtype=torch.long),
            parts=(TokenSpan(kind="inline", start=0, end=1),),
        ),
    }


def write_generation_artifact(
    path: Path,
    cache: GenerationCache,
    *,
    model_path: str = "unused",
) -> None:
    path.mkdir()
    generations = []
    for key in cache.keys():
        generation = cache.get(key)
        assert generation is not None
        generations.append((key, generation))
    stores_logits = {bool(generation["logits"]) for _, generation in generations}
    assert len(stores_logits) == 1
    with StreamingGenerationCacheWriter(
        path,
        provenance={
            "model_path": model_path,
            "tokenizer_path": model_path,
            "dtype": "float32",
            "tokenizer": {
                "padding_side": "left",
                "pad_token_id": 0,
                "eos_token_id": 2,
            },
            "store_logits": stores_logits.pop(),
        },
    ) as writer:
        for key, generation in generations:
            writer.add(key, generation)
        writer.finalize()
    (path / "resolved_config.json").write_text("{}\n", encoding="utf-8")


def cache_tokenizer(**overrides):
    state = {
        "padding_side": "left",
        "pad_token_id": 0,
        "eos_token_id": 2,
    }
    state.update(overrides)
    return SimpleNamespace(**state)


def build_train_config_dict(**overrides):
    config = {
        "output_dir": "/tmp/unused",
        "total_epoch": 1,
        "batch_size": 2,
        "gen_batch_size": 1,
        "forward_batch_size": 1,
        "seed": 0,
        "cache_device": "cpu",
        "model": {
            "model_path": "unused",
            "dtype": "float32",
            "device": "cpu",
            "generation_kwargs": {},
        },
        "loss": {"type": "kl", "tau": 1.0},
        "lora": {"enabled": False},
        "packet_wrapper": {
            "enabled": True,
            "header_len": 1,
            "trailer_len": 1,
        },
        "optimizers": {
            "packet_wrapper": {
                "opt_config": {"lr": 0.1, "weight_decay": 0.0},
                "scheduler_config": {"start_factor": 1.0, "end_factor": 1.0},
            }
        },
        "data_configs": [
            {
                "dataset_name": "biography",
                "num_samples": 2,
                "num_data_strs": 1,
                "num_shots": 0,
                "subset": "1k",
                "split": "train",
                "seed": 0,
            }
        ],
    }
    config.update(overrides)
    return config


class UnifiedTrainConfigTests(unittest.TestCase):
    def test_cache_path_normalizes_one_or_multiple_artifact_directories(self):
        single = load_train_config(build_train_config_dict(cache_path="artifact-a"))
        multiple = load_train_config(build_train_config_dict(
            cache_path=["artifact-a", "artifact-b"],
        ))

        self.assertEqual(single["cache_path"], ["artifact-a"])
        self.assertEqual(multiple["cache_path"], ["artifact-a", "artifact-b"])

    def test_cache_path_rejects_empty_duplicate_or_non_cpu_backing(self):
        invalid_configs = (
            build_train_config_dict(cache_path=[]),
            build_train_config_dict(cache_path=["artifact", "artifact"]),
            build_train_config_dict(cache_path=["artifact", ""]),
            build_train_config_dict(cache_path="artifact", cache_device="cuda:0"),
        )
        for config in invalid_configs:
            with self.subTest(cache_path=config["cache_path"]):
                with self.assertRaises(ValueError):
                    load_train_config(config)

    def test_batched_cuda_attention_warns_and_uses_sdpa(self):
        with (
            mock.patch(
                "sempic.utils.train.get_model_device",
                return_value=torch.device("cuda:0"),
            ),
            self.assertLogs("sempic.utils.train", level="WARNING") as logs,
        ):
            runtime = create_train_attention_runtime(object(), 2)

        self.assertEqual(runtime.backend, "sdpa")
        self.assertTrue(runtime.verified)
        self.assertTrue(any("falling back to SDPA" in message for message in logs.output))

    def test_explicit_sdpa_attention_does_not_probe_device(self):
        with mock.patch("sempic.utils.train.get_model_device") as get_device:
            runtime = create_train_attention_runtime(
                object(),
                1,
                attention_backend="sdpa",
            )

        get_device.assert_not_called()
        self.assertEqual(runtime.backend, "sdpa")
        self.assertTrue(runtime.verified)

    def test_flex_backward_probe_falls_back_to_sdpa_once(self):
        class FailBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, parameter):
                del ctx
                return parameter.sum()

            @staticmethod
            def backward(ctx, grad_output):
                del ctx, grad_output
                raise RuntimeError("synthetic Flex backward failure")

        model = TinyTrainModel()
        optimizer = torch.optim.SGD([model.lora_weight], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=1.0
        )
        runtime = TrainAttentionRuntime(backend="flex", verified=False)
        backends = []

        def fake_loss(**kwargs):
            backend = kwargs["attention_backend"]
            backends.append(backend)
            loss = (
                FailBackward.apply(model.lora_weight)
                if backend == "flex"
                else model.lora_weight.sum()
            )
            return loss, 1, [float(loss.detach())]

        with (
            mock.patch("sempic.utils.train.probe_train_flex_attention_shapes") as shape_probe,
            mock.patch("sempic.utils.train.batched_student_loss", side_effect=fake_loss),
            self.assertLogs("sempic.utils.train", level="WARNING") as logs,
        ):
            train_components(
                samples=[make_sample()],  # type: ignore[arg-type]
                model=model,
                batch_size=1,
                forward_batch_size=1,
                optimizers={"lora": optimizer},
                schedulers={"lora": scheduler},
                generation_cache=GenerationCache(),
                loss_config={"type": "ce", "tau": 1.0},
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=None,
                attention_runtime=runtime,
            )

        shape_probe.assert_called_once()
        self.assertEqual(backends, ["flex", "sdpa", "sdpa"])
        self.assertEqual(runtime.backend, "sdpa")
        self.assertTrue(runtime.verified)
        self.assertEqual(
            sum("falling back to SDPA" in message for message in logs.output),
            1,
        )

    def test_flex_shape_probe_failure_falls_back_to_sdpa(self):
        model = TinyTrainModel()
        optimizer = torch.optim.SGD([model.lora_weight], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=1.0
        )
        runtime = TrainAttentionRuntime(backend="flex", verified=False)
        backends = []

        def fake_loss(**kwargs):
            backends.append(kwargs["attention_backend"])
            loss = model.lora_weight.sum()
            return loss, 1, [float(loss.detach())]

        with (
            mock.patch(
                "sempic.utils.train.probe_train_flex_attention_shapes",
                side_effect=RuntimeError("unsupported shape"),
            ),
            mock.patch("sempic.utils.train.batched_student_loss", side_effect=fake_loss),
            self.assertLogs("sempic.utils.train", level="WARNING") as logs,
        ):
            train_components(
                samples=[make_sample()],  # type: ignore[arg-type]
                model=model,
                batch_size=1,
                forward_batch_size=1,
                optimizers={"lora": optimizer},
                schedulers={"lora": scheduler},
                generation_cache=GenerationCache(),
                loss_config={"type": "ce", "tau": 1.0},
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=None,
                attention_runtime=runtime,
            )

        self.assertEqual(backends, ["sdpa", "sdpa"])
        self.assertEqual(runtime.backend, "sdpa")
        self.assertTrue(runtime.verified)
        self.assertTrue(any("unsupported shape" in message for message in logs.output))

    def test_train_config_defaults_targets_and_runtime_defaults(self):
        train_config = load_train_config(build_train_config_dict())

        self.assertEqual(train_config["train"]["targets"], ["packet_wrapper"])
        self.assertEqual(train_config["loss"]["type"], "kl")
        self.assertEqual(train_config["attention_backend"], "auto")
        self.assertTrue(train_config["kv_gradient_checkpointing"])
        self.assertEqual(train_config["logging"]["level"], "INFO")
        self.assertFalse(train_config["debug_dump"]["enabled"])
        self.assertEqual(train_config["output_dir"], "/tmp/unused")
        self.assertIsNone(train_config["run_suffix"])

    def test_output_dir_is_required(self):
        config = build_train_config_dict()
        del config["output_dir"]

        with self.assertRaisesRegex(ValueError, "output_dir"):
            load_train_config(config)

    def test_rejects_component_output_paths(self):
        config = build_train_config_dict()
        config["packet_wrapper"]["save_path"] = "/tmp/legacy"

        with self.assertRaisesRegex(ValueError, "Use output_dir"):
            load_train_config(config)

    def test_lora_adapter_name_is_fixed(self):
        config = build_train_config_dict(
            train={"targets": ["lora"]},
            lora={
                "enabled": True,
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": ["q_proj"],
                "adapter_name": "custom",
            },
            optimizers={
                "lora": {
                    "opt_config": {"lr": 0.1},
                    "scheduler_config": {},
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "adapter_name must be 'default'"):
            load_train_config(config)

    def test_run_suffix_validation(self):
        self.assertEqual(
            load_train_config(build_train_config_dict(run_suffix="r16-qkvo"))["run_suffix"],
            "r16-qkvo",
        )
        with self.assertRaisesRegex(ValueError, "run_suffix"):
            load_train_config(build_train_config_dict(run_suffix="../bad"))

    def test_train_config_loads_kv_gradient_checkpointing(self):
        train_config = load_train_config(
            build_train_config_dict(
                kv_gradient_checkpointing=False,
            )
        )

        self.assertFalse(train_config["kv_gradient_checkpointing"])

    def test_kv_gradient_checkpointing_must_be_boolean(self):
        with self.assertRaisesRegex(ValueError, "kv_gradient_checkpointing must be a boolean"):
            load_train_config(build_train_config_dict(kv_gradient_checkpointing="true"))

    def test_train_targets_must_be_enabled(self):
        config = build_train_config_dict(train={"targets": ["lora"]})

        with self.assertRaisesRegex(ValueError, "enabled components"):
            load_train_config(config)

    def test_rejects_legacy_training_fields(self):
        config = build_train_config_dict(**{"use" + "_logits": True})

        with self.assertRaisesRegex(ValueError, "legacy"):
            load_train_config(config)

    def test_loss_config_controls_generation_cache_logits(self):
        class FakeModel:
            generation_config = GenerationConfig()

            def eval(self):
                return None

        for loss_kind, expected_store_logits in (("kl", True), ("ce", False)):
            train_config = load_train_config(
                build_train_config_dict(loss={"type": loss_kind, "tau": 1.0})
            )
            with mock.patch.object(
                run_train,
                "build_generation_cache",
                return_value=(GenerationCache(), False),
            ) as build_cache_mock:
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[],
                    model=FakeModel(),
                    tokenizer=cache_tokenizer(),  # type: ignore[arg-type]
                    generation_config=None,
                )

            self.assertIs(build_cache_mock.call_args.kwargs["store_logits"], expected_store_logits)

    def test_teacher_generation_defaults_to_greedy_but_allows_sampling(self):
        tokenizer = SimpleNamespace(pad_token_id=17)
        model = SimpleNamespace(generation_config=GenerationConfig())

        for generation_kwargs, expected_do_sample in (({}, False), ({"do_sample": True}, True)):
            with self.subTest(generation_kwargs=generation_kwargs):
                config_dict = build_train_config_dict()
                config_dict["total_epoch"] = 0
                config_dict["model"]["generation_kwargs"] = generation_kwargs
                train_config = load_train_config(config_dict)
                with (
                    mock.patch.object(run_train, "load_tokenizer", return_value=tokenizer),
                    mock.patch.object(run_train, "load_model", return_value=model),
                    mock.patch.object(run_train, "load_packet_wrapper_for_train", return_value=None),
                    mock.patch.object(run_train, "configure_trainable_parameters"),
                    mock.patch.object(run_train, "load_samples", return_value=[make_sample()]),
                    mock.patch.object(
                        run_train,
                        "prepare_generation_cache",
                        return_value=GenerationCache(),
                    ) as prepare_cache,
                    mock.patch.object(run_train, "build_optimizers", return_value=({}, {})),
                    mock.patch.object(run_train, "save_trained_targets"),
                ):
                    run_train.train_one_config(train_config, run_train.TrainCache())

                generation_config = prepare_cache.call_args.kwargs["generation_config"]
                self.assertIs(generation_config.do_sample, expected_do_sample)
                self.assertEqual(generation_config.pad_token_id, 17)

    def test_existing_cache_ignores_train_generation_settings(self):
        tokenizer = SimpleNamespace(pad_token_id=17)
        model = SimpleNamespace(generation_config=None)
        config_dict = build_train_config_dict(cache_path="existing-artifact")
        config_dict["total_epoch"] = 0
        config_dict["model"]["generation_kwargs"] = {"not_a_generation_field": object()}
        train_config = load_train_config(config_dict)

        with (
            mock.patch.object(run_train, "load_tokenizer", return_value=tokenizer),
            mock.patch.object(run_train, "load_model", return_value=model),
            mock.patch.object(run_train, "load_packet_wrapper_for_train", return_value=None),
            mock.patch.object(run_train, "configure_trainable_parameters"),
            mock.patch.object(run_train, "load_samples", return_value=[make_sample()]),
            mock.patch.object(
                run_train,
                "prepare_generation_cache",
                return_value=GenerationCache(),
            ) as prepare_cache,
            mock.patch.object(run_train, "build_optimizers", return_value=({}, {})),
            mock.patch.object(run_train, "save_trained_targets"),
        ):
            run_train.train_one_config(train_config, run_train.TrainCache())

        self.assertIsNone(prepare_cache.call_args.kwargs["generation_config"])

    def test_existing_cache_is_read_only(self):
        sample = make_sample()
        cache = GenerationCache(device=torch.device("cpu"))
        cache.add(sample["semantic_key"], {
            "sequences": [torch.tensor([1], dtype=torch.long)],
            "logits": [torch.zeros(1, 3)],
            "text": ["x"],
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "generation_cache"
            write_generation_artifact(cache_path, cache)
            payload_path = cache_path / "cache.safetensors"
            original_bytes = payload_path.read_bytes()
            train_config = load_train_config(
                build_train_config_dict(cache_path=str(cache_path))
            )

            with mock.patch.object(run_train, "build_generation_cache") as build_cache:
                loaded = run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[sample],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=cache_tokenizer(),  # type: ignore[arg-type]
                    generation_config=None,
                )

            build_cache.assert_not_called()
            self.assertEqual(payload_path.read_bytes(), original_bytes)
            self.assertIsNotNone(loaded.get(sample["semantic_key"]))

    def test_configured_missing_cache_fails_without_generation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = str(Path(temp_dir) / "missing-artifact")
            train_config = load_train_config(
                build_train_config_dict(cache_path=missing_path)
            )

            with (
                self.assertRaisesRegex(FileNotFoundError, "artifact not found"),
                mock.patch.object(run_train, "build_generation_cache") as build_cache,
            ):
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[make_sample()],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=cache_tokenizer(),  # type: ignore[arg-type]
                    generation_config=None,
                )

        build_cache.assert_not_called()

    def test_existing_cache_requires_all_semantic_samples(self):
        cache = GenerationCache()
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            cache.add(hashlib.sha256(b"unrelated").hexdigest(), {
                "sequences": [torch.tensor([1])],
                "logits": [torch.zeros(1, 3)],
                "text": ["x"],
            })
            write_generation_artifact(cache_path, cache)
            train_config = load_train_config(
                build_train_config_dict(cache_path=str(cache_path))
            )

            with self.assertRaisesRegex(KeyError, "missing 1 required semantic samples"):
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[make_sample()],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=cache_tokenizer(),  # type: ignore[arg-type]
                    generation_config=None,
                )

    def test_existing_kl_cache_requires_logits_for_each_sequence(self):
        sample = make_sample()
        cache = GenerationCache()
        cache.add(sample["semantic_key"], {
            "sequences": [torch.tensor([1], dtype=torch.long)],
            "logits": [],
            "text": ["x"],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            write_generation_artifact(cache_path, cache)
            train_config = load_train_config(
                build_train_config_dict(cache_path=str(cache_path))
            )

            with self.assertRaisesRegex(ValueError, "without KL logits"):
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[sample],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=cache_tokenizer(),  # type: ignore[arg-type]
                    generation_config=None,
                )

    def test_existing_cache_must_match_configured_training_model(self):
        sample = make_sample()
        cache = GenerationCache()
        cache.add(sample["semantic_key"], {
            "sequences": [torch.tensor([1], dtype=torch.long)],
            "logits": [torch.zeros(1, 3)],
            "text": ["x"],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            write_generation_artifact(cache_path, cache, model_path="other-model")
            train_config = load_train_config(
                build_train_config_dict(cache_path=str(cache_path))
            )

            with self.assertRaisesRegex(ValueError, "incompatible"):
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[sample],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=object(),  # type: ignore[arg-type]
                    generation_config=None,
                )

    def test_existing_cache_must_match_current_tokenizer_state(self):
        sample = make_sample()
        cache = GenerationCache()
        cache.add(sample["semantic_key"], {
            "sequences": [torch.tensor([1], dtype=torch.long)],
            "logits": [torch.zeros(1, 3)],
            "text": ["x"],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            write_generation_artifact(cache_path, cache)
            train_config = load_train_config(
                build_train_config_dict(cache_path=str(cache_path))
            )

            with self.assertRaisesRegex(ValueError, "tokenizer"):
                run_train.prepare_generation_cache(
                    train_config=train_config,
                    samples=[sample],  # type: ignore[list-item]
                    model=object(),
                    tokenizer=cache_tokenizer(pad_token_id=99),  # type: ignore[arg-type]
                    generation_config=None,
                )

    def test_packet_wrapper_only_training_keeps_model_in_eval_mode(self):
        model = TinyTrainModel()
        model.train()
        packet_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        optimizer = torch.optim.AdamW(
            [packet_wrapper.header, packet_wrapper.trailer],
            lr=0.1,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0)
        sample = make_sample()

        with mock.patch(
            "sempic.utils.train.batched_student_loss",
            return_value=(torch.tensor(1.0, requires_grad=True), 1, [1.0]),
        ):
            train_components(
                samples=[sample],  # type: ignore[arg-type]
                model=model,
                batch_size=1,
                forward_batch_size=1,
                optimizers={"packet_wrapper": optimizer},
                schedulers={"packet_wrapper": scheduler},
                generation_cache=GenerationCache(),
                loss_config={"type": "ce", "tau": 1.0},
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=packet_wrapper,
            )

        self.assertFalse(model.training)

    def test_lora_enabled_training_sets_model_train_mode(self):
        model = TinyTrainModel()
        model.eval()
        optimizer = torch.optim.AdamW([model.lora_weight], lr=0.1)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1.0)
        sample = make_sample()

        with (
            mock.patch("sempic.utils.train.disable_lora_adapters"),
            mock.patch(
                "sempic.utils.train.batched_student_loss",
                return_value=(torch.tensor(1.0, requires_grad=True), 1, [1.0]),
            ),
        ):
            train_components(
                samples=[sample],  # type: ignore[arg-type]
                model=model,
                batch_size=1,
                forward_batch_size=1,
                optimizers={"lora": optimizer},
                schedulers={"lora": scheduler},
                generation_cache=GenerationCache(),
                loss_config={"type": "ce", "tau": 1.0},
                lora_enabled=True,
                lora_adapter_name="lora_kv_cache",
                packet_wrapper=None,
            )

        self.assertTrue(model.training)

    def test_joint_optimizers_step_on_same_boundary(self):
        train_config = load_train_config({
            **build_train_config_dict(
                train={"targets": ["lora", "packet_wrapper"]},
                lora={
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.0,
                    "target_modules": ["q_proj"],
                },
                packet_wrapper={
                    "enabled": True,
                    "header_len": 1,
                    "trailer_len": 1,
                },
                optimizers={
                    "lora": {
                        "opt_config": {"lr": 0.1, "weight_decay": 0.0},
                        "scheduler_config": {"start_factor": 1.0, "end_factor": 1.0},
                    },
                    "packet_wrapper": {
                        "opt_config": {"lr": 0.1, "weight_decay": 0.0},
                        "scheduler_config": {"start_factor": 1.0, "end_factor": 1.0},
                    },
                },
            )
        })
        model = TinyTrainModel()
        packet_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        configure_trainable_parameters(model, packet_wrapper, train_config["train"]["targets"])

        optimizers, schedulers = run_train.build_optimizers(
            train_config,
            model,
            packet_wrapper,
            num_samples=3,
        )
        step_calls = {target: [] for target in optimizers}
        for target, optimizer in optimizers.items():
            original_step = optimizer.step

            def counting_step(*args, _target=target, _original_step=original_step, **kwargs):
                step_calls[_target].append(len(step_calls[_target]))
                return _original_step(*args, **kwargs)

            optimizer.step = counting_step  # type: ignore[method-assign]

        samples = [make_sample(index + 1) for index in range(3)]

        def fake_batch_loss(**kwargs):
            microbatch_size = len(kwargs["samples"])
            fake_batch_loss.calls.append(microbatch_size)
            if microbatch_size == 2:
                return torch.tensor(4.0, requires_grad=True), 2, [2.0, 2.0]
            return torch.tensor(9.0, requires_grad=True), 3, [9.0]

        fake_batch_loss.calls = []
        fixed_now = datetime.fromisoformat("2026-07-24T12:00:00+08:00")

        with (
            mock.patch("sempic.utils.train.batched_student_loss", side_effect=fake_batch_loss),
            mock.patch(
                "sempic.utils.train.time.perf_counter",
                side_effect=[0.0, 0.0, 2.0, 2.0, 5.0],
            ),
            mock.patch("sempic.utils.train.datetime") as mock_datetime,
            self.assertLogs("sempic.utils.train", level="INFO") as logs,
        ):
            mock_datetime.now.return_value.astimezone.return_value = fixed_now
            train_components(
                samples=samples,  # type: ignore[arg-type]
                model=model,
                batch_size=2,
                forward_batch_size=2,
                optimizers=optimizers,
                schedulers=schedulers,
                generation_cache=GenerationCache(),
                loss_config=train_config["loss"],
                lora_enabled=True,
                lora_adapter_name="lora_kv_cache",
                packet_wrapper=packet_wrapper,
                epoch=2,
                total_epochs=4,
            )

        self.assertEqual(fake_batch_loss.calls, [2, 1])
        self.assertEqual(step_calls["lora"], [0, 1])
        self.assertEqual(step_calls["packet_wrapper"], [0, 1])
        progress_logs = [message for message in logs.output if "Overall" in message]
        self.assertEqual(len(progress_logs), 2)
        self.assertIn("Epoch 3/4 | Step 1/2 | Overall 5/8 (62.5%)", progress_logs[0])
        self.assertIn("step_loss 2 | epoch_avg_loss 2", progress_logs[0])
        self.assertIn("1.0 tok/s | 1.0 samples/s", progress_logs[0])
        self.assertIn("step 2.00s | epoch_elapsed 00:00:02 | ETA 00:00:06", progress_logs[0])
        self.assertIn("finish 2026-07-24 12:00:06+08:00", progress_logs[0])
        self.assertIn("Epoch 3/4 | Step 2/2 | Overall 6/8 (75.0%)", progress_logs[1])
        self.assertIn("step_loss 3 | epoch_avg_loss 2.6", progress_logs[1])
        self.assertIn("step 3.00s | epoch_elapsed 00:00:05 | ETA 00:00:05", progress_logs[1])
        self.assertIn("finish 2026-07-24 12:00:05+08:00", progress_logs[1])

    def test_final_targets_share_output_directory(self):
        model = TinyTrainModel()
        packet_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as output_dir:
            train_config = load_train_config(build_train_config_dict(
                output_dir=output_dir,
                train={"targets": ["lora", "packet_wrapper"]},
                lora={
                    "enabled": True,
                    "rank": 2,
                    "alpha": 4,
                    "dropout": 0.0,
                    "target_modules": ["q_proj"],
                    "adapter_name": "default",
                },
                optimizers={
                    "lora": {
                        "opt_config": {"lr": 0.1},
                        "scheduler_config": {},
                    },
                    "packet_wrapper": {
                        "opt_config": {"lr": 0.1},
                        "scheduler_config": {},
                    },
                },
            ))
            with mock.patch.object(run_train, "save_lora_adapter") as save_lora:
                run_train.save_trained_targets(train_config, model, packet_wrapper)

            save_lora.assert_called_once_with(
                model, os.path.join(output_dir, "lora"), "default"
            )
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "packet_wrapper.pt")))

    def test_checkpoint_round_trip_restores_joint_state(self):
        model = TinyTrainModel()
        packet_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        configure_trainable_parameters(model, packet_wrapper, ["lora", "packet_wrapper"])
        train_config = load_train_config(build_train_config_dict(
            train={"targets": ["lora", "packet_wrapper"]},
            lora={
                "enabled": True,
                "rank": 2,
                "alpha": 4,
                "dropout": 0.0,
                "target_modules": ["q_proj"],
                "adapter_name": "default",
            },
            optimizers={
                "lora": {
                    "opt_config": {"lr": 0.1},
                    "scheduler_config": {"start_factor": 1.0, "end_factor": 1.0},
                },
                "packet_wrapper": {
                    "opt_config": {"lr": 0.1},
                    "scheduler_config": {"start_factor": 1.0, "end_factor": 1.0},
                },
            },
        ))
        optimizers, schedulers = run_train.build_optimizers(
            train_config, model, packet_wrapper, num_samples=2
        )
        for optimizer in optimizers.values():
            optimizer.zero_grad()
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.grad = torch.ones_like(param)
            optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()

        expected_lora = model.lora_weight.detach().clone()
        expected_header = packet_wrapper.header.detach().clone()
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "checkpoint.pt")
            save_training_checkpoint(
                path,
                next_epoch=1,
                epoch_indices=[1, 0],
                model=model,
                packet_wrapper=packet_wrapper,
                optimizers=optimizers,
                schedulers=schedulers,
            )

            restored_model = TinyTrainModel()
            restored_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
            configure_trainable_parameters(
                restored_model, restored_wrapper, ["lora", "packet_wrapper"]
            )
            state = load_training_checkpoint(
                path, model=restored_model, packet_wrapper=restored_wrapper
            )
            restored_optimizers, restored_schedulers = run_train.build_optimizers(
                train_config, restored_model, restored_wrapper, num_samples=2
            )
            next_epoch, epoch_indices = restore_training_state(
                state,
                optimizers=restored_optimizers,
                schedulers=restored_schedulers,
            )

        self.assertEqual(next_epoch, 1)
        self.assertEqual(epoch_indices, [1, 0])
        self.assertTrue(torch.equal(restored_model.lora_weight, expected_lora))
        self.assertTrue(torch.equal(restored_wrapper.header, expected_header))
        self.assertEqual(
            restored_schedulers["lora"].state_dict(),
            schedulers["lora"].state_dict(),
        )

    def test_checkpoint_preserves_enabled_frozen_component(self):
        model = TinyTrainModel()
        packet_wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        configure_trainable_parameters(model, packet_wrapper, ["packet_wrapper"])
        expected_lora = model.lora_weight.detach().clone()
        with tempfile.TemporaryDirectory() as output_dir:
            path = os.path.join(output_dir, "checkpoint.pt")
            optimizer = torch.optim.AdamW(
                [packet_wrapper.header, packet_wrapper.trailer], lr=0.1
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=1.0
            )
            save_training_checkpoint(
                path,
                next_epoch=1,
                epoch_indices=[0],
                model=model,
                packet_wrapper=packet_wrapper,
                optimizers={"packet_wrapper": optimizer},
                schedulers={"packet_wrapper": scheduler},
            )
            model.lora_weight.data.zero_()
            load_training_checkpoint(path, model=model, packet_wrapper=packet_wrapper)

        self.assertTrue(torch.equal(model.lora_weight, expected_lora))

    def test_timestamp_run_directory_and_collision(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "sempic"
            run_dir = run_train.allocate_train_run_dir(
                method_root,
                "r16-qkvo",
                now=datetime(2026, 7, 22, 16, 45, 0),
            )
            self.assertEqual(run_dir.name, "20260722_164500_r16-qkvo")

            collision = run_train.allocate_train_run_dir(
                method_root,
                "r16-qkvo",
                now=datetime(2026, 7, 22, 16, 45, 0),
            )
            self.assertEqual(collision.name, "20260722_164500_r16-qkvo_1")

    def test_run_one_config_writes_resolved_snapshot_in_timestamp_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "joint"
            config_path = Path(temp_dir) / "train.json"
            config_path.write_text(json.dumps(build_train_config_dict(
                output_dir=str(method_root),
                run_suffix="from-config",
            )))

            def complete_with_checkpoint(train_config, *args, **kwargs):
                del args, kwargs
                torch.save({}, Path(train_config["output_dir"]) / "checkpoint.pt")

            with mock.patch.object(
                run_train,
                "train_one_config",
                side_effect=complete_with_checkpoint,
            ) as train_one:
                run_train.run_one_config(
                    str(config_path),
                    run_train.TrainCache(),
                    run_suffix_override="from-cli",
                    attention_backend_override="sdpa",
                )

            run_dirs = list(method_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertRegex(run_dir.name, r"^\d{8}_\d{6}_from-cli$")
            snapshot = json.loads((run_dir / "train_config.json").read_text())
            self.assertEqual(snapshot["output_dir"], str(run_dir))
            self.assertEqual(snapshot["run_suffix"], "from-cli")
            self.assertEqual(snapshot["attention_backend"], "sdpa")
            self.assertTrue((run_dir / "run.log").is_file())
            self.assertFalse((run_dir / "checkpoint.pt").exists())
            self.assertFalse(os.path.lexists(method_root / "latest"))
            train_one.assert_called_once()

    def test_failed_training_keeps_checkpoint_without_latest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "joint"
            config_path = Path(temp_dir) / "train.json"
            config_path.write_text(json.dumps(build_train_config_dict(
                output_dir=str(method_root)
            )))

            def fail_after_checkpoint(train_config, *args, **kwargs):
                del args, kwargs
                torch.save({}, Path(train_config["output_dir"]) / "checkpoint.pt")
                raise RuntimeError("training failed")

            with (
                mock.patch.object(run_train, "train_one_config", side_effect=fail_after_checkpoint),
                self.assertRaisesRegex(RuntimeError, "training failed"),
            ):
                run_train.run_one_config(str(config_path), run_train.TrainCache())

            new_runs = list(method_root.iterdir())
            self.assertEqual(len(new_runs), 1)
            self.assertTrue((new_runs[0] / "checkpoint.pt").is_file())
            self.assertFalse(os.path.lexists(method_root / "latest"))

    def test_resume_reuses_concrete_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "kvpacket"
            run_dir = method_root / "20260722_170000_retry"
            run_dir.mkdir(parents=True)
            torch.save({}, run_dir / "checkpoint.pt")
            config_path = Path(temp_dir) / "train.json"
            config_path.write_text(json.dumps(build_train_config_dict(
                output_dir=str(method_root)
            )))

            with mock.patch.object(run_train, "train_one_config") as train_one:
                run_train.run_one_config(
                    str(config_path),
                    run_train.TrainCache(),
                    resume_from=str(run_dir),
                )

            self.assertEqual(
                train_one.call_args.args[0]["output_dir"], str(run_dir.resolve())
            )
            self.assertFalse((run_dir / "checkpoint.pt").exists())
            self.assertFalse(os.path.lexists(method_root / "latest"))

    def test_resume_rejects_symlink_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "kvpacket"
            run_dir = method_root / "20260722_170000_retry"
            run_dir.mkdir(parents=True)
            alias = method_root / "alias"
            alias.symlink_to(run_dir.name, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "concrete timestamp run directory"):
                run_train.select_resume_run_dir(method_root, alias)


if __name__ == "__main__":
    unittest.main()
