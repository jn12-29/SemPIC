import types
import unittest
from unittest import mock

import torch

from sempic.evaluation import runtime


def base_config() -> dict:
    return {
        "model": {
            "model_path": "model",
            "dtype": "float32",
            "device": "cpu",
            "generation_kwargs": {"max_new_tokens": 3},
        },
        "dataset": {
            "dataset_name": "biography",
            "num_samples": 2,
            "num_data_strs": 3,
            "num_shots": 1,
            "subset": "1k",
            "split": "test",
            "seed": 7,
            "data_kwargs": {"key": "value"},
            "template": "default",
            "template_kwargs": {"flag": True},
        },
        "cache_comb": {"method": "no_recompute", "kwargs": {"ratio": 0.5}},
        "seed": 11,
    }


class EvalResourceTests(unittest.TestCase):
    def test_load_eval_config_preserves_defaults_and_artifact_validation(self):
        loaded = runtime.load_eval_config(base_config())

        self.assertEqual(loaded["model"]["generation_kwargs"], {"max_new_tokens": 3})
        self.assertEqual(loaded["dataset"]["template_kwargs"], {"flag": True})
        self.assertIsNone(loaded["packet_wrapper"]["path"])
        self.assertIsNone(loaded["lora"]["path"])
        self.assertIsNone(loaded["compress"])
        self.assertIsNone(loaded["quantization"])
        self.assertEqual(loaded["logging"]["level"], "INFO")
        self.assertFalse(loaded["debug_dump"]["enabled"])

        invalid = base_config()
        invalid["cache_comb"] = {"method": "sempic", "kwargs": {}}
        with self.assertRaisesRegex(ValueError, "requires lora.path"):
            runtime.load_eval_config(invalid)

    def test_resources_preserve_cache_key_and_model_initialization_side_effects(self):
        class FakeTokenizer:
            padding_side = "right"
            pad_token_id = None
            eos_token_id = 2

        class FakeModel:
            def __init__(self):
                self.device = torch.device("cpu")
                self.generation_config = types.SimpleNamespace(pad_token_id=None)

        model = FakeModel()
        tokenizer = FakeTokenizer()
        config = runtime.load_eval_config(base_config())
        cache = runtime.create_eval_resource_cache()

        with (
            mock.patch.object(
                runtime.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ) as load_model,
            mock.patch.object(
                runtime.AutoTokenizer,
                "from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
        ):
            first = runtime.load_eval_resources(config, cache)
            second = runtime.load_eval_resources(config, cache)

        key = ("model", "float32", "cpu")
        self.assertIs(first.model, model)
        self.assertIs(second.model, model)
        self.assertIs(first.tokenizer, tokenizer)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token_id, 4)
        self.assertEqual(model.generation_config.pad_token_id, 4)
        self.assertIs(cache["model"][key], model)
        self.assertIs(cache["tokenizer"][key], tokenizer)
        load_model.assert_called_once_with(
            "model",
            dtype="float32",
            device_map=torch.device("cpu"),
            low_cpu_mem_usage=True,
        )
        load_tokenizer.assert_called_once_with("model")

    def test_resources_share_base_model_and_use_unique_lora_adapters(self):
        class FakeTokenizer:
            padding_side = "right"
            pad_token_id = 0
            eos_token_id = 2

        class FakeModel:
            def __init__(self):
                self.device = torch.device("cpu")
                self.generation_config = types.SimpleNamespace(pad_token_id=None)

        wrapper = object()
        model = FakeModel()
        cache = runtime.create_eval_resource_cache()
        configs = []
        for lora_path in ("./adapter-a", "./adapter-b"):
            raw = base_config()
            raw["cache_comb"] = {"method": "sempic_kvpacket", "kwargs": {}}
            raw["packet_wrapper"] = {"path": "./wrapper.pt"}
            raw["lora"] = {"path": lora_path}
            configs.append(runtime.load_eval_config(raw))

        with (
            mock.patch.object(
                runtime.AutoModelForCausalLM,
                "from_pretrained",
                return_value=model,
            ),
            mock.patch.object(
                runtime.AutoTokenizer,
                "from_pretrained",
                return_value=FakeTokenizer(),
            ),
            mock.patch.object(runtime, "load_wrapper", return_value=wrapper) as load_wrapper,
            mock.patch.object(runtime, "load_lora_adapter_for_eval") as load_lora,
        ):
            resources = [runtime.load_eval_resources(config, cache) for config in configs]

        self.assertEqual(len(cache["model"]), 1)
        self.assertIs(resources[0].model, resources[1].model)
        self.assertIs(resources[0].packet_wrapper, wrapper)
        self.assertIs(resources[1].packet_wrapper, wrapper)
        adapter_names = [resource.lora_adapter_name for resource in resources]
        self.assertEqual(len(set(adapter_names)), 2)
        self.assertTrue(all(
            name.startswith(runtime.LORA_EVAL_ADAPTER_NAME) for name in adapter_names
        ))
        load_wrapper.assert_called_once()
        self.assertEqual(
            [call.args[1] for call in load_lora.call_args_list],
            ["./adapter-a", "./adapter-b"],
        )

    def test_dataset_generation_and_compressor_building_forward_config(self):
        config = runtime.load_eval_config(base_config())
        tokenizer = object()
        expected_generator = iter([])
        with mock.patch.object(
            runtime,
            "get_ret_eval_generator",
            return_value=expected_generator,
        ) as get_generator:
            actual_generator = runtime.build_eval_generator(config["dataset"], tokenizer)  # type: ignore[arg-type]

        self.assertIs(actual_generator, expected_generator)
        self.assertEqual(get_generator.call_args.kwargs["tokenizer"], tokenizer)
        self.assertEqual(get_generator.call_args.kwargs["data_kwargs"], {"key": "value"})
        self.assertEqual(get_generator.call_args.kwargs["template_kwargs"], {"flag": True})

        fake_compressor = object()
        compressor_cls = mock.Mock(return_value=fake_compressor)
        compress_config = runtime.CompressConfig(
            method="fake",
            compression_ratio=0.25,
            keep_filler_tokens=True,
            kwargs={"option": 1},
        )
        with mock.patch.dict(runtime.PRESS_CLASSES, {"fake": compressor_cls}):
            compressor, keep_filler_tokens = runtime.build_compressor(compress_config)

        self.assertIs(compressor, fake_compressor)
        self.assertTrue(keep_filler_tokens)
        compressor_cls.assert_called_once_with(compression_ratio=0.25, option=1)
        self.assertEqual(runtime.build_compressor(None), (None, False))


if __name__ == "__main__":
    unittest.main()
