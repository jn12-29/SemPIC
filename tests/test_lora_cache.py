import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import run_eval as run_eval_entry
from run_eval import LORA_EVAL_ADAPTER_NAME, load_eval_config, load_lora_adapter_for_eval
from sempic.cache import KVCache
from sempic.cache_comb import PrefillResult
from sempic.cache_comb import methods as cache_comb_methods
from sempic.cache_comb.methods import get_cache_comb_func
from sempic.evaluation import preparation as eval_preparation
from sempic.packet_wrapper import PacketWrapper
from sempic.prompt import ContextBlock, Inline, PromptSequence
from sempic.utils.lora import lora_adapters_enabled
from sempic.utils.train import LoRAConfigDict, build_peft_lora_model, save_lora_adapter


def build_tiny_base_model():
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(config)
    model.config._attn_implementation = "eager"
    return model


def build_tiny_lora_model():
    model = build_tiny_base_model()
    lora_config = LoRAConfigDict(
        enabled=True,
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        init_path=None,
        adapter_name="default",
    )
    return build_peft_lora_model(model, lora_config)


class LoRACacheTests(unittest.TestCase):
    def test_save_lora_adapter_writes_lora_subdirectory(self):
        model = build_tiny_lora_model()
        with tempfile.TemporaryDirectory() as save_path:
            adapter_path = os.path.join(save_path, "lora")
            save_lora_adapter(model, adapter_path, adapter_name="default")

            self.assertFalse(os.path.exists(os.path.join(save_path, "adapter_config.json")))
            self.assertTrue(os.path.exists(os.path.join(adapter_path, "adapter_config.json")))

            eval_model = build_tiny_base_model()
            load_lora_adapter_for_eval(eval_model, adapter_path)
            self.assertIn(LORA_EVAL_ADAPTER_NAME, eval_model.peft_config)
            with lora_adapters_enabled(eval_model, adapter_name=LORA_EVAL_ADAPTER_NAME):
                pass

    def test_eval_config_validation_uses_algorithm_specific_artifact_paths(self):
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
            "cache_comb": {"method": "kvpacket", "kwargs": {}},
            "seed": 0,
        }

        with self.assertRaisesRegex(ValueError, "requires packet_wrapper.path"):
            load_eval_config(base_config)

        with self.assertRaisesRegex(ValueError, "Legacy LoRA"):
            load_eval_config({
                **base_config,
                "lora" + "_adapter": "./adapter/lora_kv_cache",
            })

        with self.assertRaisesRegex(ValueError, "Unsupported cache_comb.method"):
            load_eval_config({
                **base_config,
                "cache_comb": {"method": "kv_packet", "kwargs": {}},
                "packet_wrapper": {"path": "./wrapper.pt"},
            })

        packet_only = load_eval_config({
            **base_config,
            "packet_wrapper": {"path": "./wrapper.pt"},
        })
        self.assertEqual(packet_only["packet_wrapper"]["path"], "./wrapper.pt")
        self.assertIsNone(packet_only["lora"]["path"])

        lora_only = load_eval_config({
            **base_config,
            "cache_comb": {"method": "sempic", "kwargs": {}},
            "lora": {"path": "./adapter/lora_kv_cache"},
        })
        self.assertEqual(lora_only["lora"]["path"], "./adapter/lora_kv_cache")

        with self.assertRaisesRegex(ValueError, "lora.path is only valid"):
            load_eval_config({
                **base_config,
                "packet_wrapper": {"path": "./wrapper.pt"},
                "lora": {"path": "./adapter/lora_kv_cache"},
            })

        with self.assertRaisesRegex(ValueError, "requires packet_wrapper.path and lora.path"):
            load_eval_config({
                **base_config,
                "cache_comb": {"method": "sempic_kvpacket", "kwargs": {}},
                "packet_wrapper": {"path": "./wrapper.pt"},
            })

        joint = load_eval_config({
            **base_config,
            "cache_comb": {"method": "sempic_kvpacket", "kwargs": {}},
            "packet_wrapper": {"path": "./wrapper.pt"},
            "lora": {"path": "./adapter/lora_kv_cache"},
        })
        self.assertEqual(joint["packet_wrapper"]["path"], "./wrapper.pt")
        self.assertEqual(joint["lora"]["path"], "./adapter/lora_kv_cache")

        method_funcs = {
            name: get_cache_comb_func(name)
            for name in ("kvpacket", "sempic", "sempic_kvpacket")
        }
        self.assertTrue(all(callable(func) for func in method_funcs.values()))
        self.assertEqual(len({id(func) for func in method_funcs.values()}), 3)

    def test_packet_wrapper_dtype_move_preserves_trainable_leaf_tensors(self):
        wrapper = PacketWrapper(1, 1, 4, dtype=torch.float64)

        wrapper.to(dtype=torch.float32)
        optimizer = torch.optim.AdamW([wrapper.header, wrapper.trailer], lr=0.1)

        self.assertTrue(wrapper.header.is_leaf)
        self.assertTrue(wrapper.trailer.is_leaf)
        self.assertTrue(wrapper.header.requires_grad)
        self.assertTrue(wrapper.trailer.requires_grad)
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_lora_packet_eval_prepares_context_from_canonical_ids(self):
        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, texts, **kwargs):
                del kwargs
                return {
                    "input_ids": [[ord(char) for char in text] for text in texts]
                }

        class FakeBody:
            def embed_tokens(self, input_ids):
                return torch.zeros((input_ids.size(0), input_ids.size(1), 4))

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(1))
                self.device = torch.device("cpu")

        captured_kv_calls = []
        captured_method_calls = []

        def fake_get_kv_caches(**kwargs):
            captured_kv_calls.append(kwargs)
            length = int(kwargs["attention_mask"][0].sum().item())
            return [KVCache.create_dummy(1, 1, 1, length, 1, 1)]

        def fake_cache_comb_func(**kwargs):
            captured_method_calls.append(kwargs)
            return PrefillResult(
                logits=torch.zeros((1, 1, 3)),
                past_key_values=mock.MagicMock(),
                generation_input_ids=torch.tensor([[1]]),
                position_ids=torch.tensor([[0]]),
                attention_mask=torch.ones((1, 1), dtype=torch.long),
                flops=0,
            )

        eval_entries = iter([{
            "query": "q",
            "answer": "a",
            "prompt": PromptSequence((
                ContextBlock("doc"),
                Inline(" prompt"),
            )),
        }])
        packet_wrapper = PacketWrapper(1, 1, 4, device=torch.device("cpu"))
        disabled_context_calls = []

        @contextmanager
        def fake_lora_disabled(model):
            disabled_context_calls.append(model)
            yield

        @contextmanager
        def fake_lora_enabled(model, adapter_name=None):
            del model, adapter_name
            yield

        with (
            mock.patch.object(eval_preparation, "get_causal_lm_body", return_value=FakeBody()),
            mock.patch.object(eval_preparation, "get_kv_caches", side_effect=fake_get_kv_caches),
            mock.patch.object(run_eval_entry, "lora_adapters_disabled", side_effect=fake_lora_disabled),
            mock.patch.object(eval_preparation, "lora_adapters_enabled", side_effect=fake_lora_enabled),
            mock.patch.object(
                run_eval_entry,
                "generate_from_prefill",
                return_value=(torch.tensor([[1, 2]]), 0.0),
            ),
            mock.patch.object(run_eval_entry, "get_answers", return_value=["a"]),
        ):
            with mock.patch.dict(
                cache_comb_methods.CACHE_COMB_FUNC_DICT,
                {"sempic_kvpacket": fake_cache_comb_func},
            ):
                result = run_eval_entry.run_eval(
                    model=FakeModel(),  # type: ignore[arg-type]
                    tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
                    eval_generator=eval_entries,
                    cache_comb_method="sempic_kvpacket",
                    cache_comb_kwargs={},
                    packet_wrapper=packet_wrapper,
                    lora_adapter_name=LORA_EVAL_ADAPTER_NAME,
                )

        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(len(captured_kv_calls), 1)
        self.assertEqual(captured_kv_calls[0]["input_embeds"].shape[1], 5)
        prepared = captured_method_calls[0]["prepared_kvs"]
        self.assertEqual(list(prepared), [0])
        self.assertEqual(len(disabled_context_calls), 1)


if __name__ == "__main__":
    unittest.main()
