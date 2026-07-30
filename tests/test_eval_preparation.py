import types
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from sempic.cache import KVCache
from sempic.evaluation import preparation
from sempic.packet_wrapper import PacketWrapper
from sempic.prompt import TokenSpan, TokenizedPrompt


def make_prompt() -> TokenizedPrompt:
    return TokenizedPrompt(
        input_ids=torch.tensor([10, 11, 20, 12, 30]),
        parts=(
            TokenSpan("context", 0, 2),
            TokenSpan("inline", 2, 3),
            TokenSpan("context", 3, 4),
            TokenSpan("inline", 4, 5),
        ),
    )


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")
        self.model = types.SimpleNamespace(embed_tokens=self.embed_tokens)

    @staticmethod
    def embed_tokens(input_ids):
        return input_ids.float().unsqueeze(-1)


def caches_for_mask(attention_mask: torch.Tensor) -> list[KVCache]:
    return [
        KVCache.create_dummy(1, 1, 1, int(row.sum().item()), 1, 1)
        for row in attention_mask
    ]


class EvalPreparationTests(unittest.TestCase):
    def test_source_routing_uses_prompt_part_indices(self):
        prompt = make_prompt()
        self.assertEqual(
            preparation.prepared_source_parts(prompt, "no_recompute"),
            [(0, 0, 2), (2, 3, 4)],
        )
        self.assertEqual(
            preparation.prepared_source_parts(prompt, "single_cache"),
            [(3, 0, 4)],
        )
        self.assertEqual(preparation.prepared_source_parts(prompt, "full_recompute"), [])

    def test_right_padding_wrapper_lora_mapping_and_token_counters(self):
        model = FakeModel()
        tokenizer = types.SimpleNamespace(pad_token_id=0)
        wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        with torch.no_grad():
            wrapper.header.fill_(100)
            wrapper.trailer.fill_(200)

        captured = {}
        enabled_adapters = []

        def fake_get_kv_caches(**kwargs):
            captured.update(kwargs)
            return caches_for_mask(kwargs["attention_mask"])

        @contextmanager
        def fake_lora_enabled(_model, adapter_name=None):
            enabled_adapters.append(adapter_name)
            yield

        with (
            mock.patch.object(preparation, "get_kv_caches", side_effect=fake_get_kv_caches),
            mock.patch.object(
                preparation,
                "lora_adapters_enabled",
                side_effect=fake_lora_enabled,
            ),
        ):
            result = preparation.prepare_prompt_kvs(
                model=model,  # type: ignore[arg-type]
                tokenizer=tokenizer,  # type: ignore[arg-type]
                prompt=make_prompt(),
                method_name="sempic_kvpacket",
                packet_wrapper=wrapper,
                lora_adapter_name="adapter",
            )

        self.assertEqual(list(result.prepared_kvs), [0, 2])
        self.assertEqual(result.num_orig_tokens, 3)
        self.assertEqual(result.num_wrapped_tokens, 7)
        self.assertEqual(enabled_adapters, ["adapter"])
        self.assertTrue(torch.equal(
            captured["attention_mask"],
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        ))
        self.assertTrue(torch.equal(
            captured["input_embeds"].squeeze(-1),
            torch.tensor([
                [100.0, 10.0, 11.0, 200.0],
                [100.0, 12.0, 200.0, 0.0],
            ]),
        ))

    def test_compression_precedes_quantization_and_preserves_filler_indices(self):
        model = FakeModel()
        tokenizer = types.SimpleNamespace(pad_token_id=0)
        wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        compressor = object()
        events = []

        def fake_get_kv_caches(**kwargs):
            events.append(("compress", kwargs))
            length = kwargs["input_embeds"].size(1)
            return [KVCache.create_dummy(1, 1, 1, length, 1, 1)]

        def fake_quantize(state_dict, **kwargs):
            events.append(("quantize", kwargs))
            return state_dict

        with (
            mock.patch.object(preparation, "get_kv_caches", side_effect=fake_get_kv_caches),
            mock.patch.object(
                preparation,
                "quantize_kv_cache_sd",
                side_effect=fake_quantize,
            ),
        ):
            result = preparation.prepare_prompt_kvs(
                model=model,  # type: ignore[arg-type]
                tokenizer=tokenizer,  # type: ignore[arg-type]
                prompt=make_prompt(),
                method_name="kvpacket",
                packet_wrapper=wrapper,
                compressor=compressor,  # type: ignore[arg-type]
                keep_filler_tokens=True,
                quantization_config={"num_bits": 4, "axis": 0, "group_size": 8},
            )

        self.assertEqual([event[0] for event in events], [
            "compress", "compress", "quantize", "quantize"
        ])
        first_compress = events[0][1]
        second_compress = events[1][1]
        self.assertNotIn("attention_mask", first_compress)
        self.assertEqual(first_compress["indices_to_keep"], [0, 3])
        self.assertEqual(second_compress["indices_to_keep"], [0, 2])
        self.assertEqual(list(result.prepared_kvs), [0, 2])

    def test_no_prep_method_returns_empty_mapping_without_kv_forward(self):
        with mock.patch.object(preparation, "get_kv_caches") as get_kv_caches:
            result = preparation.prepare_prompt_kvs(
                model=FakeModel(),  # type: ignore[arg-type]
                tokenizer=types.SimpleNamespace(pad_token_id=0),  # type: ignore[arg-type]
                prompt=make_prompt(),
                method_name="full_recompute",
            )

        self.assertEqual(result.prepared_kvs, {})
        self.assertEqual(result.num_orig_tokens, 0)
        self.assertEqual(result.num_wrapped_tokens, 0)
        get_kv_caches.assert_not_called()


if __name__ == "__main__":
    unittest.main()
