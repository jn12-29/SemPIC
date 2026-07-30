import types
import unittest
from unittest.mock import patch

import torch
from transformers import DynamicCache, PretrainedConfig

from sempic.cache import KVCache, KeyValue
from sempic.cache_comb.methods.sam_kv import (
    _fold_sam_prefix,
    _leave_one_out_mean,
    _ordered_prefix_ids,
    _reduce_gqa_query_heads,
    _sam_layer_indices,
    _sam_peer_parts,
)
from sempic.cache_comb.methods.sink import sink_eval
from sempic.prompt import TokenSpan, TokenizedPrompt


def make_cache(tokens: list[int], positions: list[int]) -> KVCache:
    cache = KVCache()
    key = torch.tensor(tokens, dtype=torch.float).view(1, 1, -1, 1)
    cache.update(
        0,
        KeyValue(
            key=key,
            value=torch.zeros_like(key),
            position_ids=torch.tensor([positions]),
        ),
    )
    return cache


class FakeCausalModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = PretrainedConfig()
        self.config.num_hidden_layers = 1
        self.model = self
        self.forward_calls: list[dict[str, object]] = []
        self.generate_past: list[float] = []

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, *, input_ids, past_key_values=None, position_ids=None, **kwargs):
        del kwargs
        batch_size = input_ids.size(0)
        if past_key_values is None:
            old_key = torch.empty((batch_size, 1, 0, 1))
            old_value = torch.empty_like(old_key)
        else:
            old_key = past_key_values.layers[0].keys
            old_value = past_key_values.layers[0].values
        new_key = input_ids.float().view(batch_size, 1, -1, 1)
        cache = DynamicCache(config=self.config)
        cache.update(
            torch.cat([old_key, new_key], dim=2),
            torch.cat([old_value, torch.zeros_like(new_key)], dim=2),
            0,
        )
        self.forward_calls.append({
            "input_ids": input_ids.clone(),
            "position_ids": None if position_ids is None else position_ids.clone(),
            "past": old_key[..., 0].clone(),
        })
        return types.SimpleNamespace(
            past_key_values=cache,
            logits=torch.zeros((batch_size, 1, 100)),
        )

    def generate(self, *, input_ids, past_key_values, **kwargs):
        del kwargs
        self.generate_past = past_key_values.layers[0].keys[0, 0, :, 0].tolist()
        return torch.cat([input_ids, torch.tensor([[99]])], dim=1)


class SelectionInlineSemanticsTests(unittest.TestCase):
    def test_sam_gqa_reduces_each_contiguous_query_head_group(self):
        query = torch.tensor([1.0, 3.0, 10.0, 14.0]).reshape(1, 4, 1)

        reduced = _reduce_gqa_query_heads(query, n_key_heads=2)

        self.assertEqual(reduced.squeeze(0).tolist(), [[2.0], [12.0]])

    def test_sam_single_peer_has_zero_leave_one_out_mean(self):
        only_query = torch.tensor([[[2.0, 4.0]]])
        generic_query = torch.tensor([[[10.0, 20.0]]])

        peer_mean = _leave_one_out_mean(only_query, only_query, count=1)

        self.assertTrue(torch.equal(peer_mean, torch.zeros_like(only_query)))
        self.assertTrue(torch.equal(peer_mean + generic_query, generic_query))
        self.assertEqual(
            _leave_one_out_mean(
                torch.tensor([[[14.0]]]),
                torch.tensor([[[4.0]]]),
                count=3,
            ).item(),
            5.0,
        )

    def test_sam_prefix_inline_is_a_peer_but_only_contexts_are_targets(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 20, 21, 30, 40, 50, 51]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("context", 4, 5),
                TokenSpan("inline", 5, 7),
            ),
        )

        peers, target_peer_indices = _sam_peer_parts(prompt)

        self.assertEqual([part_index for part_index, _ in peers], [0, 1, 3, 4])
        self.assertEqual(target_peer_indices, [1, 3])
        self.assertNotIn(5, [part_index for part_index, _ in peers])

        peer_values = torch.tensor([10.0, 20.0, 40.0, 80.0])
        total = peer_values.sum()
        self.assertAlmostEqual(
            _leave_one_out_mean(total, peer_values[1], len(peer_values)).item(),
            (10.0 + 40.0 + 80.0) / 3,
            places=5,
        )
        self.assertAlmostEqual(
            _leave_one_out_mean(total, peer_values[3], len(peer_values)).item(),
            (10.0 + 20.0 + 40.0) / 3,
            places=5,
        )

    def test_sam_ordered_prefix_ids_preserve_inline_positions(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 20, 21, 11, 30, 12]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("context", 4, 5),
                TokenSpan("inline", 5, 6),
            ),
        )

        ordered = _ordered_prefix_ids(
            prompt,
            [torch.tensor([[20]]), torch.tensor([[30]])],
        )

        self.assertEqual(ordered.tolist(), [[10, 20, 11, 30]])

    def test_sam_final_layer_fuses_only_context_and_keeps_inline_query_active(self):
        active, fused = _sam_layer_indices(
            context_physical_indices=[[1, 2], [4, 5]],
            context_relative_indices=[[1], [0]],
            inline_physical_indices=[0, 3],
            query_indices=[6, 7],
        )

        self.assertEqual(active, [0, 2, 3, 4, 6, 7])
        self.assertEqual(fused, [2, 4])

    def test_sam_fold_prefills_every_inline_without_making_it_a_context_unit(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 20, 21, 11, 30, 12]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("context", 4, 5),
                TokenSpan("inline", 5, 6),
            ),
        )
        model = FakeCausalModel()

        flops = types.SimpleNamespace(body_flops=lambda **_kwargs: 0)
        with patch(
            "sempic.cache_comb.methods.sam_kv.AutoFlopsCalculator",
            return_value=flops,
        ):
            state, context_indices, inline_indices, positions, num_flops = _fold_sam_prefix(
                model,
                prompt,
                [make_cache([20, 21], [1, 2]), make_cache([30], [4])],
            )

        self.assertEqual(state[0].key[0, 0, :, 0].tolist(), [10.0, 20.0, 21.0, 11.0, 30.0])
        self.assertEqual(context_indices, [[1, 2], [4]])
        self.assertEqual(inline_indices, [0, 3])
        self.assertEqual(positions.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(num_flops, 0)
        self.assertEqual(
            [call["input_ids"].tolist() for call in model.forward_calls],
            [[[10]], [[11]]],
        )

    def test_sink_prefills_middle_inline_between_batched_context_blocks(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 20, 21, 11, 30, 12, 13]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("context", 4, 5),
                TokenSpan("inline", 5, 7),
            ),
        )
        model = FakeCausalModel()
        tokenizer = types.SimpleNamespace(pad_token_id=0)
        flops = types.SimpleNamespace(
            body_flops=lambda **_kwargs: 0,
            forward_flops=lambda **_kwargs: 0,
        )

        with (
            patch(
                "sempic.cache_comb.methods.sink.get_kv_caches",
                return_value=[make_cache([10], [0])],
            ),
            patch(
                "sempic.cache_comb.methods.sink.AutoFlopsCalculator",
                return_value=flops,
            ),
        ):
            result = sink_eval(
                model=model,
                tokenizer=tokenizer,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={},
                answer="answer",
            )

        self.assertEqual(
            result.past_key_values.layers[0].keys[0, 0, :, 0].tolist(),
            [10.0, 20.0, 21.0, 11.0, 30.0, 12.0, 13.0],
        )
        middle_inline_call = model.forward_calls[1]
        self.assertEqual(middle_inline_call["input_ids"].tolist(), [[11]])
        self.assertEqual(middle_inline_call["past"].flatten().tolist(), [10.0, 20.0, 21.0])
        self.assertEqual(middle_inline_call["position_ids"].tolist(), [[3]])

if __name__ == "__main__":
    unittest.main()
