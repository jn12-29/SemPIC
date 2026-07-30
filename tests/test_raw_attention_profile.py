import unittest

import torch

from sempic.attention_metrics.basis import (
    LayerAttentionBasis,
    PhysicalChunkScope,
    PhysicalLayout,
)
from sempic.attention_metrics.reducers import RawAttentionProfileReducer


def _event(layer_index: int, probabilities: torch.Tensor) -> LayerAttentionBasis:
    key_count = probabilities.size(2)
    return LayerAttentionBasis(
        layer_index=layer_index,
        scaled_masked_logits=probabilities.log(),
        attention_probabilities=probabilities,
        physical_values=torch.ones((1, key_count, 2)),
        query_to_kv_head=torch.zeros(
            probabilities.size(0), dtype=torch.long
        ),
        keep_mask=torch.ones(
            (1, probabilities.size(1), key_count), dtype=torch.bool
        ),
        layout=PhysicalLayout(
            key_count,
            (PhysicalChunkScope("chunk", "digest", 0, key_count, 0, key_count),),
        ),
    )


class RawAttentionProfileReducerTests(unittest.TestCase):
    def test_profiles_are_each_methods_own_head_and_query_mean(self):
        methods = ("full_recompute", "sempic")
        reducer = RawAttentionProfileReducer(methods)
        full = torch.tensor(
            [
                [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]],
                [[0.4, 0.4, 0.2], [0.8, 0.1, 0.1]],
            ],
            dtype=torch.float32,
        )
        sempic = torch.tensor(
            [
                [[0.1, 0.2, 0.7], [0.1, 0.3, 0.6]],
                [[0.2, 0.2, 0.6], [0.2, 0.4, 0.4]],
            ],
            dtype=torch.bfloat16,
        )
        for method, values in (("full_recompute", full), ("sempic", sempic)):
            reducer.consume(method, _event(0, values))
            reducer.consume(method, _event(1, values.flip(-1)))
            reducer.finish_method(method)

        output = reducer.finish_sample()["chunk"]
        self.assertEqual(tuple(output), methods)
        self.assertEqual(set(output["sempic"]), {"raw"})
        self.assertEqual(output["sempic"]["raw"].dtype, torch.float32)
        self.assertEqual(output["sempic"]["raw"].device.type, "cpu")
        torch.testing.assert_close(
            output["full_recompute"]["raw"][0], full.mean(dim=(0, 1))
        )
        torch.testing.assert_close(
            output["sempic"]["raw"][0],
            sempic.mean(dim=(0, 1), dtype=torch.float32),
        )
        torch.testing.assert_close(
            output["sempic"]["raw"][1],
            sempic.flip(-1).mean(dim=(0, 1), dtype=torch.float32),
        )

    def test_rejects_incomplete_layers_and_query_count_changes(self):
        probabilities = torch.tensor([[[0.4, 0.6]]], dtype=torch.float32)
        reducer = RawAttentionProfileReducer(("full_recompute", "sempic"))
        reducer.consume("full_recompute", _event(0, probabilities))
        reducer.consume("full_recompute", _event(1, probabilities))
        reducer.finish_method("full_recompute")
        reducer.consume("sempic", _event(0, probabilities))
        reducer.finish_method("sempic")
        with self.assertRaisesRegex(ValueError, "different layer counts"):
            reducer.finish_sample()

        reducer = RawAttentionProfileReducer(("full_recompute", "sempic"))
        reducer.consume("full_recompute", _event(0, probabilities))
        changed_queries = torch.tensor(
            [[[0.4, 0.6], [0.7, 0.3]]], dtype=torch.float32
        )
        with self.assertRaisesRegex(ValueError, "query count changed"):
            reducer.consume("sempic", _event(0, changed_queries))


if __name__ == "__main__":
    unittest.main()
