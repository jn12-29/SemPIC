import unittest

import torch

from sempic.attention_metrics.basis import (
    LayerAttentionBasis,
    PhysicalChunkScope,
    PhysicalLayout,
)
from sempic.attention_metrics.reducers import (
    AttentionProfileReducer,
    PicRetrievalReducer,
)


def event(*, layer, probabilities, values, scope_start=0, scope_end=None, chunks=None):
    keep = torch.ones(
        (1, probabilities.size(1), probabilities.size(2)), dtype=torch.bool
    )
    return LayerAttentionBasis(
        layer_index=layer,
        scaled_masked_logits=probabilities.log(),
        attention_probabilities=probabilities,
        physical_values=values,
        query_to_kv_head=torch.tensor([0, 0, 1, 1]),
        keep_mask=keep,
        layout=PhysicalLayout(
            physical_length=probabilities.size(2),
            chunks=chunks or (PhysicalChunkScope(
                chunk_id="part-0",
                token_digest="tokens",
                pic_start=1,
                pic_end=3,
                scope_start=scope_start,
                scope_end=probabilities.size(2) if scope_end is None else scope_end,
            ),),
        ),
    )


class PicRetrievalReducerTests(unittest.TestCase):
    def test_multiple_chunks_cosine_and_zero_low_mass_are_separate(self):
        methods = ("full_recompute", "vanilla_pic")
        reducer = PicRetrievalReducer(methods)
        chunks = (
            PhysicalChunkScope("orthogonal", "a", 0, 2, 0, 2),
            PhysicalChunkScope("low_mass", "b", 2, 4, 2, 4),
            PhysicalChunkScope("zero_mass", "c", 4, 5, 4, 5),
        )
        full_probabilities = torch.tensor(
            [[[1.0, 0.0, 1e-8, 0.0, 0.0]]] * 4
        ).squeeze(0)
        candidate_probabilities = torch.tensor(
            [[[0.0, 1.0, 1e-8, 0.0, 0.0]]] * 4
        ).squeeze(0)
        values = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
        ])
        reducer.consume("full_recompute", event(
            layer=0, probabilities=full_probabilities, values=values, chunks=chunks,
        ))
        reducer.finish_method("full_recompute")
        reducer.consume("vanilla_pic", event(
            layer=0, probabilities=candidate_probabilities, values=values, chunks=chunks,
        ))
        reducer.finish_method("vanilla_pic")
        output = reducer.finish_sample()

        self.assertEqual(set(output), {"orthogonal", "low_mass", "zero_mass"})
        torch.testing.assert_close(
            output["orthogonal"]["vanilla_pic"]["cosine_distance_sum"],
            torch.ones((1, 4)),
        )
        torch.testing.assert_close(
            output["low_mass"]["vanilla_pic"]["cosine_distance_sum"],
            torch.zeros((1, 4)),
        )
        torch.testing.assert_close(
            output["zero_mass"]["vanilla_pic"]["cosine_valid_count"],
            torch.zeros((1, 4), dtype=torch.int64),
        )

    def test_wrapper_scope_and_gqa_heads_are_preserved(self):
        methods = ("full_recompute", "sempic", "kvpacket")
        reducer = PicRetrievalReducer(methods)
        full_probabilities = torch.zeros((4, 2, 4))
        full_probabilities[:, :, 1:3] = 0.5
        full_values = torch.zeros((2, 4, 2))
        full_values[0, 1] = torch.tensor([1.0, 0.0])
        full_values[0, 2] = torch.tensor([0.0, 1.0])
        full_values[1, 1] = torch.tensor([2.0, 0.0])
        full_values[1, 2] = torch.tensor([0.0, 2.0])

        candidate_probabilities = torch.full((4, 2, 4), 0.25)
        candidate_values = torch.zeros((2, 4, 2))
        candidate_values[0, :, :] = torch.tensor([0.5, 0.5])
        candidate_values[1, :, :] = torch.tensor([1.0, 1.0])

        reducer.consume("full_recompute", event(
            layer=0,
            probabilities=full_probabilities,
            values=full_values,
            scope_start=1,
            scope_end=3,
        ))
        reducer.finish_method("full_recompute")
        reducer.consume("sempic", event(
            layer=0,
            probabilities=candidate_probabilities,
            values=candidate_values,
            scope_start=1,
            scope_end=3,
        ))
        reducer.finish_method("sempic")
        reducer.consume("kvpacket", event(
            layer=0,
            probabilities=candidate_probabilities,
            values=candidate_values,
            scope_start=0,
            scope_end=4,
        ))
        reducer.finish_method("kvpacket")
        output = reducer.finish_sample()["part-0"]

        self.assertEqual(
            tuple(output["full_recompute"]["reference_energy_sum"].shape), (1, 4)
        )
        torch.testing.assert_close(
            output["kvpacket"]["squared_error_sum"],
            torch.zeros((1, 4), dtype=torch.float32),
        )
        torch.testing.assert_close(
            output["kvpacket"][
                "absolute_mass_error_sum"
            ],
            torch.zeros((1, 4), dtype=torch.float32),
        )
        self.assertTrue(torch.all(
            output["sempic"]["squared_error_sum"] > 0
        ))
        torch.testing.assert_close(
            output["sempic"]["absolute_mass_error_sum"],
            torch.ones((1, 4), dtype=torch.float32),
        )
        self.assertGreater(
            output["full_recompute"]["reference_energy_sum"][0, 2],
            output["full_recompute"]["reference_energy_sum"][0, 0],
        )

    def test_full_reference_preserves_source_dtype_on_cpu(self):
        reducer = PicRetrievalReducer(("full_recompute", "vanilla_pic"))
        probabilities = torch.full((4, 1, 4), 0.25, dtype=torch.bfloat16)
        values = torch.ones((2, 4, 2), dtype=torch.bfloat16)

        reducer.consume("full_recompute", event(
            layer=0,
            probabilities=probabilities,
            values=values,
        ))

        reference, reference_mass = next(iter(reducer._full.values()))
        self.assertEqual(reference.device.type, "cpu")
        self.assertEqual(reference.dtype, torch.bfloat16)
        self.assertEqual(reference_mass.device.type, "cpu")
        self.assertEqual(reference_mass.dtype, torch.bfloat16)

    def test_attention_reducer_derives_raw_and_conditional_from_same_event(self):
        methods = ("full_recompute", "vanilla_pic")
        reducer = AttentionProfileReducer(methods)
        values = torch.tensor([
            [[0.1, 0.6, 0.3, 0.0]],
            [[0.2, 0.4, 0.4, 0.0]],
            [[0.1, 0.2, 0.7, 0.0]],
            [[0.2, 0.5, 0.3, 0.0]],
        ], dtype=torch.bfloat16)
        candidate_values = values.clone()
        full = event(
            layer=0,
            probabilities=values,
            values=torch.ones((2, 4, 2)),
            scope_start=1,
            scope_end=3,
        )
        reducer.consume("full_recompute", full)
        stored_reference = next(iter(reducer._full.values()))
        self.assertEqual(stored_reference.dtype, torch.bfloat16)
        full.attention_probabilities.zero_()
        self.assertGreater(torch.count_nonzero(stored_reference).item(), 0)
        reducer.finish_method("full_recompute")
        reducer.consume("vanilla_pic", event(
            layer=0,
            probabilities=candidate_values,
            values=torch.ones((2, 4, 2)),
            scope_start=1,
            scope_end=3,
        ))
        reducer.finish_method("vanilla_pic")
        output = reducer.finish_sample()["part-0"]
        self.assertEqual(tuple(output["full_recompute"]["raw"].shape), (1, 2))
        self.assertEqual(output["full_recompute"]["raw"].dtype, torch.float32)
        torch.testing.assert_close(
            output["vanilla_pic"]["raw_absolute_error"],
            torch.zeros((1, 2)),
        )
        torch.testing.assert_close(
            output["vanilla_pic"][
                "chunk_conditional_absolute_error"
            ],
            torch.zeros((1, 2)),
        )


if __name__ == "__main__":
    unittest.main()
