import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sempic.attention_metrics.query_pass import (
    QueryPassForwardProvider,
    _stream_compact_terminal_basis,
)
from sempic.cache import KVCache
from sempic.cache_comb.compact_prefill import CompactPrefillExecutor
from sempic.prompt import TokenSpan, TokenizedPrompt


class Tokenizer:
    pad_token_id = 0


def tiny_model(num_layers: int = 2) -> LlamaForCausalLM:
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    ))
    model.eval()
    return model


class AttentionQueryPassTests(unittest.TestCase):
    def test_provider_exposes_model_shape_properties(self):
        provider = QueryPassForwardProvider.__new__(QueryPassForwardProvider)
        provider.model = tiny_model(num_layers=3)

        self.assertEqual(provider.layer_count, 3)
        self.assertEqual(provider.query_head_count, 4)

    def test_terminal_pass_executes_all_inline_and_emits_selected_rows_once_per_layer(self):
        model = tiny_model()
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 5),
            ),
        )
        seen = []

        shape = _stream_compact_terminal_basis(
            method_name="full_recompute",
            executor=CompactPrefillExecutor(model, backend="observable_eager"),
            tokenizer=Tokenizer(),
            prompt=prompt,
            prepared_kvs={},
            header_len=0,
            trailer_len=0,
            basis_sink=seen.append,
        )

        self.assertEqual(len(seen), model.config.num_hidden_layers)
        self.assertEqual([event.layer_index for event in seen], [0, 1])
        self.assertEqual(shape.layer_count, 2)
        self.assertEqual(shape.query_count, 2)
        self.assertEqual(shape.query_head_count, 4)
        for event in seen:
            self.assertEqual(tuple(event.attention_probabilities.shape), (4, 2, 5))
            self.assertEqual(tuple(event.physical_values.shape), (2, 5, 4))
            self.assertEqual(event.query_to_kv_head.tolist(), [0, 0, 1, 1])
            self.assertEqual(event.layout.physical_length, 5)
            self.assertEqual(
                [(chunk.pic_start, chunk.pic_end) for chunk in event.layout.chunks],
                [(1, 3)],
            )
            torch.testing.assert_close(
                event.attention_probabilities.sum(dim=-1),
                torch.ones((4, 2)),
            )
            torch.testing.assert_close(
                event.attention_probabilities[:, 0, -1],
                torch.zeros(4),
            )

    def test_pic_terminal_pass_uses_interleaved_context_scope(self):
        model = tiny_model(num_layers=1)
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([1, 2, 3, 4, 5]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 5),
            ),
        )
        with torch.no_grad():
            output = model(input_ids=torch.tensor([[2, 3]]), use_cache=True)
        prepared = {1: KVCache.from_hf_cache(
            output.past_key_values,
            position_ids=torch.arange(2).unsqueeze(0),
        )}
        seen = []

        shape = _stream_compact_terminal_basis(
            method_name="no_recompute",
            executor=CompactPrefillExecutor(model, backend="observable_eager"),
            tokenizer=Tokenizer(),
            prompt=prompt,
            prepared_kvs=prepared,
            header_len=0,
            trailer_len=0,
            basis_sink=seen.append,
        )

        self.assertEqual(shape.query_count, 2)
        self.assertEqual(shape.layout.physical_length, 5)
        self.assertEqual(
            [
                (chunk.pic_start, chunk.pic_end,
                 chunk.scope_start, chunk.scope_end)
                for chunk in shape.layout.chunks
            ],
            [(1, 3, 1, 3)],
        )
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(
            seen[0].attention_probabilities[:, 0, -1],
            torch.zeros(4),
        )

    def test_full_terminal_pass_preserves_multiple_context_scopes(self):
        model = tiny_model(num_layers=1)
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([1, 2, 3, 4, 5, 6]),
            parts=(
                TokenSpan("context", 0, 2),
                TokenSpan("inline", 2, 3),
                TokenSpan("context", 3, 4),
                TokenSpan("inline", 4, 6),
            ),
        )
        seen = []

        _stream_compact_terminal_basis(
            method_name="full_recompute",
            executor=CompactPrefillExecutor(model, backend="observable_eager"),
            tokenizer=Tokenizer(),
            prompt=prompt,
            prepared_kvs={},
            header_len=0,
            trailer_len=0,
            basis_sink=seen.append,
        )

        self.assertEqual(len(seen), 1)
        self.assertEqual(
            [
                (chunk.pic_start, chunk.pic_end,
                 chunk.scope_start, chunk.scope_end)
                for chunk in seen[0].layout.chunks
            ],
            [(0, 2, 0, 2), (3, 4, 3, 4)],
        )


if __name__ == "__main__":
    unittest.main()
