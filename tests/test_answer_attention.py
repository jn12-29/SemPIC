import unittest
from unittest.mock import patch

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from sempic.attention_metrics.answer import (
    stream_answer_attention,
    stream_shifted_prediction_attention,
    tokenize_gold_answer,
)
from sempic.attention_metrics.basis import PhysicalChunkScope, PhysicalLayout
from sempic.cache import KVCache
from sempic.cache_comb.abc import PrefillResult
from sempic.cache_comb.recompute_kv import recompute_kv
from sempic.cache_comb.recompute_kv import prepare_pos_embed_and_mask


def build_models():
    return (
        LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )),
        Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )),
    )


def prefill_result(model, prompt_ids: torch.Tensor) -> PrefillResult:
    with torch.no_grad():
        output = model(input_ids=prompt_ids, use_cache=True)
    assert output.past_key_values is not None
    return PrefillResult(
        logits=output.logits[:, -1:, :],
        past_key_values=output.past_key_values,
        generation_input_ids=prompt_ids,
        position_ids=torch.tensor([[prompt_ids.size(1) - 1]]),
        attention_mask=torch.ones_like(prompt_ids),
        flops=0,
    )


def prefix_layout(length: int) -> PhysicalLayout:
    return PhysicalLayout(length, (
        PhysicalChunkScope("chunk", "digest", 0, length, 0, length),
    ))


class AnswerAttentionTests(unittest.TestCase):
    def test_literal_and_shifted_each_execute_one_real_layer_pass(self):
        model = build_models()[0]
        model.eval()
        model.set_attn_implementation("eager")
        for streamer in (
            stream_answer_attention,
            stream_shifted_prediction_attention,
        ):
            with self.subTest(streamer=streamer.__name__):
                prefill = prefill_result(model, torch.tensor([[1, 2, 3]]))
                seen = []
                with patch(
                    "sempic.attention_metrics.answer.recompute_kv",
                    wraps=recompute_kv,
                ) as recompute:
                    streamer(
                        model=model,
                        prefill=prefill,
                        answer_ids=torch.tensor([4, 5]),
                        basis_sink=lambda event: seen.append(event.layer_index),
                        layout=prefix_layout(3),
                    )
                self.assertEqual(recompute.call_count, model.config.num_hidden_layers)
                self.assertEqual(seen, list(range(model.config.num_hidden_layers)))

    def test_recompute_default_result_does_not_add_attention_probabilities(self):
        model = build_models()[1]
        model.eval()
        model.set_attn_implementation("eager")
        input_ids = torch.tensor([[1, 2]])
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
        assert output.past_key_values is not None
        cache = KVCache.from_hf_cache(
            output.past_key_values,
            position_ids=torch.tensor([[0, 1]]),
        )

        result = recompute_kv(
            model=model,
            kv_cache=cache,
            hidden_states=model.model.embed_tokens(input_ids),
            pos_ids=torch.tensor([[0, 1]]),
            token_idx=[0, 1],
            layer_idx=0,
        )

        self.assertNotIn("attention_probs", result)

    def test_gold_answer_tokenization_is_standalone_without_special_tokens_or_eos(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def __call__(self, text, **kwargs):
                self.calls.append((text, kwargs))
                return {"input_ids": [7, 8]}

        tokenizer = FakeTokenizer()
        answer_ids = tokenize_gold_answer(tokenizer, "gold")  # type: ignore[arg-type]

        self.assertEqual(answer_ids.tolist(), [7, 8])
        self.assertEqual(tokenizer.calls, [(
            "gold",
            {"add_special_tokens": False, "padding": False},
        )])

    def test_streamed_probabilities_match_eager_forward_and_keep_query_heads(self):
        prompt_ids = torch.tensor([[1, 2, 3]])
        answer_ids = torch.tensor([4, 5])

        for model in build_models():
            with self.subTest(model=type(model).__name__):
                torch.manual_seed(7)
                model.eval()
                model.set_attn_implementation("eager")
                expected_prefill = prefill_result(model, prompt_ids)
                actual_prefill = prefill_result(model, prompt_ids)

                prefix_len = prompt_ids.size(1)
                with torch.no_grad():
                    expected = model(
                        input_ids=answer_ids.unsqueeze(0),
                        past_key_values=expected_prefill.past_key_values,
                        attention_mask=torch.ones((1, prefix_len + answer_ids.numel()), dtype=torch.long),
                        position_ids=torch.tensor([[3, 4]]),
                        cache_position=torch.tensor([3, 4]),
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                assert expected.attentions is not None

                streamed = []
                meta = stream_answer_attention(
                    model=model,
                    prefill=actual_prefill,
                    answer_ids=answer_ids,
                    basis_sink=lambda event: streamed.append((
                        event.layer_index,
                        event.attention_probabilities.cpu().clone(),
                    )),
                    layout=prefix_layout(prefix_len),
                )

                self.assertEqual([layer for layer, _ in streamed], [0, 1])
                self.assertEqual(meta.prefix_physical_len, 3)
                self.assertEqual(meta.answer_len, 2)
                self.assertEqual(meta.num_layers, 2)
                self.assertEqual(meta.num_query_heads, 4)
                self.assertEqual(actual_prefill.past_key_values.get_seq_length(), 3)
                for layer_index, (_, probabilities) in enumerate(streamed):
                    self.assertEqual(tuple(probabilities.shape), (4, 2, 5))
                    torch.testing.assert_close(
                        probabilities,
                        expected.attentions[layer_index].squeeze(0),
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    torch.testing.assert_close(
                        probabilities[:, 0, -1],
                        torch.zeros(4),
                    )
                    torch.testing.assert_close(
                        probabilities.sum(dim=-1),
                        torch.ones((4, 2)),
                    )

    def test_answer_attention_rejects_non_eager_backend(self):
        model = build_models()[1]
        model.eval()
        model.set_attn_implementation("sdpa")
        prefill = prefill_result(model, torch.tensor([[1, 2]]))

        with self.assertRaisesRegex(ValueError, "requires eager"):
            stream_answer_attention(
                model=model,
                prefill=prefill,
                answer_ids=torch.tensor([3]),
                basis_sink=lambda _event: None,
                layout=prefix_layout(2),
            )

    def test_shifted_probabilities_match_causal_prediction_rows(self):
        prompt_ids = torch.tensor([[1, 2, 3]])
        answer_ids = torch.tensor([4, 5, 6])

        for model in build_models():
            with self.subTest(model=type(model).__name__):
                model.eval()
                model.set_attn_implementation("eager")
                actual_prefill = prefill_result(model, prompt_ids)
                original_cache = [
                    (layer.keys.clone(), layer.values.clone())
                    for layer in actual_prefill.past_key_values.layers
                ]
                expected_ids = torch.cat((prompt_ids, answer_ids[:-1].unsqueeze(0)), dim=1)
                with torch.no_grad():
                    expected = model(
                        input_ids=expected_ids,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                assert expected.attentions is not None

                streamed = []
                meta = stream_shifted_prediction_attention(
                    model=model,
                    prefill=actual_prefill,
                    answer_ids=answer_ids,
                    basis_sink=lambda event: streamed.append((
                        event.layer_index,
                        event.attention_probabilities.cpu().clone(),
                    )),
                    layout=prefix_layout(3),
                )

                self.assertEqual(meta.prefix_physical_len, 3)
                self.assertEqual(meta.answer_len, 3)
                for layer_index, (_, probabilities) in enumerate(streamed):
                    self.assertEqual(tuple(probabilities.shape), (4, 3, 5))
                    torch.testing.assert_close(
                        probabilities,
                        expected.attentions[layer_index][0, :, -3:, :],
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    torch.testing.assert_close(
                        probabilities.sum(dim=-1),
                        torch.ones((4, 3)),
                    )
                    for query_index in range(3):
                        torch.testing.assert_close(
                            probabilities[:, query_index, 3 + query_index:],
                            torch.zeros((4, 2 - query_index)),
                        )
                for layer_index, (original_keys, original_values) in enumerate(
                    original_cache
                ):
                    layer = actual_prefill.past_key_values.layers[layer_index]
                    torch.testing.assert_close(layer.keys, original_keys)
                    torch.testing.assert_close(layer.values, original_values)

    def test_shifted_single_token_answer_uses_only_prompt_last_query(self):
        prompt_ids = torch.tensor([[1, 2, 3]])
        answer_ids = torch.tensor([4])

        for model in build_models():
            with self.subTest(model=type(model).__name__):
                model.eval()
                model.set_attn_implementation("eager")
                prefill = prefill_result(model, prompt_ids)
                with torch.no_grad():
                    expected = model(
                        input_ids=prompt_ids,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                assert expected.attentions is not None
                streamed = []

                stream_shifted_prediction_attention(
                    model=model,
                    prefill=prefill,
                    answer_ids=answer_ids,
                    basis_sink=lambda event: streamed.append((
                        event.layer_index,
                        event.attention_probabilities.cpu().clone(),
                    )),
                    layout=prefix_layout(3),
                )

                for layer_index, (_, probabilities) in enumerate(streamed):
                    self.assertEqual(tuple(probabilities.shape), (4, 1, 3))
                    torch.testing.assert_close(
                        probabilities,
                        expected.attentions[layer_index][0, :, -1:, :],
                        rtol=1e-5,
                        atol=1e-7,
                    )

    def test_shifted_query_uses_prefill_semantic_position(self):
        model = build_models()[1]
        model.eval()
        model.set_attn_implementation("eager")
        prefill = prefill_result(model, torch.tensor([[1, 2, 3]]))
        prefill = prefill._replace(position_ids=torch.tensor([[9]]))

        with patch(
            "sempic.attention_metrics.answer.prepare_pos_embed_and_mask",
            wraps=prepare_pos_embed_and_mask,
        ) as prepare:
            stream_shifted_prediction_attention(
                model=model,
                prefill=prefill,
                answer_ids=torch.tensor([4, 5]),
                basis_sink=lambda _event: None,
                layout=prefix_layout(3),
            )

        position_ids = prepare.call_args.kwargs["pos_ids"]
        recompute_indices = prepare.call_args.kwargs["recompute_indices"]
        self.assertEqual(recompute_indices, [2, 3])
        self.assertEqual(position_ids.tolist(), [[0, 1, 9, 10]])


if __name__ == "__main__":
    unittest.main()
