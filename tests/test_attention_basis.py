import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from sempic.cache import KVCache
from sempic.cache_comb.recompute_kv import recompute_kv


def build_models():
    return (
        LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )),
        Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )),
    )


def make_cache(model, input_ids):
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=True)
    return KVCache.from_hf_cache(
        output.past_key_values,
        position_ids=torch.arange(input_ids.size(1)).unsqueeze(0),
    )


class AttentionBasisTests(unittest.TestCase):
    def test_basis_reuses_exact_eager_probabilities_and_physical_values(self):
        input_ids = torch.tensor([[1, 2, 3]])
        positions = torch.tensor([[0, 1, 2]])
        for model in build_models():
            with self.subTest(model=type(model).__name__):
                model.eval()
                model.set_attn_implementation("eager")
                expected_cache = make_cache(model, input_ids)
                actual_cache = make_cache(model, input_ids)
                hidden = model.model.embed_tokens(input_ids)
                expected = recompute_kv(
                    model=model,
                    kv_cache=expected_cache,
                    hidden_states=hidden,
                    pos_ids=positions,
                    token_idx=[0, 1, 2],
                    layer_idx=0,
                    update_cache=True,
                    return_attention_probs=True,
                )
                seen = []
                actual = recompute_kv(
                    model=model,
                    kv_cache=actual_cache,
                    hidden_states=hidden,
                    pos_ids=positions,
                    token_idx=[0, 1, 2],
                    layer_idx=0,
                    update_cache=True,
                    return_attention_probs=True,
                    attention_basis_sink=seen.append,
                )
                self.assertEqual(len(seen), 1)
                basis = seen[0]
                self.assertEqual(tuple(basis.scaled_masked_logits.shape), (1, 4, 3, 3))
                self.assertEqual(tuple(basis.physical_values.shape), (1, 2, 3, 4))
                self.assertEqual(basis.query_to_kv_head.tolist(), [0, 0, 1, 1])
                self.assertEqual(basis.keep_mask.dtype, torch.bool)
                torch.testing.assert_close(
                    basis.attention_probabilities,
                    expected["attention_probs"],
                )
                torch.testing.assert_close(
                    actual["attention_probs"],
                    expected["attention_probs"],
                )
                torch.testing.assert_close(
                    actual["recomputed_hidden_states"],
                    expected["recomputed_hidden_states"],
                )
                torch.testing.assert_close(
                    basis.physical_values,
                    actual_cache[0].value,
                )
                torch.testing.assert_close(
                    basis.attention_probabilities[:, :, 0, 1:],
                    torch.zeros((1, 4, 2)),
                )

    def test_basis_requires_eager_attention(self):
        model = build_models()[1]
        model.eval()
        model.set_attn_implementation("sdpa")
        input_ids = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "requires eager"):
            recompute_kv(
                model=model,
                kv_cache=make_cache(model, input_ids),
                hidden_states=model.model.embed_tokens(input_ids),
                pos_ids=torch.tensor([[0, 1]]),
                token_idx=[0, 1],
                layer_idx=0,
                attention_basis_sink=lambda _basis: None,
            )


if __name__ == "__main__":
    unittest.main()
