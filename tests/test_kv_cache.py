import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from sempic.cache import KVCache, KeyValue, get_kv_caches_with_grad
from sempic.cache_comb.recompute_kv import recompute_kv
from sempic.cache_comb.recompute_kv.utils import (
    adapt_recompute_mask,
    create_recompute_mask,
    update_recomputed_kv,
)


class KVCacheSelectionTests(unittest.TestCase):
    def test_select_seq_keeps_position_ids_aligned(self):
        cache = KVCache()
        cache.update(0, KeyValue(
            key=torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1),
            value=torch.arange(10, 14, dtype=torch.float32).view(1, 1, 4, 1),
            position_ids=torch.tensor([[3, 5, 8, 13]]),
        ))

        selected = cache.select_seq(torch.tensor([3, 1]))

        self.assertEqual(selected[0].key.flatten().tolist(), [3.0, 1.0])
        self.assertEqual(selected[0].value.flatten().tolist(), [13.0, 11.0])
        self.assertEqual(selected[0].position_ids.tolist(), [[13, 5]])

    def test_select_seq_rejects_missing_physical_position_mapping(self):
        cache = KVCache()
        cache.update(0, KeyValue(
            key=torch.zeros(1, 1, 2, 1),
            value=torch.zeros(1, 1, 2, 1),
            position_ids=torch.tensor([[0, 1, 2]]),
        ))

        with self.assertRaisesRegex(ValueError, "one-to-one position mapping"):
            cache.select_seq(torch.tensor([0]))


class RecomputeMaskTests(unittest.TestCase):
    def test_selective_fusion_overwrites_non_fused_positions(self):
        key = torch.tensor([0.0, 10.0, 20.0]).view(1, 1, 3, 1)
        value = torch.tensor([30.0, 40.0, 50.0]).view(1, 1, 3, 1)
        new_key = torch.tensor([100.0, 200.0, 300.0]).view(1, 1, 3, 1)
        new_value = torch.tensor([400.0, 500.0, 600.0]).view(1, 1, 3, 1)

        update_recomputed_kv(
            key,
            value,
            new_key,
            new_value,
            token_idx=[0, 1, 2],
            update_indices=None,
            fuse_indices=[1],
            fuse_theta=0.25,
        )

        self.assertEqual(key.flatten().tolist(), [100.0, 57.5, 300.0])
        self.assertEqual(value.flatten().tolist(), [400.0, 155.0, 600.0])

    def test_selective_fusion_requires_an_explicit_valid_subset(self):
        states = torch.zeros(1, 1, 2, 1)
        new_states = torch.ones_like(states)

        with self.assertRaisesRegex(ValueError, "provided together"):
            update_recomputed_kv(
                states.clone(), states.clone(), new_states, new_states,
                token_idx=[0, 1], update_indices=None,
                fuse_indices=None, fuse_theta=0.5,
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            update_recomputed_kv(
                states.clone(), states.clone(), new_states, new_states,
                token_idx=[0, 1], update_indices=[0],
                fuse_indices=[1], fuse_theta=0.5,
            )

    def test_backend_mask_adaptation(self):
        keep_mask = create_recompute_mask(
            query_len=2,
            key_len=4,
            token_idx=[0, 2],
            device=torch.device("cpu"),
            to_4d=True,
        )
        expected = torch.tensor([[[
            [True, False, False, False],
            [True, True, True, False],
        ]]])
        torch.testing.assert_close(keep_mask, expected)

        sdpa_mask = adapt_recompute_mask(
            keep_mask,
            attn_implementation="sdpa",
            dtype=torch.float32,
        )
        self.assertEqual(sdpa_mask.dtype, torch.bool)
        self.assertEqual(sdpa_mask.device, keep_mask.device)
        torch.testing.assert_close(sdpa_mask, keep_mask)

        eager_mask = adapt_recompute_mask(
            keep_mask,
            attn_implementation="eager",
            dtype=torch.float32,
        )
        self.assertEqual(eager_mask.shape, keep_mask.shape)
        self.assertEqual(eager_mask.device, keep_mask.device)
        self.assertEqual(eager_mask.dtype, torch.float32)
        torch.testing.assert_close(eager_mask[keep_mask], torch.zeros(4))
        torch.testing.assert_close(
            eager_mask[~keep_mask],
            torch.full((4,), torch.finfo(torch.float32).min),
        )

        with self.assertRaisesRegex(TypeError, "boolean keep-mask"):
            adapt_recompute_mask(
                keep_mask.float(),
                attn_implementation="eager",
                dtype=torch.float32,
            )
        with self.assertRaisesRegex(NotImplementedError, "only 'eager' and 'sdpa'"):
            adapt_recompute_mask(
                keep_mask,
                attn_implementation="flash_attention_2",
                dtype=torch.float32,
            )

    def test_tiny_model_recompute_matches_full_layer_output(self):
        model_factories = (
            ("llama", lambda: LlamaForCausalLM(LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                attention_dropout=0.0,
            ))),
            ("qwen3", lambda: Qwen3ForCausalLM(Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                max_position_embeddings=32,
                attention_dropout=0.0,
            ))),
        )
        input_ids = torch.tensor([[1, 5, 7, 3]])
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        token_idx = list(range(input_ids.size(1)))

        for model_name, model_factory in model_factories:
            for attn_implementation in ("eager", "sdpa"):
                with self.subTest(
                    model=model_name,
                    attn_implementation=attn_implementation,
                ):
                    torch.manual_seed(7)
                    model = model_factory()
                    model.config._attn_implementation = attn_implementation
                    model.eval()
                    layer_output: list[torch.Tensor] = []

                    def capture_layer_output(_module, _args, output):
                        layer_output.append(output.clone())

                    handle = model.model.layers[0].register_forward_hook(capture_layer_output)
                    try:
                        outputs = model(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            use_cache=True,
                            return_dict=True,
                        )
                    finally:
                        handle.remove()

                    self.assertIsNotNone(outputs.past_key_values)
                    kv_cache = KVCache.from_hf_cache(outputs.past_key_values, position_ids)
                    keep_mask = create_recompute_mask(
                        query_len=len(token_idx),
                        key_len=input_ids.size(1),
                        token_idx=token_idx,
                        device=torch.device("cpu"),
                        to_4d=True,
                    )
                    result = recompute_kv(
                        model=model,
                        kv_cache=kv_cache,
                        hidden_states=model.model.embed_tokens(input_ids),
                        pos_ids=position_ids,
                        token_idx=token_idx,
                        layer_idx=0,
                        recompute_mask=keep_mask,
                    )

                    self.assertEqual(keep_mask.dtype, torch.bool)
                    torch.testing.assert_close(
                        result["recomputed_hidden_states"],
                        layer_output[0],
                        rtol=1e-5,
                        atol=1e-7,
                    )


class KVGradientCheckpointTests(unittest.TestCase):
    def test_layerwise_checkpoint_matches_full_forward(self):
        models = (
            LlamaForCausalLM(LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )),
            Qwen3ForCausalLM(Qwen3Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )),
        )
        attention_mask = torch.tensor([
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ])

        for model in models:
            with self.subTest(model=type(model).__name__):
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False
                torch.manual_seed(0)
                source = torch.randn(2, 4, model.config.hidden_size)
                direct_input = source.clone().requires_grad_(True)
                checkpoint_input = source.clone().requires_grad_(True)

                direct = get_kv_caches_with_grad(
                    model,
                    input_embeds=direct_input,
                    attention_mask=attention_mask,
                )
                checkpointed = get_kv_caches_with_grad(
                    model,
                    input_embeds=checkpoint_input,
                    attention_mask=attention_mask,
                    checkpoint_grad=True,
                )

                direct_loss = direct_input.new_zeros(())
                checkpoint_loss = checkpoint_input.new_zeros(())
                for direct_cache, checkpoint_cache in zip(direct, checkpointed, strict=True):
                    self.assertEqual(direct_cache.layers, [0, 1, 2])
                    self.assertEqual(checkpoint_cache.layers, [0, 1, 2])
                    for layer_index in direct_cache.layers:
                        direct_kv = direct_cache[layer_index]
                        checkpoint_kv = checkpoint_cache[layer_index]
                        torch.testing.assert_close(checkpoint_kv.key, direct_kv.key)
                        torch.testing.assert_close(checkpoint_kv.value, direct_kv.value)
                        torch.testing.assert_close(checkpoint_kv.position_ids, direct_kv.position_ids)
                        direct_loss = direct_loss + direct_kv.key.sum() + direct_kv.value.sum()
                        checkpoint_loss = checkpoint_loss + checkpoint_kv.key.sum() + checkpoint_kv.value.sum()

                direct_loss.backward()
                checkpoint_loss.backward()
                torch.testing.assert_close(checkpoint_input.grad, direct_input.grad)
                self.assertEqual([cache[2].key.size(2) for cache in checkpointed], [4, 2])


if __name__ == "__main__":
    unittest.main()
