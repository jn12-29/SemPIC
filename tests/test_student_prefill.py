import copy
import unittest
from unittest import mock

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM

from sempic.cache import KVCache, KeyValue
from sempic.cache.rotate import rerotate_embeddings
from sempic.packet_wrapper import PacketWrapper
from sempic.prompt import TokenSpan, TokenizedPrompt
from sempic.utils.generate import GenerationCache
from sempic.utils.lora import disable_lora_adapters, set_lora_trainable_only
from sempic.utils.student_prefill import (
    _dense_train_flex_attention,
    _interleave_layer_kv,
    _length_aware_context_embeds,
    build_logical_causal_mask,
    build_student_layout,
    batched_student_loss,
    collect_context_blocks,
    pack_rerotated_contexts,
    prepare_context_blocks,
)


def make_prompt(ids: list[int], spans: list[tuple[str, int, int]]) -> TokenizedPrompt:
    return TokenizedPrompt(
        input_ids=torch.tensor(ids, dtype=torch.long),
        parts=tuple(TokenSpan(kind=kind, start=start, end=end) for kind, start, end in spans),  # type: ignore[arg-type]
    )


def make_sample(prompt: TokenizedPrompt, index: int = 0) -> dict:
    return {"prompt": prompt, "semantic_key": f"sample-{index}"}


class FakeBody:
    def __init__(self, hidden_size: int = 3):
        self.embed_tokens = torch.nn.Embedding(32, hidden_size)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = FakeBody()


class StudentPrefillHelperTests(unittest.TestCase):
    def test_flex_score_mod_matches_independent_batched_padding_reference(self):
        torch.manual_seed(0)
        batch_size = 2
        query_heads = 4
        kv_heads = 2
        query_capacity = 4
        kv_capacity = 5
        head_dim = 3
        scale = 0.5
        query_lengths = torch.tensor([4, 2], dtype=torch.int32)
        kv_lengths = torch.tensor([5, 3], dtype=torch.int32)
        frontiers = torch.tensor(
            [[0, 2, 3, 4], [1, 2, 0, 0]],
            dtype=torch.long,
        )

        base_query = torch.randn(
            batch_size,
            query_heads,
            query_capacity,
            head_dim,
            dtype=torch.float64,
        )
        base_key = torch.randn(
            batch_size,
            kv_heads,
            kv_capacity,
            head_dim,
            dtype=torch.float64,
        )
        base_value = torch.randn_like(base_key)
        base_query[1, :, 2:] = 1_000.0
        base_key[1, :, 3:] = -2_000.0
        base_value[1, :, 3:] = 3_000.0

        query = base_query.clone().requires_grad_()
        key = base_key.clone().requires_grad_()
        value = base_value.clone().requires_grad_()
        reference_query = base_query.clone().requires_grad_()
        reference_key = base_key.clone().requires_grad_()
        reference_value = base_value.clone().requires_grad_()

        def fake_flex_attention(
            fake_query,
            fake_key,
            fake_value,
            *,
            score_mod,
            scale,
            enable_gqa,
            kernel_options,
        ):
            self.assertTrue(enable_gqa)
            self.assertEqual(kernel_options, {"FORCE_USE_FLEX_ATTENTION": True})
            repeats = fake_query.size(1) // fake_key.size(1)
            expanded_key = fake_key.repeat_interleave(repeats, dim=1)
            expanded_value = fake_value.repeat_interleave(repeats, dim=1)
            scores = torch.matmul(fake_query, expanded_key.transpose(-2, -1)) * scale
            batch_index = torch.arange(batch_size).view(batch_size, 1, 1, 1)
            head_index = torch.arange(query_heads).view(1, query_heads, 1, 1)
            query_index = torch.arange(query_capacity).view(1, 1, query_capacity, 1)
            key_index = torch.arange(kv_capacity).view(1, 1, 1, kv_capacity)
            modified_scores = score_mod(
                scores,
                batch_index,
                head_index,
                query_index,
                key_index,
            )
            safe_rows = torch.isfinite(modified_scores).any(dim=-1, keepdim=True)
            safe_scores = torch.where(
                safe_rows,
                modified_scores,
                torch.zeros_like(modified_scores),
            )
            probabilities = torch.softmax(safe_scores, dim=-1)
            probabilities = torch.where(
                safe_rows,
                probabilities,
                torch.zeros_like(probabilities),
            )
            return torch.matmul(probabilities, expanded_value)

        def independent_reference(reference_q, reference_k, reference_v):
            groups = query_heads // kv_heads
            batch_outputs = []
            for batch_index in range(batch_size):
                head_outputs = []
                for query_head in range(query_heads):
                    kv_head = query_head // groups
                    rows = []
                    for query_index in range(query_capacity):
                        if query_index >= int(query_lengths[batch_index]):
                            rows.append(
                                reference_q[batch_index, query_head, query_index] * 0.0
                            )
                            continue
                        key_count = min(
                            int(kv_lengths[batch_index]),
                            int(frontiers[batch_index, query_index]) + 1,
                        )
                        row_scores = (
                            reference_q[batch_index, query_head, query_index]
                            * reference_k[batch_index, kv_head, :key_count]
                        ).sum(dim=-1) * scale
                        probabilities = torch.softmax(row_scores, dim=-1)
                        rows.append(
                            torch.matmul(
                                probabilities,
                                reference_v[batch_index, kv_head, :key_count],
                            )
                        )
                    head_outputs.append(torch.stack(rows))
                batch_outputs.append(torch.stack(head_outputs))
            return torch.stack(batch_outputs)

        with mock.patch(
            "sempic.utils.student_prefill.flex_attention",
            side_effect=fake_flex_attention,
        ):
            actual = _dense_train_flex_attention(
                query,
                key,
                value,
                frontiers,
                query_lengths,
                kv_lengths,
                scale,
            )
        expected = independent_reference(
            reference_query,
            reference_key,
            reference_value,
        )
        upstream = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(actual, (query, key, value), upstream)
        expected_gradients = torch.autograd.grad(
            expected,
            (reference_query, reference_key, reference_value),
            upstream,
        )

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_gradient, expected_gradient in zip(
            actual_gradients,
            expected_gradients,
            strict=True,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                atol=1e-12,
                rtol=1e-12,
            )
        torch.testing.assert_close(actual[1, :, 2:], torch.zeros_like(actual[1, :, 2:]))
        torch.testing.assert_close(
            actual_gradients[0][1, :, 2:],
            torch.zeros_like(actual_gradients[0][1, :, 2:]),
        )
        for gradient in actual_gradients[1:]:
            torch.testing.assert_close(
                gradient[1, :, 3:],
                torch.zeros_like(gradient[1, :, 3:]),
            )

    def test_collects_context_blocks_with_canonical_views_and_mapping(self):
        prompts = [
            make_prompt(
                [1, 2, 3, 4],
                [("inline", 0, 1), ("context", 1, 3), ("inline", 3, 4)],
            ),
            make_prompt(
                [5, 6],
                [("context", 0, 1), ("inline", 1, 2)],
            ),
        ]

        items = collect_context_blocks(prompts)

        self.assertEqual([(item.sample_index, item.part_index) for item in items], [(0, 1), (1, 0)])
        self.assertEqual([item.input_ids.tolist() for item in items], [[2, 3], [5]])

        caches = [
            KVCache.create_dummy(1, 1, 1, 2, 1, 1),
            KVCache.create_dummy(1, 1, 1, 1, 1, 1),
        ]
        with mock.patch(
            "sempic.utils.student_prefill.get_kv_caches_with_grad",
            return_value=caches,
        ) as get_kv:
            mapped, lengths = prepare_context_blocks(
                items,
                FakeModel(),
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=None,
            )

        get_kv.assert_called_once()
        self.assertIs(mapped[(0, 1)], caches[0])
        self.assertIs(mapped[(1, 0)], caches[1])
        self.assertEqual(lengths, {(0, 1): 2, (1, 0): 1})
        self.assertEqual(get_kv.call_args.kwargs["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_length_aware_wrapper_places_trailer_before_padding(self):
        source = torch.tensor([
            [[1.0], [2.0], [0.0]],
            [[3.0], [0.0], [0.0]],
        ])
        wrapper = PacketWrapper(1, 1, 1, device=torch.device("cpu"))
        wrapper.header.data.fill_(8.0)
        wrapper.trailer.data.fill_(9.0)

        embeds, mask, lengths = _length_aware_context_embeds(source, [2, 1], wrapper)

        self.assertEqual(lengths, [4, 3])
        self.assertEqual(embeds[:, :, 0].tolist(), [[8.0, 1.0, 2.0, 9.0], [8.0, 3.0, 9.0, 0.0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])

    def test_layout_preserves_real_zero_token_and_teacher_shift(self):
        prompts = [
            make_prompt(
                [0, 4, 5, 6],
                [("inline", 0, 1), ("context", 1, 3), ("inline", 3, 4)],
            ),
            make_prompt([7, 0], [("inline", 0, 2)]),
        ]
        teachers = [torch.tensor([9, 10]), torch.tensor([11])]

        layout = build_student_layout(
            prompts,
            teachers,
            {(0, 1): 2},
            torch.device("cpu"),
        )

        self.assertEqual(layout.input_ids.tolist(), [[0, 6, 9], [7, 0, 0]])
        self.assertEqual(layout.query_valid.tolist(), [[True, True, True], [True, True, False]])
        self.assertEqual(layout.query_position_ids.tolist(), [[0, 3, 4], [0, 1, 0]])
        self.assertEqual(layout.context_position_ids.tolist(), [[1, 2], [0, 0]])
        self.assertEqual(layout.physical_valid.tolist(), [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ])
        self.assertEqual(layout.physical_source_indices.tolist(), [
            [2, 0, 1, 3, 4],
            [2, 3, 5, 5, 5],
        ])
        self.assertEqual(layout.query_frontiers.tolist(), [[0, 3, 4], [0, 1, 0]])
        self.assertEqual([rows.tolist() for rows in layout.target_rows], [[1, 2], [1]])
        self.assertEqual(
            [[placement.part_index for placement in row] for row in layout.context_placements],
            [[1], []],
        )
        first = layout.interleaved_layouts[0]
        self.assertEqual(first.inline_canonical_indices.tolist(), [0, 3, -1])
        self.assertEqual(first.inline_physical_indices.tolist(), [0, 3, 4])
        self.assertEqual(first.physical_source_indices.tolist(), [2, 0, 1, 3, 4])
        self.assertEqual(first.terminal_inline_row, 1)
        self.assertEqual(first.context_placements[0].physical_start, 1)
        self.assertEqual(first.context_placements[0].physical_end, 3)

    def test_global_context_rerotation_is_once_interleaved_and_non_mutating(self):
        prompt = make_prompt(
            [1, 2, 3, 4],
            [("inline", 0, 1), ("context", 1, 3), ("inline", 3, 4)],
        )
        layout = build_student_layout(
            [prompt],
            [torch.tensor([9, 10])],
            {(0, 1): 2},
            torch.device("cpu"),
        )
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        ))
        cache = KVCache()
        original = []
        for layer_index in range(2):
            layer_original = []
            for position in range(2):
                key = torch.randn(1, 2, 1, 4, requires_grad=True)
                value = torch.randn(1, 2, 1, 4, requires_grad=True)
                positions = torch.tensor([[position]])
                cache.update(layer_index, KeyValue(key, value, positions))
                layer_original.append((key.clone(), value.clone(), positions.clone()))
            original.append(layer_original)

        with mock.patch(
            "sempic.utils.student_prefill.rerotate_embeddings",
            wraps=rerotate_embeddings,
        ) as rerotate:
            packed = pack_rerotated_contexts(
                layout,
                {(0, 1): cache},
                2,
                model.model,
                None,
            )

        self.assertIsNotNone(packed)
        assert packed is not None
        rerotate.assert_called_once()
        self.assertEqual(rerotate.call_args.args[0].shape, (2, 2, 2, 4))
        self.assertEqual(packed.key.shape, (2, 1, 2, 2, 4))
        self.assertEqual(packed.value.shape, (2, 1, 2, 2, 4))
        for layer_index, layer_original in enumerate(original):
            self.assertEqual(len(cache._cache[layer_index]), 2)
            for item, (key, value, positions) in zip(
                cache._cache[layer_index], layer_original, strict=True
            ):
                self.assertTrue(torch.equal(item.key, key))
                self.assertTrue(torch.equal(item.value, value))
                self.assertTrue(torch.equal(item.position_ids, positions))
        (packed.key.sum() + packed.value.sum()).backward()
        for layer_index in range(2):
            for item in cache._cache[layer_index]:
                self.assertIsNotNone(item.key.grad)
                self.assertIsNotNone(item.value.grad)

        inline_key = torch.tensor([[[[20.0], [21.0], [22.0]]]])
        inline_value = inline_key + 100
        context_key = torch.tensor([[[[10.0], [11.0]]]])
        context_value = context_key + 100
        key, value = _interleave_layer_kv(
            layout,
            context_key,
            context_value,
            inline_key,
            inline_value,
        )
        self.assertEqual(key.flatten().tolist(), [20.0, 10.0, 11.0, 21.0, 22.0])
        self.assertEqual(value.flatten().tolist(), [120.0, 110.0, 111.0, 121.0, 122.0])

    def test_logical_mask_hides_future_context_and_padding(self):
        query_positions = torch.tensor([[0, 3, 4], [0, 1, 0]])
        context_positions = torch.tensor([[1, 2], [0, 0]])
        query_valid = torch.tensor([[True, True, True], [True, True, False]])
        context_valid = torch.tensor([[True, True], [False, False]])

        mask = build_logical_causal_mask(
            query_positions,
            context_positions,
            query_valid,
            context_valid,
            torch.float32,
        )[:, 0]

        minimum = torch.finfo(torch.float32).min
        self.assertTrue(torch.equal(mask[0, 0], torch.tensor([minimum, minimum, 0.0, minimum, minimum])))
        self.assertTrue(torch.equal(mask[0, 1], torch.tensor([0.0, 0.0, 0.0, 0.0, minimum])))
        self.assertTrue(torch.equal(mask[1, 1], torch.tensor([minimum, minimum, 0.0, 0.0, minimum])))
        self.assertTrue((mask[1, 2] == minimum).all())


class StudentPrefillModelTests(unittest.TestCase):
    def _build_cache(self, prompts, vocab_size, store_logits=False):
        cache = GenerationCache()
        teacher_sequences = ([12, 13], [14])
        for sample_index, prompt in enumerate(prompts):
            sequence = teacher_sequences[sample_index]
            logits = (
                [torch.randn(len(sequence), vocab_size)]
                if store_logits
                else []
            )
            cache.add(f"sample-{sample_index}", {
                "sequences": [torch.tensor(sequence, dtype=torch.long)],
                "logits": logits,
                "text": [""],
            })
        return cache

    def _build_target_cache(self, prompt, sequences):
        del prompt
        cache = GenerationCache()
        cache.add("sample-0", {
            "sequences": [torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
            "logits": [],
            "text": [""] * len(sequences),
        })
        return cache

    def test_teacher_cache_tensor_validation_is_operational_only(self):
        prompt = make_prompt([1, 2], [("context", 0, 1), ("inline", 1, 2)])
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        ))
        cases = (
            (torch.tensor([1.0]), [], {"type": "ce", "tau": 1.0}, "torch.long"),
            (torch.tensor([32]), [], {"type": "ce", "tau": 1.0}, "student vocabulary"),
            (
                torch.tensor([1]),
                [torch.zeros(1, 32, dtype=torch.long)],
                {"type": "kl", "tau": 1.0},
                "floating-point",
            ),
            (
                torch.tensor([1]),
                [torch.zeros(2, 32)],
                {"type": "kl", "tau": 1.0},
                "sequence length and student vocabulary",
            ),
        )
        for sequence, logits, loss_config, message in cases:
            with self.subTest(message=message):
                cache = GenerationCache()
                cache.add("sample-0", {
                    "sequences": [sequence],
                    "logits": logits,
                    "text": [""],
                })
                with self.assertRaisesRegex(ValueError, message):
                    batched_student_loss(
                        [make_sample(prompt)],
                        model,
                        cache,
                        loss_config,
                        lora_enabled=False,
                        lora_adapter_name=None,
                        packet_wrapper=None,
                    )

    def test_tiny_llama_and_qwen_batches_preserve_packet_gradients(self):
        prompts = [
            make_prompt(
                [1, 2, 3, 4],
                [("inline", 0, 1), ("context", 1, 3), ("inline", 3, 4)],
            ),
            make_prompt(
                [5, 0, 6],
                [("context", 0, 1), ("inline", 1, 3)],
            ),
        ]
        samples = [make_sample(prompt, index) for index, prompt in enumerate(prompts)]

        models = [
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
        ]

        for model in models:
            with self.subTest(model=type(model).__name__):
                for parameter in model.parameters():
                    parameter.requires_grad = False
                model.eval()
                wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
                cache = self._build_cache(prompts, model.config.vocab_size)

                loss, token_count, sample_losses = batched_student_loss(
                    samples,
                    model,
                    cache,
                    {"type": "ce", "tau": 1.0},
                    lora_enabled=False,
                    lora_adapter_name=None,
                    packet_wrapper=wrapper,
                )
                loss.backward()

                self.assertEqual(token_count, 3)
                self.assertEqual(len(sample_losses or []), 2)
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(wrapper.header.grad)
                self.assertIsNotNone(wrapper.trailer.grad)
                self.assertGreater(float(wrapper.header.grad.norm()), 0.0)
                self.assertGreater(float(wrapper.trailer.grad.norm()), 0.0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for SDPA and Flex parity."
    )
    def test_cuda_sdpa_and_batched_flex_loss_and_packet_gradients_match_cpu_eager(self):
        torch.manual_seed(0)
        prompts = [
            make_prompt(
                [1, 2, 3, 4],
                [("inline", 0, 1), ("context", 1, 3), ("inline", 3, 4)],
            ),
            make_prompt(
                [5, 6, 7],
                [("context", 0, 1), ("inline", 1, 3)],
            ),
        ]
        samples = [make_sample(prompt, index) for index, prompt in enumerate(prompts)]
        cpu_model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )).eval()
        for parameter in cpu_model.parameters():
            parameter.requires_grad = False
        cuda_model = copy.deepcopy(cpu_model).to("cuda").eval()

        for loss_type in ("ce", "kl"):
            with self.subTest(loss_type=loss_type):
                cpu_wrapper = PacketWrapper(1, 1, 64, device=torch.device("cpu"))
                cuda_wrapper = PacketWrapper(1, 1, 64, device=torch.device("cuda"))
                cuda_wrapper.load_state_dict(cpu_wrapper.state_dict())
                cache = self._build_cache(
                    prompts,
                    cpu_model.config.vocab_size,
                    store_logits=loss_type == "kl",
                )
                config = {"type": loss_type, "tau": 2.0}

                cpu_loss, cpu_tokens, _ = batched_student_loss(
                    samples,
                    cpu_model,
                    cache,
                    config,
                    lora_enabled=False,
                    lora_adapter_name=None,
                    packet_wrapper=cpu_wrapper,
                )
                cuda_loss, cuda_tokens, _ = batched_student_loss(
                    samples,
                    cuda_model,
                    cache,
                    config,
                    lora_enabled=False,
                    lora_adapter_name=None,
                    packet_wrapper=cuda_wrapper,
                )
                cpu_loss.backward()
                cuda_loss.backward()

                self.assertEqual(cuda_tokens, cpu_tokens)
                torch.testing.assert_close(
                    cuda_loss.cpu(), cpu_loss, atol=2e-5, rtol=2e-5
                )
                torch.testing.assert_close(
                    cuda_wrapper.header.grad.cpu(),
                    cpu_wrapper.header.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    cuda_wrapper.trailer.grad.cpu(),
                    cpu_wrapper.trailer.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )

                flex_cpu_wrapper = PacketWrapper(1, 1, 64, device=torch.device("cpu"))
                flex_cuda_wrapper = PacketWrapper(1, 1, 64, device=torch.device("cuda"))
                flex_cuda_wrapper.load_state_dict(flex_cpu_wrapper.state_dict())
                flex_cpu_loss, flex_cpu_tokens, _ = batched_student_loss(
                    samples,
                    cpu_model,
                    cache,
                    config,
                    lora_enabled=False,
                    lora_adapter_name=None,
                    packet_wrapper=flex_cpu_wrapper,
                    attention_backend="flex",
                )
                with mock.patch(
                    "sempic.utils.student_prefill.build_physical_causal_mask",
                    side_effect=AssertionError(
                        "CUDA FlexAttention must not build a dense physical mask."
                    ),
                ):
                    flex_cuda_loss, flex_cuda_tokens, _ = batched_student_loss(
                        samples,
                        cuda_model,
                        cache,
                        config,
                        lora_enabled=False,
                        lora_adapter_name=None,
                        packet_wrapper=flex_cuda_wrapper,
                        attention_backend="flex",
                    )
                flex_cpu_loss.backward()
                flex_cuda_loss.backward()

                self.assertEqual(flex_cuda_tokens, flex_cpu_tokens)
                torch.testing.assert_close(
                    flex_cuda_loss.cpu(), flex_cpu_loss, atol=2e-5, rtol=2e-5
                )
                torch.testing.assert_close(
                    flex_cuda_wrapper.header.grad.cpu(),
                    flex_cpu_wrapper.header.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )
                torch.testing.assert_close(
                    flex_cuda_wrapper.trailer.grad.cpu(),
                    flex_cpu_wrapper.trailer.grad,
                    atol=2e-5,
                    rtol=2e-5,
                )

    def test_tiny_llama_kl_is_summed_across_samples(self):
        torch.manual_seed(0)
        prompts = [
            make_prompt([1, 2], [("context", 0, 1), ("inline", 1, 2)]),
            make_prompt([3, 4], [("context", 0, 1), ("inline", 1, 2)]),
        ]
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        ))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        cache = self._build_cache(prompts, model.config.vocab_size, store_logits=True)

        loss, token_count, sample_losses = batched_student_loss(
            [make_sample(prompt, index) for index, prompt in enumerate(prompts)],
            model,
            cache,
            {"type": "kl", "tau": 2.0},
            lora_enabled=False,
            lora_adapter_name=None,
            packet_wrapper=None,
        )

        self.assertEqual(token_count, 3)
        assert sample_losses is not None
        self.assertAlmostEqual(float(loss), sum(sample_losses), places=5)

    def test_multiple_teacher_targets_sum_loss_and_gradients(self):
        torch.manual_seed(0)
        prompt = make_prompt(
            [1, 2, 3],
            [("inline", 0, 1), ("context", 1, 2), ("inline", 2, 3)],
        )
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        ))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        sequences = ([12, 13], [14])

        multi_wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
        split_wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
        split_wrapper.load_state_dict(multi_wrapper.state_dict())

        with mock.patch(
            "sempic.utils.student_prefill.prepare_context_blocks",
            wraps=prepare_context_blocks,
        ) as prepare:
            multi_loss, token_count, sample_losses = batched_student_loss(
                [make_sample(prompt)],
                model,
                self._build_target_cache(prompt, sequences),
                {"type": "ce", "tau": 1.0},
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=multi_wrapper,
            )
        multi_loss.backward()

        split_losses = []
        for sequence in sequences:
            loss, _, _ = batched_student_loss(
                [make_sample(prompt)],
                model,
                self._build_target_cache(prompt, [sequence]),
                {"type": "ce", "tau": 1.0},
                lora_enabled=False,
                lora_adapter_name=None,
                packet_wrapper=split_wrapper,
            )
            split_losses.append(loss)
        split_loss = torch.stack(split_losses).sum()
        split_loss.backward()

        prepare.assert_called_once()
        self.assertEqual(len(prepare.call_args.args[0]), 1)
        self.assertEqual(token_count, 3)
        self.assertEqual(len(sample_losses or []), 1)
        self.assertAlmostEqual(
            float(multi_loss.detach()),
            float(split_loss.detach()),
            places=5,
        )
        self.assertTrue(torch.allclose(multi_wrapper.header.grad, split_wrapper.header.grad, atol=1e-6))
        self.assertTrue(torch.allclose(multi_wrapper.trailer.grad, split_wrapper.trailer.grad, atol=1e-6))

    def test_lora_context_gradients_survive_checkpointing(self):
        prompt = make_prompt([1, 2], [("context", 0, 1), ("inline", 1, 2)])
        base_model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
        ))
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "k_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
            adapter_name="lora_kv_cache",
        )
        disable_lora_adapters(model)
        set_lora_trainable_only(model)
        cache = self._build_cache([prompt], model.config.vocab_size)

        loss, _, _ = batched_student_loss(
            [make_sample(prompt)],
            model,
            cache,
            {"type": "ce", "tau": 1.0},
            lora_enabled=True,
            lora_adapter_name="lora_kv_cache",
            packet_wrapper=None,
            kv_gradient_checkpointing=True,
        )
        loss.backward()

        lora_grad = sum(
            float(parameter.grad.norm())
            for name, parameter in model.named_parameters()
            if "lora_" in name and parameter.grad is not None
        )
        self.assertGreater(lora_grad, 0.0)

    def test_packet_context_gradients_survive_checkpointing(self):
        prompt = make_prompt([1, 2], [("context", 0, 1), ("inline", 1, 2)])
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
        ))
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
        wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
        cache = self._build_cache([prompt], model.config.vocab_size)

        loss, _, _ = batched_student_loss(
            [make_sample(prompt)],
            model,
            cache,
            {"type": "ce", "tau": 1.0},
            lora_enabled=False,
            lora_adapter_name=None,
            packet_wrapper=wrapper,
            kv_gradient_checkpointing=True,
        )
        loss.backward()

        self.assertIsNotNone(wrapper.header.grad)
        self.assertIsNotNone(wrapper.trailer.grad)
        self.assertGreater(float(wrapper.header.grad.norm()), 0.0)
        self.assertGreater(float(wrapper.trailer.grad.norm()), 0.0)

    def test_joint_checkpoint_matches_direct_gradients(self):
        prompt = make_prompt([1, 2, 3], [("context", 0, 2), ("inline", 2, 3)])
        direct_model = get_peft_model(
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
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
            adapter_name="lora_kv_cache",
        )
        checkpoint_model = copy.deepcopy(direct_model)
        direct_wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
        checkpoint_wrapper = PacketWrapper(1, 1, 16, device=torch.device("cpu"))
        checkpoint_wrapper.load_state_dict(direct_wrapper.state_dict())
        cache = self._build_cache([prompt], direct_model.config.vocab_size)

        for model in (direct_model, checkpoint_model):
            disable_lora_adapters(model)
            set_lora_trainable_only(model)

        direct_loss, _, _ = batched_student_loss(
            [make_sample(prompt)],
            direct_model,
            cache,
            {"type": "ce", "tau": 1.0},
            lora_enabled=True,
            lora_adapter_name="lora_kv_cache",
            packet_wrapper=direct_wrapper,
        )
        checkpoint_loss, _, _ = batched_student_loss(
            [make_sample(prompt)],
            checkpoint_model,
            cache,
            {"type": "ce", "tau": 1.0},
            lora_enabled=True,
            lora_adapter_name="lora_kv_cache",
            packet_wrapper=checkpoint_wrapper,
            kv_gradient_checkpointing=True,
        )
        torch.testing.assert_close(checkpoint_loss, direct_loss)

        direct_loss.backward()
        checkpoint_loss.backward()
        direct_grads = {
            name: parameter.grad
            for name, parameter in direct_model.named_parameters()
            if "lora_" in name
        }
        checkpoint_grads = {
            name: parameter.grad
            for name, parameter in checkpoint_model.named_parameters()
            if "lora_" in name
        }
        self.assertEqual(direct_grads.keys(), checkpoint_grads.keys())
        lora_grad = 0.0
        for name in direct_grads:
            direct_grad = direct_grads[name]
            checkpoint_grad = checkpoint_grads[name]
            self.assertEqual(direct_grad is None, checkpoint_grad is None)
            if direct_grad is not None and checkpoint_grad is not None:
                torch.testing.assert_close(checkpoint_grad, direct_grad)
                lora_grad += float(checkpoint_grad.norm())

        self.assertGreater(lora_grad, 0.0)
        torch.testing.assert_close(checkpoint_wrapper.header.grad, direct_wrapper.header.grad)
        torch.testing.assert_close(checkpoint_wrapper.trailer.grad, direct_wrapper.trailer.grad)
        self.assertGreater(float(checkpoint_wrapper.header.grad.norm()), 0.0)
        self.assertGreater(float(checkpoint_wrapper.trailer.grad.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
