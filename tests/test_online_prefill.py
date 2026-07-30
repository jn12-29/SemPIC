import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

from sempic.cache import KVCache, concate_kv_caches
from sempic.cache.rotate import rerotate_kv, rerotate_kv_flops
from sempic.cache_comb import PrefillResult
from sempic.cache_comb.compact_prefill import CompactPrefillExecutor
from sempic.cache_comb.methods.default import full_recompute, no_cache_eval
from sempic.cache_comb.methods.no_recompute import no_recompute_eval
from sempic.cache_comb.runtime import TTFTTimer, generate_from_prefill
from sempic.cache_comb.utils.flops import AutoFlopsCalculator
from sempic.prompt import TokenSpan, TokenizedPrompt


class OnlinePrefillRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )).eval()
        self.input_ids = torch.tensor([[5, 6, 7, 8]])
        self.tokenizer = SimpleNamespace(pad_token_id=0)

    def _prefill(self) -> PrefillResult:
        attention_mask = torch.ones_like(self.input_ids)
        position_ids = torch.arange(self.input_ids.size(1)).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(
                input_ids=self.input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )
        return PrefillResult(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            generation_input_ids=self.input_ids,
            position_ids=position_ids[:, -1:],
            attention_mask=attention_mask,
            flops=0,
        )

    @staticmethod
    def _timer(device: torch.device) -> TTFTTimer:
        timer = TTFTTimer(device)
        timer.start()
        return timer

    def test_ttft_timer_synchronizes_at_formal_start_and_first_token(self) -> None:
        events = []
        clock = iter((10.0, 10.25))
        timer = TTFTTimer(torch.device("cuda:0"))

        with (
            patch(
                "sempic.cache_comb.runtime.synchronize",
                side_effect=lambda device: events.append(("sync", device)),
            ),
            patch(
                "sempic.cache_comb.runtime.perf_counter",
                side_effect=lambda: next(clock),
            ),
        ):
            timer.start()
            stopped = timer(
                torch.ones((1, 2), dtype=torch.long, device="cpu"),
                torch.empty(0),
            )

        self.assertEqual(
            events,
            [
                ("sync", torch.device("cuda:0")),
                ("sync", torch.device("cpu")),
            ],
        )
        self.assertEqual(stopped.tolist(), [False])
        self.assertEqual(timer.result(), 0.25)

    def _assert_multi_token_decode(self, result: PrefillResult) -> None:
        sequences, _ = generate_from_prefill(
            model=self.model,
            tokenizer=None,  # type: ignore[arg-type]
            generation_config=GenerationConfig(
                max_new_tokens=3,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            ),
            result=result,
            ttft_timer=self._timer(result.generation_input_ids.device),
        )
        self.assertEqual(
            sequences.size(1),
            result.generation_input_ids.size(1) + 3,
        )

    def _assert_compact_matches_serial_reference(self, device: torch.device) -> None:
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )).to(device).eval()
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([5, 6, 7, 8, 9]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 5),
            ),
        )
        with torch.no_grad():
            context_output = model(
                input_ids=torch.tensor([[6, 7]], device=device),
                position_ids=torch.tensor([[0, 1]], device=device),
                use_cache=True,
            )
            preamble_output = model(
                input_ids=torch.tensor([[5]], device=device),
                position_ids=torch.tensor([[0]], device=device),
                use_cache=True,
            )
        prepared = KVCache.from_hf_cache(
            context_output.past_key_values,
            position_ids=torch.tensor([[0, 1]], device=device),
        )
        shifted_context = rerotate_kv(
            prepared.copy(),
            model.model.rotary_emb,
            shift=1,
            nope_dim=None,
        )
        serial_prefix = concate_kv_caches([
            KVCache.from_hf_cache(
                preamble_output.past_key_values,
                position_ids=torch.tensor([[0]], device=device),
            ),
            shifted_context,
        ])
        with torch.no_grad():
            reference_output = model(
                input_ids=torch.tensor([[8, 9]], device=device),
                attention_mask=torch.ones((1, 5), dtype=torch.long, device=device),
                position_ids=torch.tensor([[3, 4]], device=device),
                cache_position=torch.tensor([3, 4], device=device),
                past_key_values=serial_prefix.to_hf_cache(config=model.config),
                use_cache=True,
                logits_to_keep=1,
            )

        executor = CompactPrefillExecutor(model, backend="flex")
        prefill_kwargs = dict(
            method_name="no_recompute",
            tokenizer=self.tokenizer,
            prompt=prompt,
            prepared_kvs={1: prepared},
        )
        executor.warm_request(**prefill_kwargs)
        compact = executor.prefill(**prefill_kwargs)
        torch.testing.assert_close(compact.logits, reference_output.logits, atol=1e-5, rtol=1e-5)
        for compact_layer, reference_layer in zip(
            compact.past_key_values.layers,
            reference_output.past_key_values.layers,
            strict=True,
        ):
            torch.testing.assert_close(compact_layer.keys, reference_layer.keys, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(compact_layer.values, reference_layer.values, atol=1e-5, rtol=1e-5)

        generation_config = GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
        compact_sequences, _ = generate_from_prefill(
            model=model,
            tokenizer=None,  # type: ignore[arg-type]
            generation_config=generation_config,
            result=compact,
            ttft_timer=self._timer(compact.generation_input_ids.device),
        )
        reference = PrefillResult(
            logits=reference_output.logits,
            past_key_values=reference_output.past_key_values,
            generation_input_ids=prompt.input_ids.to(device).unsqueeze(0),
            position_ids=torch.tensor([[4]], device=device),
            attention_mask=torch.ones((1, 5), dtype=torch.long, device=device),
            flops=0,
        )
        reference_sequences, _ = generate_from_prefill(
            model=model,
            tokenizer=None,  # type: ignore[arg-type]
            generation_config=generation_config,
            result=reference,
            ttft_timer=self._timer(reference.generation_input_ids.device),
        )
        self.assertTrue(torch.equal(compact_sequences[:, -3:], reference_sequences[:, -3:]))

    def test_compact_prefill_matches_serial_reference(self):
        self._assert_compact_matches_serial_reference(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FlexAttention parity.")
    def test_cuda_flex_compact_prefill_matches_serial_reference(self):
        self._assert_compact_matches_serial_reference(torch.device("cuda"))

    def test_warmup_uses_shape_metadata_without_building_request_layout(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([5, 6, 7, 8, 9]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 5),
            ),
        )
        with torch.no_grad():
            context_output = self.model(
                input_ids=prompt.input_ids[1:3].unsqueeze(0),
                use_cache=True,
            )
        prepared = KVCache.from_hf_cache(
            context_output.past_key_values,
            position_ids=torch.arange(2).unsqueeze(0),
        )
        prepared = concate_kv_caches([
            prepared.select_seq(torch.tensor([0])),
            prepared.select_seq(torch.tensor([1])),
        ])
        chunks_before = {
            layer: tuple(id(key_value) for key_value in prepared._cache[layer])
            for layer in prepared.layers
        }
        executor = CompactPrefillExecutor(self.model, backend="flex")
        warm_shape = Mock()
        executor.model = SimpleNamespace(device=torch.device("cuda"))
        executor._flex = SimpleNamespace(warm_shape=warm_shape)

        with patch.object(
            executor,
            "build_layout",
            side_effect=AssertionError("warmup must not build the request layout"),
        ):
            executor.warm_request(
                method_name="no_recompute",
                prompt=prompt,
                prepared_kvs={1: prepared},
                kwargs={},
            )

        warm_shape.assert_called_once_with(3, 5, executor.causal_lm)
        self.assertEqual(
            chunks_before,
            {
                layer: tuple(id(key_value) for key_value in prepared._cache[layer])
                for layer in prepared.layers
            },
        )

    def test_warmup_rejects_method_kwargs_before_backend_setup(self):
        executor = CompactPrefillExecutor(self.model, backend="flex")
        with self.assertRaisesRegex(ValueError, "unexpected kwargs"):
            executor.warm_request(
                method_name="no_recompute",
                kwargs={"unsupported": True},
            )

    def _assert_matches_native(
        self,
        generation_config: GenerationConfig,
        *,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        expected = self.model.generate(
            input_ids=self.input_ids,
            generation_config=generation_config,
        )
        expected_sequences = (
            expected if isinstance(expected, torch.Tensor) else expected.sequences
        )
        result = self._prefill()
        torch.manual_seed(seed)
        actual, ttft = generate_from_prefill(
            model=self.model,
            tokenizer=None,  # type: ignore[arg-type]
            generation_config=generation_config,
            result=result,
            ttft_timer=self._timer(result.generation_input_ids.device),
        )
        self.assertTrue(torch.equal(actual, expected_sequences))
        self.assertGreater(ttft, 0.0)

    def test_greedy_generation_matches_native_and_does_not_repeat_prefill(self):
        config = GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
        expected = self.model.generate(
            input_ids=self.input_ids,
            generation_config=config,
        )
        result = self._prefill()

        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            actual, _ = generate_from_prefill(
                model=self.model,
                tokenizer=None,  # type: ignore[arg-type]
                generation_config=config,
                result=result,
                ttft_timer=self._timer(result.generation_input_ids.device),
            )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(forward.call_count, config.max_new_tokens - 1)
        self.assertNotIn("_prefill", self.model.__dict__)
        self.assertTrue(all(
            call.kwargs["input_ids"].size(1) == 1
            for call in forward.call_args_list
        ))

    def test_sampled_multi_return_generation_matches_native(self):
        self._assert_matches_native(
            GenerationConfig(
                max_new_tokens=3,
                do_sample=True,
                num_return_sequences=2,
                eos_token_id=None,
                pad_token_id=0,
            ),
            seed=42,
        )

    def test_beam_generation_matches_native(self):
        self._assert_matches_native(GenerationConfig(
            max_new_tokens=3,
            do_sample=False,
            num_beams=3,
            num_return_sequences=2,
            return_dict_in_generate=True,
            eos_token_id=None,
            pad_token_id=0,
        ))

    def test_full_recompute_and_no_cache_count_their_actual_prompt(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([5, 6, 20, 21, 7]),
            parts=(
                TokenSpan("inline", 0, 2),
                TokenSpan("context", 2, 4),
                TokenSpan("inline", 4, 5),
            ),
        )
        calculator = AutoFlopsCalculator(self.model)

        full = full_recompute(
            self.model, self.tokenizer, None, prompt, {}, ""
        )
        no_cache = no_cache_eval(
            self.model, self.tokenizer, None, prompt, {}, ""
        )

        self.assertTrue(torch.equal(full.generation_input_ids, prompt.input_ids[None]))
        self.assertEqual(
            full.flops,
            calculator.forward_flops(batch_size=1, seq_len=5, logits_rows=1),
        )
        self.assertEqual(no_cache.generation_input_ids.tolist(), [[5, 6, 7]])
        self.assertEqual(
            no_cache.flops,
            calculator.forward_flops(batch_size=1, seq_len=3, logits_rows=1),
        )

    def test_compact_cache_counts_all_inline_rows_and_terminal_output_only(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([5, 6, 7, 8, 9]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("inline", 4, 5),
            ),
        )
        context_ids = prompt.input_ids[1:3].unsqueeze(0)
        with torch.no_grad():
            context_outputs = self.model(input_ids=context_ids, use_cache=True)
        prepared = KVCache.from_hf_cache(
            context_outputs.past_key_values,
            position_ids=torch.arange(2).unsqueeze(0),
        )
        calculator = AutoFlopsCalculator(self.model)

        result = no_recompute_eval(
            self.model,
            self.tokenizer,
            None,
            prompt,
            {1: prepared},
            "",
        )

        expected = rerotate_kv_flops(prepared, nope_dim=None)
        expected += calculator.total_flops(
            batch_size=1, seq_len=3, cache_len=2
        )
        expected += calculator.output_flops(
            batch_size=1, hidden_rows=1, logits_rows=1
        )
        self.assertEqual(result.flops, expected)
        self.assertEqual(result.past_key_values.get_seq_length(), 5)
        self.assertEqual(result.generation_input_ids.size(1), 5)

    def test_compact_prefill_rerotates_globally_reuses_one_mask_and_keeps_sources(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([5, 6, 7, 8, 9, 10]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("context", 4, 5),
                TokenSpan("inline", 5, 6),
            ),
        )
        prepared = {}
        for part_index, start, end in ((1, 1, 3), (3, 4, 5)):
            with torch.no_grad():
                outputs = self.model(input_ids=prompt.input_ids[start:end].unsqueeze(0), use_cache=True)
            prepared[part_index] = KVCache.from_hf_cache(
                outputs.past_key_values,
                position_ids=torch.arange(end - start).unsqueeze(0),
            )
        snapshots = {
            (part_index, layer_index): (
                cache[layer_index].key.clone(),
                cache[layer_index].value.clone(),
                cache[layer_index].position_ids.clone(),
            )
            for part_index, cache in prepared.items()
            for layer_index in cache.layers
        }

        with (
            patch(
                "sempic.cache_comb.compact_prefill.rerotate_embeddings",
                wraps=__import__(
                    "sempic.cache_comb.compact_prefill", fromlist=["rerotate_embeddings"]
                ).rerotate_embeddings,
            ) as rerotate,
            patch.object(self.model.model, "forward", wraps=self.model.model.forward) as body_forward,
        ):
            result = no_recompute_eval(
                self.model, self.tokenizer, None, prompt, prepared, ""
            )

        self.assertEqual(rerotate.call_count, 1)
        self.assertEqual(body_forward.call_count, 0)
        self.assertEqual(result.past_key_values.get_seq_length(), 6)
        self.assertEqual(result.generation_input_ids.size(1), 6)
        self.assertEqual(result.attention_mask.size(1), 6)
        self.assertEqual(result.generation_input_ids[0, -1], prompt.input_ids[-1])
        for (part_index, layer_index), expected in snapshots.items():
            actual = prepared[part_index][layer_index]
            self.assertTrue(torch.equal(actual.key, expected[0]))
            self.assertTrue(torch.equal(actual.value, expected[1]))
            self.assertTrue(torch.equal(actual.position_ids, expected[2]))

    def test_interleaved_cache_preserves_logical_extent_after_compression(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([6, 7, 8, 9, 10]),
            parts=(
                TokenSpan("context", 0, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("inline", 4, 5),
            ),
        )
        context_ids = prompt.input_ids[:3].unsqueeze(0)
        with torch.no_grad():
            context_outputs = self.model(input_ids=context_ids, use_cache=True)
        prepared = KVCache.from_hf_cache(
            context_outputs.past_key_values,
            position_ids=torch.arange(3).unsqueeze(0),
        ).select_seq(torch.tensor([0, 2]))

        result = no_recompute_eval(
            self.model,
            self.tokenizer,
            None,
            prompt,
            {0: prepared},
            "",
        )

        self.assertEqual(result.position_ids.tolist(), [[4]])
        self.assertEqual(result.past_key_values.get_seq_length(), 4)
        self.assertEqual(result.generation_input_ids.size(1), 4)
        self._assert_multi_token_decode(result)

    def test_interleaved_cache_preserves_wrapper_soft_token_extent(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([6, 7, 8, 9, 10]),
            parts=(
                TokenSpan("context", 0, 3),
                TokenSpan("inline", 3, 4),
                TokenSpan("inline", 4, 5),
            ),
        )
        wrapped_ids = torch.tensor([[11, 6, 7, 8, 12]])
        with torch.no_grad():
            wrapped_outputs = self.model(input_ids=wrapped_ids, use_cache=True)
        prepared = KVCache.from_hf_cache(
            wrapped_outputs.past_key_values,
            position_ids=torch.arange(5).unsqueeze(0),
        )

        result = no_recompute_eval(
            self.model,
            self.tokenizer,
            None,
            prompt,
            {0: prepared},
            "",
        )

        self.assertEqual(result.position_ids.tolist(), [[6]])
        self.assertEqual(result.past_key_values.get_seq_length(), 7)
        self.assertEqual(result.generation_input_ids.size(1), 7)
        self._assert_multi_token_decode(result)

    def test_prefill_injection_is_restored_when_generate_raises(self):
        result = self._prefill()

        with patch.object(
            self.model,
            "generate",
            side_effect=RuntimeError("generation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                generate_from_prefill(
                    model=self.model,
                    tokenizer=None,  # type: ignore[arg-type]
                    generation_config=GenerationConfig(max_new_tokens=1),
                    result=result,
                    ttft_timer=self._timer(result.generation_input_ids.device),
                )

        self.assertNotIn("_prefill", self.model.__dict__)


if __name__ == "__main__":
    unittest.main()
