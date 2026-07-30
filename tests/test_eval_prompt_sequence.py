import inspect
import types
import unittest
from types import MappingProxyType
from unittest.mock import patch

import torch

from sempic.cache import KVCache, KeyValue
from sempic.cache_comb.methods import (
    CACHE_COMB_FUNC_DICT,
    CONTEXT_BLOCK_METHODS,
    NO_PREP_METHODS,
    WHOLE_PREFIX_METHODS,
    get_cache_comb_func,
)
from sempic.cache_comb.methods._prompt_utils import (
    inline_only_ids,
    require_one_to_one_context_kvs,
)
from sempic.cache_comb.methods.default import full_recompute, no_cache_eval
from sempic.cache_comb.methods.sink import sink_eval
from sempic.prompt import TokenSpan, TokenizedPrompt
from run_eval import _prepared_source_parts, run_eval


def make_cache(length: int, positions: torch.Tensor | None = None) -> KVCache:
    cache = KVCache()
    if positions is None:
        positions = torch.arange(length).unsqueeze(0)
    cache.update(
        0,
        KeyValue(
            key=torch.zeros((1, 1, length, 2)),
            value=torch.zeros((1, 1, length, 2)),
            position_ids=positions,
        ),
    )
    return cache


class EvalPromptSequenceTests(unittest.TestCase):
    def test_registry_is_fully_partitioned(self):
        self.assertEqual(len(CACHE_COMB_FUNC_DICT), 13)
        self.assertNotIn("full_context", CACHE_COMB_FUNC_DICT)
        self.assertEqual(
            NO_PREP_METHODS | WHOLE_PREFIX_METHODS | CONTEXT_BLOCK_METHODS,
            frozenset(CACHE_COMB_FUNC_DICT),
        )
        self.assertFalse(NO_PREP_METHODS & WHOLE_PREFIX_METHODS)
        self.assertFalse(NO_PREP_METHODS & CONTEXT_BLOCK_METHODS)
        self.assertFalse(WHOLE_PREFIX_METHODS & CONTEXT_BLOCK_METHODS)

    def test_all_registry_methods_use_the_canonical_protocol(self):
        expected = [
            "model",
            "tokenizer",
            "generation_config",
            "prompt",
            "prepared_kvs",
            "answer",
            "answer_postprocess_func",
            "kwargs",
        ]
        for name, func in CACHE_COMB_FUNC_DICT.items():
            with self.subTest(name=name):
                self.assertEqual(list(inspect.signature(func).parameters), expected)

    def test_run_eval_accepts_only_registered_method_names(self):
        parameters = inspect.signature(run_eval).parameters
        self.assertIn("cache_comb_method", parameters)
        self.assertIn("lora_adapter_name", parameters)
        self.assertNotIn("cache_comb_func", parameters)
        self.assertNotIn("eval_config", parameters)
        self.assertNotIn("debug", parameters)

        with self.assertRaisesRegex(ValueError, "Unsupported cache combination method"):
            get_cache_comb_func("custom")

    def test_prepared_sources_are_keyed_by_part_index_and_reject_empty_prefix(self):
        prompt = TokenizedPrompt(
            torch.arange(4),
            (
                TokenSpan("context", 0, 1),
                TokenSpan("inline", 1, 2),
                TokenSpan("context", 2, 3),
                TokenSpan("inline", 3, 4),
            ),
        )
        self.assertEqual(
            _prepared_source_parts(prompt, "no_recompute"),
            [(0, 0, 1), (2, 2, 3)],
        )

        terminal_only = TokenizedPrompt(
            torch.tensor([1]),
            (TokenSpan("inline", 0, 1),),
        )
        with self.assertRaisesRegex(ValueError, "non-empty prompt prefix"):
            _prepared_source_parts(terminal_only, "single_cache")

    def test_no_cache_concatenates_existing_inline_slices(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 11, 20, 21, 12, 30, 13]),
            parts=(
                TokenSpan("inline", 0, 2),
                TokenSpan("context", 2, 4),
                TokenSpan("inline", 4, 5),
                TokenSpan("context", 5, 6),
                TokenSpan("inline", 6, 7),
            ),
        )
        self.assertTrue(torch.equal(inline_only_ids(prompt), torch.tensor([[10, 11, 12, 13]])))

    def test_no_cache_evaluates_all_inline_tokens_and_drops_context_blocks(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 11, 20, 21, 12, 30, 13]),
            parts=(
                TokenSpan("inline", 0, 2),
                TokenSpan("context", 2, 4),
                TokenSpan("inline", 4, 5),
                TokenSpan("context", 5, 6),
                TokenSpan("inline", 6, 7),
            ),
        )
        expected_result = object()

        with patch(
            "sempic.cache_comb.methods.default._ordinary_prompt_eval",
            return_value=expected_result,
        ) as ordinary_eval:
            result = no_cache_eval(
                model=types.SimpleNamespace(device=torch.device("cpu")),
                tokenizer=None,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={},
                answer="",
            )

        self.assertIs(result, expected_result)
        call = ordinary_eval.call_args.kwargs
        self.assertTrue(torch.equal(call["input_ids"], torch.tensor([[10, 11, 12, 13]])))

        with self.assertRaisesRegex(ValueError, "does not consume prepared KV"):
            no_cache_eval(
                model=types.SimpleNamespace(device=torch.device("cpu")),
                tokenizer=None,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={1: make_cache(2)},
                answer="",
            )

    def test_full_recompute_uses_complete_canonical_prompt_without_prepared_kv(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 20, 21, 11, 12]),
            parts=(
                TokenSpan("inline", 0, 1),
                TokenSpan("context", 1, 3),
                TokenSpan("inline", 3, 5),
            ),
        )
        model = types.SimpleNamespace(device=torch.device("cpu"))
        expected_result = object()
        with patch(
            "sempic.cache_comb.methods.default._ordinary_prompt_eval",
            return_value=expected_result,
        ) as ordinary_eval:
            result = full_recompute(
                model=model,
                tokenizer=None,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={},
                answer="",
            )

        self.assertIs(result, expected_result)
        call = ordinary_eval.call_args.kwargs
        self.assertTrue(torch.equal(call["input_ids"], prompt.input_ids.unsqueeze(0)))

        with self.assertRaisesRegex(ValueError, "does not consume prepared KV"):
            full_recompute(
                model=model,
                tokenizer=None,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={1: make_cache(2)},
                answer="",
            )

    def test_sink_without_context_blocks_uses_full_ordinary_prefill(self):
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([10, 11, 12]),
            parts=(TokenSpan("inline", 0, 3),),
        )
        expected_result = object()

        with patch(
            "sempic.cache_comb.methods.sink._ordinary_prompt_eval",
            return_value=expected_result,
        ) as ordinary_eval:
            result = sink_eval(
                model=types.SimpleNamespace(device=torch.device("cpu")),
                tokenizer=None,
                generation_config=None,
                prompt=prompt,
                prepared_kvs={},
                answer="",
            )

        self.assertIs(result, expected_result)
        call = ordinary_eval.call_args.kwargs
        self.assertTrue(torch.equal(call["input_ids"], prompt.input_ids.unsqueeze(0)))

    def test_context_mapping_requires_exact_part_indices_and_one_to_one_layout(self):
        prompt = TokenizedPrompt(
            torch.arange(5),
            (TokenSpan("context", 0, 3), TokenSpan("inline", 3, 5)),
        )
        valid = MappingProxyType({0: make_cache(3)})
        require_one_to_one_context_kvs("method", prompt, valid)
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            require_one_to_one_context_kvs("method", prompt, {})
        compressed = MappingProxyType({0: make_cache(2, torch.tensor([[0, 2]]))})
        with self.assertRaisesRegex(ValueError, "one physical KV position"):
            require_one_to_one_context_kvs("method", prompt, compressed)

if __name__ == "__main__":
    unittest.main()
