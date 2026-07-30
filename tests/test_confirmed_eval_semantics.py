import unittest
import types
from types import MappingProxyType
from unittest.mock import patch

import torch

from sempic.cache import KVCache, KeyValue
from sempic.cache_comb.methods.a3 import a3_eval
from sempic.cache_comb.methods.cache_blend import cache_blend_eval
from sempic.cache_comb.methods.epic import _epic_context_indices
from sempic.cache_comb.methods._prompt_utils import (
    prepare_recompute_inputs,
    recompute_prefix_indices,
)
from sempic.prompt import TokenSpan, TokenizedPrompt


class AcceptedRatio(Exception):
    pass


def make_cache(length: int) -> KVCache:
    cache = KVCache()
    for layer in range(2):
        cache.update(
            layer,
            KeyValue(
                key=torch.zeros((1, 1, length, 2)),
                value=torch.zeros((1, 1, length, 2)),
                position_ids=torch.arange(length).unsqueeze(0),
            ),
        )
    return cache


class ConfirmedEvalSemanticsTests(unittest.TestCase):
    def test_a3_and_cache_blend_reject_zero_ratio_with_no_recompute_direction(self):
        for eval_func in (a3_eval, cache_blend_eval):
            with self.subTest(method=eval_func.__name__):
                with self.assertRaisesRegex(ValueError, "no_recompute"):
                    eval_func(None, None, None, None, {}, "", kwargs={"recompute_ratio": 0.0})  # type: ignore[arg-type]

    def test_a3_and_cache_blend_require_ratio_in_open_closed_interval(self):
        invalid_ratios = (-0.1, 1.1, 1, None)
        for eval_func in (a3_eval, cache_blend_eval):
            for ratio in invalid_ratios:
                with self.subTest(method=eval_func.__name__, ratio=ratio):
                    with self.assertRaises(ValueError):
                        eval_func(None, None, None, None, {}, "", kwargs={"recompute_ratio": ratio})  # type: ignore[arg-type]

    def test_a3_and_cache_blend_accept_positive_float_ratios(self):
        patch_targets = (
            (a3_eval, "sempic.cache_comb.methods.a3.prepare_recompute_inputs"),
            (cache_blend_eval, "sempic.cache_comb.methods.cache_blend.prepare_recompute_inputs"),
        )
        for eval_func, patch_target in patch_targets:
            for ratio in (0.5, 1.0):
                with self.subTest(method=eval_func.__name__, ratio=ratio):
                    with patch(patch_target, side_effect=AcceptedRatio):
                        with self.assertRaises(AcceptedRatio):
                            eval_func(None, None, None, None, {}, "", kwargs={"recompute_ratio": ratio})  # type: ignore[arg-type]

    def test_a3_and_cache_blend_reject_empty_runtime_selection(self):
        patch_targets = (
            (a3_eval, "sempic.cache_comb.methods.a3.prepare_recompute_inputs"),
            (cache_blend_eval, "sempic.cache_comb.methods.cache_blend.prepare_recompute_inputs"),
        )
        for eval_func, patch_target in patch_targets:
            for ratio, candidates in ((1.0, ()), (0.5, (7,))):
                with self.subTest(
                    method=eval_func.__name__, ratio=ratio, candidates=candidates
                ):
                    inputs = types.SimpleNamespace(candidate_indices=candidates)
                    with patch(patch_target, return_value=inputs):
                        with self.assertRaisesRegex(ValueError, "no_recompute"):
                            eval_func(  # type: ignore[arg-type]
                                None,
                                None,
                                None,
                                None,
                                {},
                                "",
                                kwargs={"recompute_ratio": ratio},
                            )

    def test_recompute_inputs_keep_inline_out_of_context_candidates(self):
        prompt = TokenizedPrompt(
            input_ids=torch.arange(9),
            parts=(
                TokenSpan("inline", 0, 2),
                TokenSpan("context", 2, 4),
                TokenSpan("inline", 4, 5),
                TokenSpan("context", 5, 8),
                TokenSpan("inline", 8, 9),
            ),
        )
        model = types.SimpleNamespace(device=torch.device("cpu"))

        inputs = prepare_recompute_inputs(
            "test",
            model,
            prompt,
            MappingProxyType({1: make_cache(2), 3: make_cache(3)}),
        )

        self.assertEqual(inputs.candidate_indices, (2, 3, 5, 6, 7))
        self.assertEqual(inputs.inline_indices, (0, 1, 4))
        self.assertEqual(inputs.input_ids.tolist(), [[0, 1, 2, 3, 4, 5, 6, 7, 8]])
        self.assertEqual(
            inputs.prefix_cache[0].position_ids.tolist(),
            [[0, 1, 0, 1, 4, 0, 1, 2]],
        )
        self.assertEqual(recompute_prefix_indices(inputs, [3, 6]), [0, 1, 3, 4, 6])

    def test_context_at_position_zero_is_the_only_exact_context_prefix(self):
        prompt = TokenizedPrompt(
            input_ids=torch.arange(6),
            parts=(
                TokenSpan("context", 0, 2),
                TokenSpan("inline", 2, 3),
                TokenSpan("context", 3, 5),
                TokenSpan("inline", 5, 6),
            ),
        )

        inputs = prepare_recompute_inputs(
            "test",
            types.SimpleNamespace(device=torch.device("cpu")),
            prompt,
            MappingProxyType({0: make_cache(2), 2: make_cache(2)}),
        )

        self.assertEqual(inputs.candidate_indices, (3, 4))
        self.assertEqual(inputs.inline_indices, (2,))

    def test_epic_budgets_every_context_and_never_inline(self):
        prompt = TokenizedPrompt(
            input_ids=torch.arange(13),
            parts=(
                TokenSpan("inline", 0, 2),
                TokenSpan("context", 2, 5),
                TokenSpan("inline", 5, 7),
                TokenSpan("context", 7, 12),
                TokenSpan("inline", 12, 13),
            ),
        )

        self.assertEqual(_epic_context_indices(prompt, 2), [2, 3, 7, 8])
        self.assertEqual(_epic_context_indices(prompt, 0), [])
        self.assertEqual(
            _epic_context_indices(prompt, 99),
            [2, 3, 4, 7, 8, 9, 10, 11],
        )

if __name__ == "__main__":
    unittest.main()
