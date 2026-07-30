import unittest
from types import SimpleNamespace

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.generation.utils import (
    GenerateBeamDecoderOnlyOutput,
    GenerateDecoderOnlyOutput,
)

from sempic.prompt import TokenSpan, TokenizedPrompt
from sempic.utils.generate import (
    GenerationCache,
    generation_cache_key,
    get_generation,
    get_teacher_logits,
    resolve_generation_config,
)
from sempic.utils.train import TrainSample, build_generation_cache


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(int(token_id)) for token_id in token_ids)


class FakeGenerationModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.generation_config = GenerationConfig(max_new_tokens=2, do_sample=False)
        self.calls = []
        self.forward_calls = []
        self.generated_rows = None

    def parameters(self):
        yield torch.zeros((), dtype=torch.bfloat16)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        num_return_sequences = kwargs["generation_config"].num_return_sequences
        expanded_input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
        num_sequences = expanded_input_ids.size(0)
        if self.generated_rows is None:
            generated = torch.tensor(
                [[7 + row % 2, 9] for row in range(num_sequences)],
                dtype=torch.long,
            )
        else:
            generated = self.generated_rows[:num_sequences].clone()
        sequences = torch.cat([expanded_input_ids, generated], dim=1)
        num_beams = kwargs["generation_config"].num_beams
        if num_beams == 1:
            logits = tuple(
                torch.nn.functional.one_hot(
                    generated[:, step],
                    num_classes=11,
                ).float()
                for step in range(generated.size(1))
            )
        else:
            logits = tuple(
                torch.full((input_ids.size(0) * num_beams, 11), -1.0)
                for _ in range(generated.size(1))
            )
        output_type = (
            GenerateBeamDecoderOnlyOutput
            if num_beams > 1
            else GenerateDecoderOnlyOutput
        )
        return output_type(sequences=sequences, logits=logits)

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        logits_to_keep,
    ):
        self.forward_calls.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "logits_to_keep": logits_to_keep,
        })
        logits = torch.zeros((*input_ids.shape, 11))
        logits[:, :-1].scatter_(
            2,
            input_ids[:, 1:].unsqueeze(-1),
            1.0,
        )
        return SimpleNamespace(logits=logits.index_select(1, logits_to_keep))


def make_prompt(ids: list[int], kinds: list[str] | None = None) -> TokenizedPrompt:
    if kinds is None:
        kinds = ["inline"]
    width = len(ids) // len(kinds)
    spans = []
    start = 0
    for index, kind in enumerate(kinds):
        end = len(ids) if index == len(kinds) - 1 else start + width
        spans.append(TokenSpan(kind=kind, start=start, end=end))  # type: ignore[arg-type]
        start = end
    return TokenizedPrompt(torch.tensor(ids, dtype=torch.long), tuple(spans))


def make_semantic_payload(query: str = "Who wrote it?") -> dict:
    return {
        "documents": ["Document one.", "Document two."],
        "query": query,
        "shots": [{
            "documents": ["Example document."],
            "query": "Example?",
            "answer": "Example.",
            "task": {},
        }],
        "task": {"subset": "default", "answer_type": "text"},
    }


def make_train_sample(prompt: TokenizedPrompt, query: str) -> TrainSample:
    return TrainSample(
        prompt=prompt,
        semantic_key=generation_cache_key(make_semantic_payload(query=query)),
    )


class GenerationCacheTests(unittest.TestCase):
    def test_resolve_generation_config_fills_model_defaults_without_mutation(self):
        model = FakeGenerationModel()
        model.generation_config.eos_token_id = 9
        model.generation_config.pad_token_id = 0
        explicit = GenerationConfig(max_new_tokens=1, eos_token_id=None, pad_token_id=None)

        resolved = resolve_generation_config(model, explicit)  # type: ignore[arg-type]

        self.assertEqual(resolved.max_new_tokens, 1)
        self.assertEqual(resolved.eos_token_id, 9)
        self.assertEqual(resolved.pad_token_id, 0)
        self.assertIsNone(explicit.eos_token_id)
        self.assertIsNone(explicit.pad_token_id)

    def test_key_depends_only_on_canonical_raw_semantic_payload(self):
        first = make_semantic_payload()
        reordered = {
            "task": {"answer_type": "text", "subset": "default"},
            "shots": [{
                "task": {},
                "answer": "Example.",
                "query": "Example?",
                "documents": ["Example document."],
            }],
            "query": "Who wrote it?",
            "documents": ["Document one.", "Document two."],
        }

        self.assertEqual(
            generation_cache_key(first),
            generation_cache_key(reordered),
        )
        self.assertNotEqual(
            generation_cache_key(first),
            generation_cache_key(make_semantic_payload(query="Who edited it?")),
        )

    def test_key_rejects_rendering_and_generation_metadata(self):
        for field in (
            "token_ids",
            "tokenizer",
            "template",
            "think",
            "teacher",
            "generation",
        ):
            with self.subTest(field=field):
                payload = {**make_semantic_payload(), field: "not semantic"}
                with self.assertRaisesRegex(ValueError, "exactly these fields"):
                    generation_cache_key(payload)  # type: ignore[arg-type]

    def test_get_generation_uses_explicit_mask_and_does_not_mutate_config(self):
        model = FakeGenerationModel()
        tokenizer = FakeTokenizer()
        config = GenerationConfig(max_new_tokens=2, do_sample=False)
        input_ids = torch.tensor([[0, 1, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[0, 1, 1]], dtype=torch.long)

        output = get_generation(
            model,  # type: ignore[arg-type]
            tokenizer,  # type: ignore[arg-type]
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=config,
        )

        self.assertTrue(torch.equal(model.calls[0]["attention_mask"], attention_mask))
        self.assertIsNot(model.calls[0]["generation_config"], config)
        self.assertFalse(config.return_dict_in_generate)
        self.assertFalse(config.output_logits)
        self.assertEqual(output["sequences"][0].tolist(), [7, 9])
        self.assertEqual(output["logits"][0].shape, (2, 11))
        self.assertEqual(output["logits"][0].dtype, torch.bfloat16)
        self.assertEqual(model.forward_calls, [])
        self.assertTrue(model.calls[0]["generation_config"].output_logits)

    def test_get_generation_returns_all_sequences(self):
        model = FakeGenerationModel()
        output = get_generation(
            model,  # type: ignore[arg-type]
            FakeTokenizer(),  # type: ignore[arg-type]
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            generation_config=GenerationConfig(
                max_new_tokens=2,
                num_beams=2,
                num_return_sequences=2,
            ),
        )

        self.assertEqual(
            [sequence.tolist() for sequence in output["sequences"]],
            [[7, 9], [8, 9]],
        )
        self.assertEqual(len(output["logits"]), 2)
        self.assertEqual(output["text"], ["7,9", "8,9"])
        self.assertEqual(len(model.forward_calls), 1)
        self.assertFalse(model.calls[0]["generation_config"].output_logits)

    def test_direct_logits_map_sampled_return_sequences_across_batch_rows(self):
        model = FakeGenerationModel()
        output = get_generation(
            model,  # type: ignore[arg-type]
            FakeTokenizer(),  # type: ignore[arg-type]
            input_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
            generation_config=GenerationConfig(
                max_new_tokens=2,
                do_sample=True,
                num_return_sequences=2,
            ),
        )

        self.assertEqual(
            [sequence.tolist() for sequence in output["sequences"]],
            [[7, 9], [8, 9], [7, 9], [8, 9]],
        )
        self.assertEqual(
            [logits.argmax(dim=-1).tolist() for logits in output["logits"]],
            [[7, 9], [8, 9], [7, 9], [8, 9]],
        )
        self.assertEqual(model.forward_calls, [])

    def test_build_cache_left_pads_by_length_even_when_pad_id_is_real_content(self):
        model = FakeGenerationModel()
        tokenizer = FakeTokenizer()
        samples = [
            make_train_sample(make_prompt([5, 0]), "First query"),
            make_train_sample(make_prompt([6, 7, 8]), "Second query"),
        ]

        cache, changed = build_generation_cache(
            samples=samples,
            batch_size=2,
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,  # type: ignore[arg-type]
            store_logits=False,
        )

        self.assertTrue(changed)
        self.assertEqual(len(cache.cache), 2)
        self.assertTrue(torch.equal(
            model.calls[0]["input_ids"],
            torch.tensor([[0, 5, 0], [6, 7, 8]]),
        ))
        self.assertTrue(torch.equal(
            model.calls[0]["attention_mask"],
            torch.tensor([[0, 1, 1], [1, 1, 1]]),
        ))
        self.assertEqual(model.forward_calls, [])
        self.assertEqual(
            set(cache.cache),
            {sample["semantic_key"] for sample in samples},
        )

    def test_build_cache_supports_sampled_teacher_generation(self):
        model = FakeGenerationModel()

        cache, changed = build_generation_cache(
            samples=[make_train_sample(make_prompt([1, 2]), "Sampled query")],
            batch_size=1,
            model=model,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=GenerationConfig(max_new_tokens=2, do_sample=True),
        )

        self.assertTrue(changed)
        self.assertEqual(len(cache.cache), 1)
        self.assertTrue(model.calls[0]["generation_config"].do_sample)

    def test_build_cache_samples_duplicate_prompts_once(self):
        model = FakeGenerationModel()
        prompt = make_prompt([1, 2])
        sample = make_train_sample(prompt, "Duplicate query")

        cache, changed = build_generation_cache(
            samples=[sample, sample],
            batch_size=2,
            model=model,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=GenerationConfig(max_new_tokens=2, do_sample=True),
        )

        self.assertTrue(changed)
        self.assertEqual(len(cache.cache), 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0]["input_ids"].size(0), 1)

        same_cache, changed = build_generation_cache(
            samples=[sample, sample],
            batch_size=2,
            model=model,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=GenerationConfig(max_new_tokens=2, do_sample=True),
            generation_cache=cache,
        )

        self.assertIs(same_cache, cache)
        self.assertFalse(changed)
        self.assertEqual(len(model.calls), 1)

    def test_build_cache_can_stream_without_retaining_generated_payloads(self):
        model = FakeGenerationModel()
        sample = make_train_sample(make_prompt([1, 2]), "Streaming query")
        streamed = []

        cache, changed = build_generation_cache(
            samples=[sample],
            batch_size=1,
            model=model,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_sink=lambda key, generation: streamed.append((key, generation)),
        )

        self.assertTrue(changed)
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual([key for key, _ in streamed], [sample["semantic_key"]])
        self.assertEqual(streamed[0][1]["sequences"][0].tolist(), [7, 9])

    def test_build_cache_groups_all_teacher_sequences_by_prompt(self):
        prompts = [make_prompt([1, 2]), make_prompt([3, 4])]
        samples = [
            make_train_sample(prompt, f"Query {index}")
            for index, prompt in enumerate(prompts)
        ]

        cache, changed = build_generation_cache(
            samples=samples,
            batch_size=2,
            model=FakeGenerationModel(),  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=GenerationConfig(
                max_new_tokens=2,
                num_beams=2,
                num_return_sequences=2,
            ),
        )

        self.assertTrue(changed)
        for sample in samples:
            generation = cache.get(sample["semantic_key"])
            assert generation is not None
            self.assertEqual(
                [sequence.tolist() for sequence in generation["sequences"]],
                [[7, 9], [8, 9]],
            )
            self.assertEqual(len(generation["logits"]), 2)
            self.assertEqual(generation["text"], ["7,9", "8,9"])

    def test_beam_search_logits_are_teacher_forced_against_returned_sequences(self):
        prompt = make_prompt([1, 2])
        sample = make_train_sample(prompt, "Beam query")

        cache, changed = build_generation_cache(
            samples=[sample],
            batch_size=1,
            model=FakeGenerationModel(),  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=GenerationConfig(
                max_new_tokens=2,
                num_beams=3,
                num_return_sequences=2,
            ),
        )

        self.assertTrue(changed)
        generation = cache.get(sample["semantic_key"])
        assert generation is not None
        self.assertEqual(
            [logits.argmax(dim=-1).tolist() for logits in generation["logits"]],
            [sequence.tolist() for sequence in generation["sequences"]],
        )

    def test_resolved_multiple_eos_ids_trim_sequences_and_logits(self):
        model = FakeGenerationModel()
        model.generated_rows = torch.tensor([[7, 8, 6]])

        output = get_generation(
            model,  # type: ignore[arg-type]
            FakeTokenizer(),  # type: ignore[arg-type]
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            generation_config=GenerationConfig(
                max_new_tokens=3,
                eos_token_id=[8, 9],
            ),
        )

        self.assertEqual(output["sequences"][0].tolist(), [7, 8])
        self.assertEqual(output["logits"][0].argmax(dim=-1).tolist(), [7, 8])
        self.assertEqual(output["text"], ["7,8"])

    def test_teacher_fallback_uses_left_padding_positions_and_target_logits_only(self):
        model = FakeGenerationModel()

        logits = get_teacher_logits(
            model,  # type: ignore[arg-type]
            prompt_input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            prompt_attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            sequences=[torch.tensor([7, 9]), torch.tensor([8])],
        )

        call = model.forward_calls[0]
        self.assertTrue(torch.equal(
            call["position_ids"],
            torch.tensor([[1, 0, 1, 2, 3], [0, 1, 2, 3, 1]]),
        ))
        self.assertEqual(call["logits_to_keep"].tolist(), [2, 3])
        self.assertEqual([item.shape for item in logits], [(2, 11), (1, 11)])

    def test_direct_generation_logits_match_teacher_forcing_on_tiny_llama(self):
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            pad_token_id=0,
            eos_token_id=31,
            bos_token_id=1,
        ))
        model.eval()
        input_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

        generation = get_generation(
            model,  # type: ignore[arg-type]
            FakeTokenizer(),  # type: ignore[arg-type]
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=31,
            ),
        )
        reference = get_teacher_logits(
            model,  # type: ignore[arg-type]
            prompt_input_ids=input_ids,
            prompt_attention_mask=attention_mask,
            sequences=generation["sequences"],
        )

        for direct, expected in zip(generation["logits"], reference, strict=True):
            self.assertTrue(torch.equal(direct, expected))

    def test_missing_logits_refresh_preserves_sampled_sequences_without_generation(self):
        model = FakeGenerationModel()
        prompt = make_prompt([1, 2])
        config = GenerationConfig(max_new_tokens=2, do_sample=True)
        sample = make_train_sample(prompt, "Refresh query")
        key = sample["semantic_key"]
        sequence = torch.tensor([8, 9])
        cache = GenerationCache()
        cache.add(key, {
            "sequences": [sequence.clone()],
            "logits": [],
            "text": ["materialized sample"],
        })
        original_generation = cache.get(key)
        assert original_generation is not None
        original_sequence = original_generation["sequences"][0]
        original_text = original_generation["text"]

        refreshed, changed = build_generation_cache(
            samples=[sample],
            batch_size=1,
            model=model,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
            generation_config=config,
            generation_cache=cache,
            store_logits=True,
        )

        self.assertTrue(changed)
        generation = refreshed.get(key)
        assert generation is not None
        self.assertEqual(model.calls, [])
        self.assertIs(generation["sequences"][0], original_sequence)
        self.assertIs(generation["text"], original_text)
        self.assertTrue(torch.equal(generation["sequences"][0], sequence))
        self.assertEqual(generation["text"], ["materialized sample"])
        self.assertEqual(generation["logits"][0].argmax(dim=-1).tolist(), [8, 9])

if __name__ == "__main__":
    unittest.main()
