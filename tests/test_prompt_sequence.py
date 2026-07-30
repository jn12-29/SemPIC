import unittest

import torch

from sempic.prompt import (
    ContextBlock,
    Inline,
    PromptSequence,
    TokenSpan,
    TokenizedPrompt,
    compile_prompt,
    normalize_text_prompt,
)


class RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            "input_ids": [
                [len(text), *text.encode("utf-8")]
                for text in texts
            ],
        }


class EmptyTerminalTokenizer(RecordingTokenizer):
    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        encoded["input_ids"][-1] = []
        return encoded


class EmptyContextTokenizer(RecordingTokenizer):
    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        encoded["input_ids"][0] = []
        return encoded


class PromptSequenceTests(unittest.TestCase):
    def test_requires_non_empty_terminal_inline(self):
        with self.assertRaisesRegex(ValueError, "end with an Inline"):
            PromptSequence((Inline("question"), ContextBlock("document")))

        with self.assertRaisesRegex(ValueError, "non-empty Inline"):
            PromptSequence((ContextBlock("document"), Inline("")))

    def test_normalization_merges_adjacent_inline_and_drops_empty_inline(self):
        prompt = PromptSequence((
            Inline("prefix"),
            Inline(""),
            Inline(" separator"),
            ContextBlock("document"),
            Inline(" question"),
        ))

        normalized = normalize_text_prompt(prompt)

        self.assertEqual(normalized.parts, (
            Inline("prefix separator"),
            ContextBlock("document"),
            Inline(" question"),
        ))

    def test_tokenized_prompt_owns_canonical_structure_validation(self):
        invalid_prompts = [
            (
                "one-dimensional",
                lambda: TokenizedPrompt(
                    torch.tensor([[1]]),
                    (TokenSpan("inline", 0, 1),),
                ),
            ),
            (
                "at least one part",
                lambda: TokenizedPrompt(torch.tensor([], dtype=torch.long), ()),
            ),
            (
                "Unsupported token span kind",
                lambda: TokenizedPrompt(
                    torch.tensor([1]),
                    (TokenSpan("unknown", 0, 1),),  # type: ignore[arg-type]
                ),
            ),
            (
                "ordered, gap-free",
                lambda: TokenizedPrompt(
                    torch.tensor([1, 2]),
                    (TokenSpan("inline", 1, 2),),
                ),
            ),
            (
                "cover the canonical input IDs",
                lambda: TokenizedPrompt(
                    torch.tensor([1, 2]),
                    (TokenSpan("inline", 0, 1),),
                ),
            ),
            (
                "ContextBlock spans must be non-empty",
                lambda: TokenizedPrompt(
                    torch.tensor([1]),
                    (TokenSpan("context", 0, 0), TokenSpan("inline", 0, 1)),
                ),
            ),
            (
                "non-empty Inline",
                lambda: TokenizedPrompt(
                    torch.tensor([1]),
                    (TokenSpan("context", 0, 1),),
                ),
            ),
        ]

        for message, build in invalid_prompts:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build()

    def test_compiler_uses_parts_as_token_boundaries_and_one_flat_stream(self):
        tokenizer = RecordingTokenizer()
        prompt = PromptSequence((
            Inline("pre"),
            Inline("fix"),
            ContextBlock("doc-a"),
            Inline(" between "),
            ContextBlock("doc-b"),
            Inline(" question"),
        ))

        compiled = compile_prompt(tokenizer, prompt)

        self.assertEqual(
            tokenizer.calls,
            [(
                ["prefix", "doc-a", " between ", "doc-b", " question"],
                {"add_special_tokens": False, "padding": False},
            )],
        )
        self.assertEqual(
            [span.kind for span in compiled.parts],
            ["inline", "context", "inline", "context", "inline"],
        )
        self.assertEqual(compiled.parts[0].start, 0)
        self.assertEqual(compiled.parts[-1].end, compiled.input_ids.numel())
        span = compiled.parts[2]
        self.assertTrue(torch.equal(
            compiled.input_ids[span.start:span.end],
            torch.tensor([9, *b" between "]),
        ))

    def test_compiler_requires_a_terminal_token(self):
        prompt = PromptSequence((ContextBlock("document"), Inline("question")))

        with self.assertRaisesRegex(ValueError, "non-empty Inline"):
            compile_prompt(EmptyTerminalTokenizer(), prompt)

    def test_compiler_requires_context_tokens(self):
        prompt = PromptSequence((ContextBlock("document"), Inline("question")))

        with self.assertRaisesRegex(ValueError, "ContextBlock"):
            compile_prompt(EmptyContextTokenizer(), prompt)

if __name__ == "__main__":
    unittest.main()
