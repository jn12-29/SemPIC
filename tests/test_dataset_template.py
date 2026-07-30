import re
import unittest
from unittest import mock

import sempic.dataset as dataset_module
from sempic.dataset import get_ret_eval_generator
from sempic.dataset.abc import build_prompt_sequence
from sempic.dataset.template import TOKENIZER_CHAT_SENTINEL
from sempic.prompt import ContextBlock, Inline, PromptSequence, compile_prompt


BOUNDARY_SENTINEL_RE = re.compile(re.escape(TOKENIZER_CHAT_SENTINEL) + r"\[\d+\]")


class FakeChatTokenizer:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        self.calls.append({
            "messages": messages,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        })
        content = messages[-1]["content"]
        if self.mode == "missing_sentinel":
            content = BOUNDARY_SENTINEL_RE.sub("", content, count=1)
        elif self.mode == "repeated_sentinel":
            first_sentinel = BOUNDARY_SENTINEL_RE.search(content)
            if first_sentinel is not None:
                content = f"{content}{first_sentinel.group(0)}"
        elif self.mode == "out_of_order":
            sentinels = BOUNDARY_SENTINEL_RE.findall(content)
            segments = BOUNDARY_SENTINEL_RE.split(content)
            if len(sentinels) >= 2:
                pieces = [
                    segments[0],
                    sentinels[1],
                    segments[1],
                    sentinels[0],
                    segments[2],
                ]
                for idx in range(2, len(sentinels)):
                    pieces.extend([sentinels[idx], segments[idx + 1]])
                content = "".join(pieces)
        elif self.mode == "trim_user_content":
            content = content.strip()
        elif self.mode == "modify_context":
            content = content.replace("doc0", "changed", 1)
        return f"<chat>{content}</chat>"


class DatasetTemplateTests(unittest.TestCase):
    def render_entry(
        self,
        entry,
        tokenizer,
        template="tokenizer_chat",
        template_kwargs=None,
    ):
        def fake_generator(**kwargs):
            del kwargs
            yield entry

        with mock.patch.dict(
            dataset_module.RET_EVAL_GENERATOR_DICT,
            {"fake_dataset": fake_generator},
        ):
            return list(get_ret_eval_generator(
                name="fake_dataset",
                num_samples=1,
                num_data_strs=2,
                num_shots=0,
                subset="default",
                split="test",
                seed=0,
                template=template,
                template_kwargs=template_kwargs or {},
                tokenizer=tokenizer,
            ))[0]

    def make_entry(self, prompt=None, **overrides):
        semantic = {
            "documents": ["doc0", "doc1"],
            "query": "query",
            "shots": [],
            "task": {},
        }
        entry = {
            "prompt": prompt or build_prompt_sequence(
                "preamble",
                ["doc0", "doc1"],
                "task",
            ),
            "query": "query",
            "answer": "answer",
            "semantic": semantic,
        }
        entry.update(overrides)
        return entry

    def test_default_composition_uses_typed_parts_and_explicit_separators(self):
        prompt = build_prompt_sequence("preamble", ["doc0", "doc1"], "task")

        self.assertEqual(prompt.parts, (
            Inline("preamble"),
            ContextBlock("doc0"),
            Inline(" "),
            ContextBlock("doc1"),
            Inline(" task"),
        ))

    def test_default_composition_compiles_to_canonical_spans(self):
        class CharTokenizer:
            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": [list(text.encode("utf-8")) for text in texts]}

        tokenizer = CharTokenizer()
        prompt = build_prompt_sequence("", ["aa", "bbb"], "q")

        compiled = compile_prompt(tokenizer, prompt)

        self.assertEqual(prompt.parts, (
            ContextBlock("aa"),
            Inline(" "),
            ContextBlock("bbb"),
            Inline(" q"),
        ))
        self.assertEqual(
            [(span.kind, span.start, span.end) for span in compiled.parts],
            [
                ("context", 0, 2),
                ("inline", 2, 3),
                ("context", 3, 6),
                ("inline", 6, 8),
            ],
        )
        self.assertEqual(tokenizer.kwargs, {"add_special_tokens": False, "padding": False})

    def test_tokenizer_chat_preserves_context_and_defaults_enable_thinking(self):
        tokenizer = FakeChatTokenizer()

        rendered = self.render_entry(self.make_entry(), tokenizer)

        self.assertEqual(rendered["prompt"].parts, (
            Inline("<chat>preamble"),
            ContextBlock("doc0"),
            Inline(" "),
            ContextBlock("doc1"),
            Inline(" task</chat>"),
        ))
        self.assertEqual(rendered["query"], "query")
        self.assertEqual(rendered["answer"], "answer")
        self.assertEqual(len(tokenizer.calls), 1)
        self.assertFalse(tokenizer.calls[0]["tokenize"])
        self.assertTrue(tokenizer.calls[0]["add_generation_prompt"])
        self.assertFalse(tokenizer.calls[0]["enable_thinking"])

    def test_tokenizer_chat_passes_enable_thinking_true_and_system_prompt(self):
        tokenizer = FakeChatTokenizer()

        self.render_entry(
            self.make_entry(),
            tokenizer,
            template_kwargs={
                "system_prompt": "system",
                "enable_thinking": True,
            },
        )

        self.assertTrue(tokenizer.calls[0]["enable_thinking"])
        self.assertEqual(
            tokenizer.calls[0]["messages"][0],
            {"role": "system", "content": "system"},
        )

    def test_tokenizer_chat_accepts_trailing_terminal_whitespace_trim(self):
        tokenizer = FakeChatTokenizer(mode="trim_user_content")
        prompt = build_prompt_sequence("preamble", ["doc0", "doc1"], "task\n")

        rendered = self.render_entry(self.make_entry(prompt), tokenizer)

        self.assertEqual(
            [part.content for part in rendered["prompt"].parts if isinstance(part, ContextBlock)],
            ["doc0", "doc1"],
        )
        self.assertEqual(rendered["prompt"].parts[-1], Inline(" task</chat>"))

    def test_tokenizer_chat_keeps_wrapper_owned_by_inline_with_empty_preamble(self):
        prompt = build_prompt_sequence("", ["doc"], "task")

        rendered = self.render_entry(self.make_entry(prompt), FakeChatTokenizer())

        self.assertEqual(rendered["prompt"].parts, (
            Inline("<chat>"),
            ContextBlock("doc"),
            Inline(" task</chat>"),
        ))

    def test_tokenizer_chat_preserves_interleaved_inline_parts(self):
        prompt = PromptSequence((
            Inline("preamble"),
            ContextBlock("doc0"),
            Inline("\nSource 2:\n"),
            ContextBlock("doc1"),
            Inline("question"),
        ))

        rendered = self.render_entry(self.make_entry(prompt), FakeChatTokenizer())

        self.assertEqual(rendered["prompt"].parts, (
            Inline("<chat>preamble"),
            ContextBlock("doc0"),
            Inline("\nSource 2:\n"),
            ContextBlock("doc1"),
            Inline("question</chat>"),
        ))

    def test_tokenizer_chat_rejects_modified_context_content(self):
        with self.assertRaisesRegex(ValueError, "modified ContextBlock content"):
            self.render_entry(
                self.make_entry(),
                FakeChatTokenizer(mode="modify_context"),
            )

    def test_manual_templates_preserve_context_and_put_wrappers_in_inline(self):
        for template in ("llama_chat", "qwen_3_chat"):
            with self.subTest(template=template):
                rendered = self.render_entry(
                    self.make_entry(),
                    tokenizer=None,
                    template=template,
                )
                parts = rendered["prompt"].parts
                self.assertIsInstance(parts[0], Inline)
                self.assertIsInstance(parts[-1], Inline)
                self.assertEqual(
                    [part.content for part in parts if isinstance(part, ContextBlock)],
                    ["doc0", "doc1"],
                )

    def test_all_templates_preserve_semantic_payload_verbatim(self):
        entry = self.make_entry()
        semantic = entry["semantic"]

        for template, tokenizer in (
            ("default", None),
            ("llama_chat", None),
            ("qwen_3_chat", None),
            ("tokenizer_chat", FakeChatTokenizer()),
        ):
            with self.subTest(template=template):
                rendered = self.render_entry(entry, tokenizer, template=template)
                self.assertIs(rendered["semantic"], semantic)

    def test_tokenizer_chat_rejects_old_think_kwarg(self):
        with self.assertRaisesRegex(ValueError, "Unknown tokenizer_chat template kwargs"):
            self.render_entry(
                self.make_entry(),
                FakeChatTokenizer(),
                template_kwargs={"think": True},
            )

    def test_tokenizer_chat_requires_apply_chat_template(self):
        with self.assertRaisesRegex(ValueError, "apply_chat_template"):
            self.render_entry(self.make_entry(), object())

    def test_tokenizer_chat_rejects_sentinel_collision(self):
        prompt = build_prompt_sequence(
            f"bad {TOKENIZER_CHAT_SENTINEL}",
            ["doc0", "doc1"],
            "task",
        )

        with self.assertRaisesRegex(ValueError, "sentinel collision"):
            self.render_entry(self.make_entry(prompt), FakeChatTokenizer())

    def test_tokenizer_chat_rejects_missing_sentinel_after_render(self):
        with self.assertRaisesRegex(ValueError, "sentinel count mismatch"):
            self.render_entry(
                self.make_entry(),
                FakeChatTokenizer(mode="missing_sentinel"),
            )

    def test_tokenizer_chat_rejects_repeated_sentinel_after_render(self):
        with self.assertRaisesRegex(ValueError, "sentinel count mismatch"):
            self.render_entry(
                self.make_entry(),
                FakeChatTokenizer(mode="repeated_sentinel"),
            )

    def test_tokenizer_chat_rejects_out_of_order_segments_after_render(self):
        with self.assertRaisesRegex(ValueError, "out of order"):
            self.render_entry(
                self.make_entry(),
                FakeChatTokenizer(mode="out_of_order"),
            )


if __name__ == "__main__":
    unittest.main()
