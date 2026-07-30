import unittest

import torch

from sempic.attention_metrics.layout import project_physical_layout
from sempic.prefill import PrefillSegment, build_interleaved_layout
from sempic.prompt import TokenSpan, TokenizedPrompt


def prompt() -> TokenizedPrompt:
    return TokenizedPrompt(
        input_ids=torch.arange(10),
        parts=(
            TokenSpan("inline", 0, 1),
            TokenSpan("context", 1, 3),
            TokenSpan("inline", 3, 5),
            TokenSpan("context", 5, 8),
            TokenSpan("inline", 8, 10),
        ),
    )


def interleaved_layout(*, header_len: int = 0, trailer_len: int = 0):
    segments = []
    for part_index, span in enumerate(prompt().parts):
        length = span.end - span.start
        if span.kind == "context":
            length += header_len + trailer_len
        segments.append(PrefillSegment(
            kind=span.kind,
            position_ids=torch.arange(length),
            canonical_start=span.start,
            canonical_end=span.end,
            part_index=part_index if span.kind == "context" else None,
            terminal=part_index == len(prompt().parts) - 1,
        ))
    return build_interleaved_layout(segments)


class AttentionLayoutTests(unittest.TestCase):
    def test_full_and_unwrapped_project_identical_canonical_tokens(self):
        full_layout = build_interleaved_layout((PrefillSegment(
            kind="inline",
            position_ids=torch.arange(10),
            canonical_start=0,
            canonical_end=10,
            terminal=True,
        ),))
        full = project_physical_layout(
            prompt=prompt(), method_name="full_recompute", layout=full_layout
        )

        self.assertEqual(
            [(chunk.pic_start, chunk.pic_end) for chunk in full.chunks],
            [(1, 3), (5, 8)],
        )
        for method_name in ("no_recompute", "sempic"):
            with self.subTest(method=method_name):
                actual = project_physical_layout(
                    prompt=prompt(),
                    method_name=method_name,
                    layout=interleaved_layout(),
                )
                self.assertEqual(actual, full)

    def test_packet_projection_marks_pic_inside_shared_wrapper_scopes(self):
        expected = [(2, 4, 1, 6), (9, 12, 8, 14)]
        for method_name in ("kvpacket", "sempic_kvpacket"):
            with self.subTest(method=method_name):
                actual = project_physical_layout(
                    prompt=prompt(),
                    method_name=method_name,
                    layout=interleaved_layout(header_len=1, trailer_len=2),
                    header_len=1,
                    trailer_len=2,
                )
                self.assertEqual(
                    [
                        (chunk.pic_start, chunk.pic_end,
                         chunk.scope_start, chunk.scope_end)
                        for chunk in actual.chunks
                    ],
                    expected,
                )

    def test_projection_rejects_compression_like_non_one_to_one_scope(self):
        segments = (
            PrefillSegment("inline", torch.arange(1), 0, 1),
            PrefillSegment("context", torch.arange(1), 1, 3, part_index=1),
            PrefillSegment("inline", torch.arange(7), 3, 10, terminal=True),
        )
        with self.assertRaisesRegex(ValueError, "one physical document position"):
            project_physical_layout(
                prompt=prompt(),
                method_name="no_recompute",
                layout=build_interleaved_layout(segments),
            )

    def test_projection_rejects_unsupported_method_and_wrong_filler_metadata(self):
        layout = interleaved_layout()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            project_physical_layout(
                prompt=prompt(), method_name="a3", layout=layout
            )
        with self.assertRaisesRegex(ValueError, "cannot contain packet filler"):
            project_physical_layout(
                prompt=prompt(),
                method_name="sempic",
                layout=layout,
                header_len=1,
            )


if __name__ == "__main__":
    unittest.main()
