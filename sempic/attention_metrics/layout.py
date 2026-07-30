"""Project the shared interleaved prefill layout for attention reducers."""

from __future__ import annotations

import hashlib

import torch

from ..prefill.layout import InterleavedPrefillLayout
from ..prompt import TokenizedPrompt
from .basis import PhysicalChunkScope, PhysicalLayout


SUPPORTED_METHODS = frozenset({
    "full_recompute",
    "no_recompute",
    "kvpacket",
    "sempic",
    "sempic_kvpacket",
})
WRAPPED_METHODS = frozenset({"kvpacket", "sempic_kvpacket"})


def _tensor_digest(values: torch.Tensor) -> str:
    contiguous = values.detach().to(device="cpu", dtype=torch.long).contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _full_context_scope(
    layout: InterleavedPrefillLayout,
    *,
    canonical_start: int,
    canonical_end: int,
) -> tuple[int, int]:
    mask = (
        (layout.inline_canonical_indices >= canonical_start)
        & (layout.inline_canonical_indices < canonical_end)
    )
    physical_indices = layout.inline_physical_indices[mask]
    expected_length = canonical_end - canonical_start
    if physical_indices.numel() != expected_length:
        raise ValueError("Full attention layout does not cover every canonical token.")
    expected = torch.arange(
        int(physical_indices[0].item()),
        int(physical_indices[0].item()) + expected_length,
        dtype=physical_indices.dtype,
        device=physical_indices.device,
    )
    if not torch.equal(physical_indices, expected):
        raise ValueError("Full attention ContextBlock is not physically contiguous.")
    return int(physical_indices[0].item()), int(physical_indices[-1].item()) + 1


def project_physical_layout(
    *,
    prompt: TokenizedPrompt,
    method_name: str,
    layout: InterleavedPrefillLayout,
    header_len: int = 0,
    trailer_len: int = 0,
) -> PhysicalLayout:
    """Return the reducer view of an authoritative interleaved layout."""
    if method_name not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported attention layout method: {method_name}")
    if header_len < 0 or trailer_len < 0:
        raise ValueError("Packet header and trailer lengths must be non-negative.")
    if method_name not in WRAPPED_METHODS and (header_len or trailer_len):
        raise ValueError(f"{method_name} layout cannot contain packet filler tokens.")

    placements = {value.part_index: value for value in layout.context_placements}
    chunks: list[PhysicalChunkScope] = []
    for part_index, span in enumerate(prompt.parts[:-1]):
        if span.kind != "context":
            continue
        canonical_length = span.end - span.start
        if method_name == "full_recompute":
            if placements:
                raise ValueError("Full Recompute cannot contain prepared Context placements.")
            scope_start, scope_end = _full_context_scope(
                layout,
                canonical_start=span.start,
                canonical_end=span.end,
            )
            pic_start, pic_end = scope_start, scope_end
        else:
            try:
                placement = placements.pop(part_index)
            except KeyError as exc:
                raise ValueError(
                    f"Interleaved layout has no Context placement for part {part_index}."
                ) from exc
            if (
                placement.canonical_start != span.start
                or placement.canonical_end != span.end
            ):
                raise ValueError(
                    f"Context placement {part_index} has mismatched canonical bounds."
                )
            scope_start = placement.physical_start
            scope_end = placement.physical_end
            if scope_end - scope_start != canonical_length + header_len + trailer_len:
                raise ValueError(
                    "Attention layout requires one physical document position per "
                    f"canonical token plus fixed packet filler; part {part_index} has "
                    f"source={canonical_length}, physical={scope_end - scope_start}, "
                    f"header={header_len}, trailer={trailer_len}."
                )
            pic_start = scope_start + header_len
            pic_end = scope_end - trailer_len

        chunks.append(PhysicalChunkScope(
            chunk_id=f"part-{part_index}:{span.start}:{span.end}",
            token_digest=_tensor_digest(prompt.input_ids[span.start:span.end]),
            pic_start=pic_start,
            pic_end=pic_end,
            scope_start=scope_start,
            scope_end=scope_end,
        ))

    if placements:
        raise ValueError("Interleaved layout contains unexpected Context placements.")
    if not chunks:
        raise ValueError("Attention query passes require at least one ContextBlock.")
    return PhysicalLayout(physical_length=layout.physical_length, chunks=tuple(chunks))


__all__ = ["SUPPORTED_METHODS", "project_physical_layout"]
