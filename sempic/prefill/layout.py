from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch


@dataclass(frozen=True, slots=True)
class PrefillSegment:
    kind: Literal["inline", "context"]
    position_ids: torch.Tensor
    canonical_start: int | None = None
    canonical_end: int | None = None
    part_index: int | None = None
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class ContextPlacement:
    part_index: int
    canonical_start: int
    canonical_end: int
    source_start: int
    source_end: int
    physical_start: int
    physical_end: int
    position_ids: torch.Tensor


@dataclass(frozen=True, slots=True)
class InterleavedPrefillLayout:
    physical_position_ids: torch.Tensor
    physical_is_context: torch.Tensor
    physical_source_indices: torch.Tensor
    inline_position_ids: torch.Tensor
    inline_canonical_indices: torch.Tensor
    inline_physical_indices: torch.Tensor
    inline_physical_frontiers: torch.Tensor
    context_position_ids: torch.Tensor
    context_physical_indices: torch.Tensor
    context_placements: tuple[ContextPlacement, ...]
    terminal_inline_row: int

    @property
    def physical_length(self) -> int:
        return int(self.physical_position_ids.numel())

    @property
    def inline_length(self) -> int:
        return int(self.inline_position_ids.numel())

    @property
    def context_length(self) -> int:
        return int(self.context_position_ids.numel())

    @property
    def terminal_position_id(self) -> torch.Tensor:
        return self.inline_position_ids[self.terminal_inline_row]


def _canonical_bounds(segment: PrefillSegment) -> tuple[int, int]:
    if segment.canonical_start is None or segment.canonical_end is None:
        if segment.canonical_start is not None or segment.canonical_end is not None:
            raise ValueError("canonical_start and canonical_end must be provided together.")
        return -1, -1
    if segment.canonical_start < 0 or segment.canonical_end < segment.canonical_start:
        raise ValueError("Canonical bounds must describe a non-negative ordered span.")
    return segment.canonical_start, segment.canonical_end


def build_interleaved_layout(
    segments: Sequence[PrefillSegment],
) -> InterleavedPrefillLayout:
    if not segments:
        raise ValueError("Interleaved prefill requires at least one segment.")

    device = segments[0].position_ids.device
    physical_positions: list[torch.Tensor] = []
    physical_is_context: list[torch.Tensor] = []
    inline_positions: list[torch.Tensor] = []
    inline_canonical_indices: list[torch.Tensor] = []
    inline_physical_indices: list[torch.Tensor] = []
    context_positions: list[torch.Tensor] = []
    context_physical_indices: list[torch.Tensor] = []
    context_placements: list[ContextPlacement] = []
    physical_sources: list[tuple[bool, int]] = []
    physical_cursor = 0
    inline_cursor = 0
    context_cursor = 0
    terminal_inline_row: int | None = None

    for segment in segments:
        if segment.position_ids.ndim != 1:
            raise ValueError("Segment position_ids must be one-dimensional.")
        if segment.position_ids.device != device:
            raise ValueError("All prefill segments must be on the same device.")
        length = int(segment.position_ids.numel())
        if length == 0:
            raise ValueError("Prefill segments must be non-empty.")
        canonical_start, canonical_end = _canonical_bounds(segment)

        physical_positions.append(segment.position_ids)
        physical_is_context.append(torch.full(
            (length,),
            segment.kind == "context",
            dtype=torch.bool,
            device=device,
        ))
        physical_range = torch.arange(
            physical_cursor,
            physical_cursor + length,
            dtype=torch.long,
            device=device,
        )

        if segment.kind == "inline":
            if segment.part_index is not None:
                raise ValueError("Inline segments cannot have a Context part_index.")
            if canonical_start == -1:
                canonical_indices = torch.full(
                    (length,), -1, dtype=torch.long, device=device
                )
            else:
                if canonical_end - canonical_start != length:
                    raise ValueError("Inline canonical spans must match their physical length.")
                canonical_indices = torch.arange(
                    canonical_start,
                    canonical_end,
                    dtype=torch.long,
                    device=device,
                )
            inline_positions.append(segment.position_ids)
            inline_canonical_indices.append(canonical_indices)
            inline_physical_indices.append(physical_range)
            physical_sources.extend((False, inline_cursor + offset) for offset in range(length))
            if segment.terminal:
                if terminal_inline_row is not None:
                    raise ValueError("Exactly one Inline segment may be terminal.")
                terminal_inline_row = inline_cursor + length - 1
            inline_cursor += length
        elif segment.kind == "context":
            if segment.part_index is None:
                raise ValueError("Context segments require part_index.")
            if segment.terminal:
                raise ValueError("Context segments cannot be terminal.")
            if canonical_start == -1:
                raise ValueError("Context segments require canonical bounds.")
            context_positions.append(segment.position_ids)
            context_physical_indices.append(physical_range)
            context_placements.append(ContextPlacement(
                part_index=segment.part_index,
                canonical_start=canonical_start,
                canonical_end=canonical_end,
                source_start=context_cursor,
                source_end=context_cursor + length,
                physical_start=physical_cursor,
                physical_end=physical_cursor + length,
                position_ids=segment.position_ids,
            ))
            physical_sources.extend((True, context_cursor + offset) for offset in range(length))
            context_cursor += length
        else:
            raise ValueError(f"Unsupported prefill segment kind: {segment.kind!r}.")
        physical_cursor += length

    if terminal_inline_row is None:
        raise ValueError("Interleaved prefill requires one terminal Inline segment.")

    source_indices = torch.tensor([
        source_index if is_context else context_cursor + source_index
        for is_context, source_index in physical_sources
    ], dtype=torch.long, device=device)
    inline_physical = torch.cat(inline_physical_indices)
    return InterleavedPrefillLayout(
        physical_position_ids=torch.cat(physical_positions),
        physical_is_context=torch.cat(physical_is_context),
        physical_source_indices=source_indices,
        inline_position_ids=torch.cat(inline_positions),
        inline_canonical_indices=torch.cat(inline_canonical_indices),
        inline_physical_indices=inline_physical,
        inline_physical_frontiers=inline_physical.clone(),
        context_position_ids=(
            torch.cat(context_positions)
            if context_positions
            else torch.empty(0, dtype=torch.long, device=device)
        ),
        context_physical_indices=(
            torch.cat(context_physical_indices)
            if context_physical_indices
            else torch.empty(0, dtype=torch.long, device=device)
        ),
        context_placements=tuple(context_placements),
        terminal_inline_row=terminal_inline_row,
    )
