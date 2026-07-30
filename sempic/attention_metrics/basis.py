"""Ephemeral layer-level inputs shared by attention statistics reducers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache_comb.compact_prefill import LayerPrefillObservation
from ..cache_comb.recompute_kv.utils import RecomputeAttentionBasis


@dataclass(frozen=True, slots=True)
class PhysicalChunkScope:
    chunk_id: str
    token_digest: str
    pic_start: int
    pic_end: int
    scope_start: int
    scope_end: int

    def __post_init__(self) -> None:
        if not self.chunk_id or not self.token_digest:
            raise ValueError("Chunk identity fields must be non-empty.")
        if not (
            0 <= self.scope_start <= self.pic_start < self.pic_end
            <= self.scope_end
        ):
            raise ValueError("Chunk PIC and retrieval scope bounds are invalid.")

    @property
    def token_length(self) -> int:
        return self.pic_end - self.pic_start


@dataclass(frozen=True, slots=True)
class PhysicalLayout:
    physical_length: int
    chunks: tuple[PhysicalChunkScope, ...]

    def __post_init__(self) -> None:
        if self.physical_length <= 0 or not self.chunks:
            raise ValueError("Physical layout must contain keys and chunks.")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Physical layout chunk IDs must be unique.")
        for chunk in self.chunks:
            if chunk.scope_end > self.physical_length:
                raise ValueError("Chunk retrieval scope exceeds physical keys.")


@dataclass(frozen=True, slots=True)
class LayerAttentionBasis:
    """A synchronous view that reducers must not retain beyond one callback."""

    layer_index: int
    scaled_masked_logits: torch.Tensor  # [Hq, Q, K]
    attention_probabilities: torch.Tensor  # [Hq, Q, K]
    physical_values: torch.Tensor  # [Hkv, K, D]
    query_to_kv_head: torch.LongTensor  # [Hq]
    keep_mask: torch.BoolTensor  # [1, Q, K]
    layout: PhysicalLayout

    @classmethod
    def from_recompute(
        cls,
        *,
        layer_index: int,
        value: RecomputeAttentionBasis,
        layout: PhysicalLayout,
    ) -> LayerAttentionBasis:
        if any(tensor.size(0) != 1 for tensor in (
            value.scaled_masked_logits,
            value.attention_probabilities,
            value.physical_values,
            value.keep_mask,
        )):
            raise ValueError("Attention basis collection requires batch size one.")
        basis = cls(
            layer_index=layer_index,
            scaled_masked_logits=value.scaled_masked_logits.squeeze(0),
            attention_probabilities=value.attention_probabilities.squeeze(0),
            physical_values=value.physical_values.squeeze(0),
            query_to_kv_head=value.query_to_kv_head,
            keep_mask=value.keep_mask.squeeze(0),
            layout=layout,
        )
        basis.validate()
        return basis

    @classmethod
    def from_compact_prefill(
        cls,
        *,
        value: LayerPrefillObservation,
        layout: PhysicalLayout,
    ) -> LayerAttentionBasis:
        if any(tensor.size(0) != 1 for tensor in (
            value.scaled_masked_logits,
            value.probabilities,
            value.values,
            value.keep_mask,
        )):
            raise ValueError("Attention basis collection requires batch size one.")
        basis = cls(
            layer_index=value.layer_index,
            scaled_masked_logits=value.scaled_masked_logits.squeeze(0),
            attention_probabilities=value.probabilities.squeeze(0),
            physical_values=value.values.squeeze(0),
            query_to_kv_head=value.query_to_kv_head,
            keep_mask=value.keep_mask.squeeze(0),
            layout=layout,
        )
        basis.validate()
        return basis

    def validate(self) -> None:
        if self.layer_index < 0:
            raise ValueError("Attention basis layer index must be non-negative.")
        logits = self.scaled_masked_logits
        probabilities = self.attention_probabilities
        values = self.physical_values
        if logits.ndim != 3 or probabilities.shape != logits.shape:
            raise ValueError("Attention logits/probabilities must have shape [Hq,Q,K].")
        if values.ndim != 3 or values.size(1) != logits.size(2):
            raise ValueError("Physical Values must have shape [Hkv,K,D].")
        if (
            self.query_to_kv_head.ndim != 1
            or self.query_to_kv_head.numel() != logits.size(0)
            or self.query_to_kv_head.dtype != torch.long
        ):
            raise ValueError("query_to_kv_head does not map every query head.")
        if self.keep_mask.shape != (1, logits.size(1), logits.size(2)):
            raise ValueError("Attention keep mask must have shape [1,Q,K].")
        if self.keep_mask.dtype != torch.bool:
            raise ValueError("Attention keep mask must be boolean.")
        if self.layout.physical_length != logits.size(2):
            raise ValueError("Attention basis and physical layout lengths differ.")


__all__ = ["LayerAttentionBasis", "PhysicalChunkScope", "PhysicalLayout"]
