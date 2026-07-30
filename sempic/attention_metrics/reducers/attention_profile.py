"""Chunk-position attention profiles from a shared layer basis."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..basis import LayerAttentionBasis


def _conditional(values: torch.Tensor) -> torch.Tensor:
    mass = values.sum(dim=-1, keepdim=True)
    return torch.where(mass > 0, values / mass.clamp_min(torch.finfo(values.dtype).tiny), 0)


class AttentionProfileReducer:
    reducer_key = "attention_profile"

    def __init__(self, methods: tuple[str, ...]):
        if not methods or methods[0] != "full_recompute":
            raise ValueError("Attention reducers require Full Recompute first.")
        self.methods = methods
        self._full = {}
        self._profiles = defaultdict(lambda: defaultdict(dict))
        self._last_layer = {method: -1 for method in methods}
        self.query_count: int | None = None

    def consume(self, method: str, event: LayerAttentionBasis) -> None:
        if method not in self._last_layer:
            raise ValueError(f"Unknown attention method: {method}")
        if event.layer_index != self._last_layer[method] + 1:
            raise ValueError("Reducer layers must be complete and ordered.")
        self._last_layer[method] = event.layer_index
        query_count = event.attention_probabilities.size(1)
        if self.query_count is None:
            self.query_count = query_count
        elif self.query_count != query_count:
            raise ValueError("Attention query count changed across methods or layers.")
        for chunk in event.layout.chunks:
            stored_values = event.attention_probabilities[
                :, :, chunk.pic_start:chunk.pic_end
            ].detach().to(device="cpu", copy=True)
            values = stored_values.float()
            key = (chunk.chunk_id, event.layer_index)
            if method == "full_recompute":
                self._full[key] = stored_values
                self._profiles[chunk.chunk_id]["full_raw"][event.layer_index] = (
                    values.mean(dim=(0, 1))
                )
                self._profiles[chunk.chunk_id]["full_conditional"][event.layer_index] = (
                    _conditional(values).mean(dim=(0, 1))
                )
                continue
            try:
                stored_reference = self._full[key]
            except KeyError as exc:
                raise ValueError("Candidate attention arrived before Full reference.") from exc
            if stored_values.shape != stored_reference.shape:
                raise ValueError("Candidate PIC attention does not align with Full.")
            reference = stored_reference.float()
            self._profiles[chunk.chunk_id][f"{method}:raw"][event.layer_index] = (
                (values - reference).abs().mean(dim=(0, 1))
            )
            self._profiles[chunk.chunk_id][f"{method}:conditional"][event.layer_index] = (
                (_conditional(values) - _conditional(reference))
                .abs().mean(dim=(0, 1))
            )

    def finish_method(self, method: str) -> None:
        if self._last_layer.get(method, -1) < 0:
            raise ValueError(f"Attention method emitted no layers: {method}")

    def finish_sample(self) -> dict[str, dict[str, object]]:
        layer_count = self._last_layer["full_recompute"] + 1
        if any(last + 1 != layer_count for last in self._last_layer.values()):
            raise ValueError("Attention methods emitted different layer counts.")
        outputs = {}
        for chunk_id, profiles in self._profiles.items():
            def stacked(key: str) -> torch.Tensor:
                layers = profiles[key]
                if set(layers) != set(range(layer_count)):
                    raise ValueError("Attention reducer profile has missing layers.")
                return torch.stack([layers[layer] for layer in range(layer_count)])

            outputs[chunk_id] = {
                "full_recompute": {
                    "raw": stacked("full_raw"),
                    "chunk_conditional": stacked("full_conditional"),
                },
                **{
                    method: {
                        "raw_absolute_error": stacked(f"{method}:raw"),
                        "chunk_conditional_absolute_error": stacked(
                            f"{method}:conditional"
                        ),
                    }
                    for method in self.methods[1:]
                },
            }
        self._full.clear()
        return outputs
