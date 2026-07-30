"""Per-method raw attention profiles over canonical reusable-block tokens."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..basis import LayerAttentionBasis


class RawAttentionProfileReducer:
    """Reduce each method independently without retaining a Full reference."""

    reducer_key = "raw_attention_profile"

    def __init__(self, methods: tuple[str, ...]):
        if not methods or methods[0] != "full_recompute":
            raise ValueError("Attention reducers require Full Recompute first.")
        self.methods = methods
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
            values = event.attention_probabilities[
                :, :, chunk.pic_start:chunk.pic_end
            ]
            profile = values.detach().mean(
                dim=(0, 1), dtype=torch.float32
            ).to(device="cpu", copy=True)
            self._profiles[chunk.chunk_id][method][event.layer_index] = profile

    def finish_method(self, method: str) -> None:
        if self._last_layer.get(method, -1) < 0:
            raise ValueError(f"Attention method emitted no layers: {method}")

    def finish_sample(self) -> dict[str, dict[str, object]]:
        layer_count = self._last_layer["full_recompute"] + 1
        if any(last + 1 != layer_count for last in self._last_layer.values()):
            raise ValueError("Attention methods emitted different layer counts.")

        outputs = {}
        for chunk_id, methods in self._profiles.items():
            method_outputs = {}
            for method in self.methods:
                layers = methods[method]
                if set(layers) != set(range(layer_count)):
                    raise ValueError("Raw attention profile has missing layers.")
                method_outputs[method] = {
                    "raw": torch.stack(
                        [layers[layer] for layer in range(layer_count)]
                    )
                }
            outputs[chunk_id] = method_outputs
        self._profiles.clear()
        return outputs


__all__ = ["RawAttentionProfileReducer"]
