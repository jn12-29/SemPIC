"""Head-preserving information retrieved from reusable cache scopes."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..basis import LayerAttentionBasis


class PicRetrievalReducer:
    reducer_key = "pic_retrieval"

    def __init__(self, methods: tuple[str, ...]):
        if not methods or methods[0] != "full_recompute":
            raise ValueError("Retrieval reducers require Full Recompute first.")
        self.methods = methods
        self._full = {}
        self._statistics = defaultdict(lambda: defaultdict(dict))
        self._last_layer = {method: -1 for method in methods}
        self.query_count: int | None = None

    def consume(self, method: str, event: LayerAttentionBasis) -> None:
        if method not in self._last_layer:
            raise ValueError(f"Unknown retrieval method: {method}")
        if event.layer_index != self._last_layer[method] + 1:
            raise ValueError("Reducer layers must be complete and ordered.")
        self._last_layer[method] = event.layer_index
        probabilities = event.attention_probabilities
        query_count = probabilities.size(1)
        if self.query_count is None:
            self.query_count = query_count
        elif self.query_count != query_count:
            raise ValueError("Retrieval query count changed across methods or layers.")
        for chunk in event.layout.chunks:
            start = chunk.pic_start if method == "full_recompute" else chunk.scope_start
            end = chunk.pic_end if method == "full_recompute" else chunk.scope_end
            scope_probabilities = probabilities[:, :, start:end]
            scope_values = event.physical_values[:, start:end, :].index_select(
                0, event.query_to_kv_head
            )
            retrieval = torch.einsum(
                "hqk,hkd->hqd",
                scope_probabilities,
                scope_values,
            )
            mass = scope_probabilities.sum(dim=-1)
            key = (chunk.chunk_id, event.layer_index)
            if method == "full_recompute":
                reference = retrieval.detach().to(device="cpu", copy=True)
                reference_mass = mass.detach().to(device="cpu", copy=True)
                self._full[key] = (reference, reference_mass)
                reference_statistics = {
                    "reference_energy_sum": reference.double().square().sum(dim=(1, 2)),
                    "scope_mass_sum": reference_mass.double().sum(dim=1),
                }
                self._statistics[chunk.chunk_id]["reference"][
                    event.layer_index
                ] = reference_statistics
                continue
            try:
                reference, reference_mass = self._full[key]
            except KeyError as exc:
                raise ValueError("Candidate retrieval arrived before Full reference.") from exc
            if retrieval.shape != reference.shape:
                raise ValueError("Candidate retrieval does not align with Full heads/queries.")
            retrieval = retrieval.detach().to(device="cpu", dtype=torch.float64)
            mass = mass.detach().to(device="cpu", dtype=torch.float64)
            reference = reference.double()
            reference_mass = reference_mass.double()
            difference = retrieval - reference
            reference_norm = torch.linalg.vector_norm(reference, dim=-1)
            candidate_norm = torch.linalg.vector_norm(retrieval, dim=-1)
            valid = (reference_norm > 0) & (candidate_norm > 0)
            cosine = torch.zeros_like(reference_norm)
            cosine[valid] = 1 - torch.nn.functional.cosine_similarity(
                reference[valid], retrieval[valid], dim=-1
            )
            cosine.clamp_(0, 2)
            reference_statistics = self._statistics[
                chunk.chunk_id
            ]["reference"][event.layer_index]
            self._statistics[chunk.chunk_id][method][event.layer_index] = {
                "squared_error_sum": difference.square().sum(dim=(1, 2)),
                "cosine_distance_sum": cosine.sum(dim=1),
                "cosine_valid_count": valid.sum(dim=1).to(torch.int64),
                "candidate_scope_mass_sum": mass.sum(dim=1),
                "absolute_mass_error_sum": (
                    mass - reference_mass
                ).abs().sum(dim=1),
                "full_scope_mass_sum": reference_statistics["scope_mass_sum"],
                "reference_energy_sum": reference_statistics[
                    "reference_energy_sum"
                ],
            }

    def finish_method(self, method: str) -> None:
        if self._last_layer.get(method, -1) < 0:
            raise ValueError(f"Retrieval method emitted no layers: {method}")

    def finish_sample(self) -> dict[str, dict[str, object]]:
        layer_count = self._last_layer["full_recompute"] + 1
        if any(last + 1 != layer_count for last in self._last_layer.values()):
            raise ValueError("Retrieval methods emitted different layer counts.")
        outputs = {}
        for chunk_id, statistics in self._statistics.items():
            def stack_records(records: dict[int, dict[str, torch.Tensor]]) -> dict:
                if set(records) != set(range(layer_count)):
                    raise ValueError("Retrieval reducer statistics have missing layers.")
                fields = records[0]
                return {
                    field: torch.stack([
                        records[layer][field] for layer in range(layer_count)
                    ]).to(
                        torch.int64 if field == "cosine_valid_count" else torch.float32
                    )
                    for field in fields
                }

            outputs[chunk_id] = {
                "full_recompute": {
                    "query_count": self.query_count,
                    **stack_records(statistics["reference"]),
                },
                **{
                    method: {
                        "query_count": self.query_count,
                        **stack_records(statistics[method]),
                    }
                    for method in self.methods[1:]
                },
            }
        self._full.clear()
        return outputs
