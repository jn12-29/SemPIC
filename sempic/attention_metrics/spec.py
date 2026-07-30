"""Query-pass specifications for real attention statistics collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


ANALYSIS_SCHEMA_VERSION = 2
QUERY_KINDS = {
    "terminal_inline_tokens",
    "gold_answer_literal_tokens",
    "gold_answer_shifted_prediction_queries",
}
REDUCER_KEYS = {
    "attention_profile",
    "pic_retrieval",
    "raw_attention_profile",
}
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class QueryPassSpec:
    """One real forward whose ephemeral layer basis feeds several reducers."""

    query_pass_id: str
    kind: str
    reducers: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> "QueryPassSpec":
        if not isinstance(value, dict) or set(value) != {
            "query_pass_id", "kind", "reducers"
        }:
            raise ValueError("query pass has missing or unknown fields.")
        query_pass_id = value["query_pass_id"]
        if not isinstance(query_pass_id, str) or not _KEY.fullmatch(query_pass_id):
            raise ValueError("query_pass_id must be a lower-snake-case key.")
        kind = value["kind"]
        if kind not in QUERY_KINDS:
            raise ValueError(f"Unsupported query pass kind: {kind!r}.")
        reducers = value["reducers"]
        if (
            not isinstance(reducers, list)
            or not reducers
            or any(item not in REDUCER_KEYS for item in reducers)
            or len(set(reducers)) != len(reducers)
        ):
            raise ValueError("reducers must be a unique non-empty supported list.")
        if "raw_attention_profile" in reducers and not {
            "attention_profile", "pic_retrieval"
        }.issubset(reducers):
            raise ValueError(
                "raw_attention_profile requires attention_profile and pic_retrieval "
                "for the current processing pipeline."
            )
        return cls(query_pass_id, kind, tuple(reducers))

    def to_dict(self) -> dict[str, object]:
        return {
            "query_pass_id": self.query_pass_id,
            "kind": self.kind,
            "reducers": list(self.reducers),
        }

    @property
    def uses_gold_answer(self) -> bool:
        return self.kind != "terminal_inline_tokens"


@dataclass(frozen=True, slots=True)
class AttentionAnalysisConfig:
    schema_version: int
    query_passes: tuple[QueryPassSpec, ...]

    @classmethod
    def from_dict(cls, value: object) -> "AttentionAnalysisConfig":
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "query_passes"
        }:
            raise ValueError("Attention analysis config has missing or unknown fields.")
        if value["schema_version"] != ANALYSIS_SCHEMA_VERSION:
            raise ValueError("Unsupported attention analysis config version.")
        raw_passes = value["query_passes"]
        if not isinstance(raw_passes, list) or not raw_passes:
            raise ValueError("query_passes must be a non-empty list.")
        passes = tuple(QueryPassSpec.from_dict(item) for item in raw_passes)
        ids = [item.query_pass_id for item in passes]
        if len(set(ids)) != len(ids):
            raise ValueError("query_pass_id values must be unique.")
        return cls(ANALYSIS_SCHEMA_VERSION, passes)

    @classmethod
    def from_file(cls, path: str | Path) -> "AttentionAnalysisConfig":
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "query_passes": [item.to_dict() for item in self.query_passes],
        }


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "AttentionAnalysisConfig",
    "QueryPassSpec",
    "QUERY_KINDS",
    "REDUCER_KEYS",
]
