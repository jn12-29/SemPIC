from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


_ARTIFACT_FIELDS = (
    (
        "packet_wrapper",
        "packet_wrapper",
        "packet_wrapper_path",
        "packet_wrapper_template",
        "PacketWrapper",
    ),
    ("lora", "lora_cache", "lora_path", "lora_template", "LoRA"),
)

_TRAIN_OUTPUTS_MARKER = "train_outputs"
_RUN_SELECTOR_PATTERN = re.compile(
    r"^\d{8}_\d{6}(?:_[A-Za-z0-9][A-Za-z0-9_.-]*)?$"
)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    checkpoint_id: str | None
    checkpoint_label: str | None
    checkpoint_source_dataset: str | None
    checkpoint_scope: str
    algorithm_variant_id: str
    algorithm_variant_label: str


@dataclass(frozen=True, slots=True)
class _ArtifactPath:
    raw: str
    source_dataset: str | None
    template: str
    suffix: str
    variant_suffix: str


def _digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _artifact_path(config: Mapping[str, Any], key: str) -> str | None:
    artifact = config.get(key)
    if isinstance(artifact, str):
        path = artifact
    elif isinstance(artifact, Mapping):
        path = artifact.get("path")
    else:
        return None
    return path if isinstance(path, str) and path else None


def _analyze_path(path: str, legacy_marker: str) -> _ArtifactPath:
    normalized = path.replace("\\", "/")
    segments = normalized.split("/")
    candidates = [
        index
        for index, segment in enumerate(segments)
        if (
            segment == _TRAIN_OUTPUTS_MARKER
            and index + 4 < len(segments)
            and segments[index + 1]
            and segments[index + 2]
            and segments[index + 3]
        )
        or (
            segment == legacy_marker
            and index + 2 < len(segments)
            and segments[index + 1]
            and segments[index + 2]
        )
    ]
    if len(candidates) != 1:
        return _ArtifactPath(path, None, normalized, normalized, normalized)

    marker_index = candidates[0]
    dataset_index = marker_index + 2
    source_dataset = segments[dataset_index]
    if segments[marker_index] == legacy_marker and source_dataset.endswith("_joint"):
        source_dataset = source_dataset.removesuffix("_joint")
    if not source_dataset:
        return _ArtifactPath(path, None, normalized, normalized, normalized)

    template_segments = list(segments)
    template_segments[dataset_index] = "{dataset}"
    if segments[marker_index] == _TRAIN_OUTPUTS_MARKER:
        selector_index = marker_index + 4
        selector = segments[selector_index]
        if _RUN_SELECTOR_PATTERN.fullmatch(selector):
            del template_segments[selector_index]
    suffix = "/".join(segments[dataset_index + 1 :]) or normalized
    variant_suffix = "/".join(template_segments[dataset_index + 1 :]) or normalized
    return _ArtifactPath(
        raw=path,
        source_dataset=source_dataset,
        template="/".join(template_segments),
        suffix=suffix,
        variant_suffix=variant_suffix,
    )


def derive_checkpoint_metadata(
    config: Mapping[str, Any],
    dataset_name: str,
    method: str,
    method_label: str,
    run_label: str,
) -> CheckpointMetadata:
    artifacts: list[tuple[str, str, str, _ArtifactPath]] = []
    for config_key, legacy_marker, checkpoint_key, template_key, display_name in _ARTIFACT_FIELDS:
        path = _artifact_path(config, config_key)
        if path is not None:
            artifacts.append(
                (
                    checkpoint_key,
                    template_key,
                    display_name,
                    _analyze_path(path, legacy_marker),
                )
            )

    variant_payload: dict[str, Any] = {
        "method": method,
        "cache_comb_kwargs": (
            config.get("cache_comb", {}).get("kwargs")
            if isinstance(config.get("cache_comb"), Mapping)
            else None
        ),
    }
    for key in ("compress", "quantization"):
        if key in config:
            variant_payload[key] = config[key]

    if not artifacts:
        variant_payload["run_label"] = run_label
        return CheckpointMetadata(
            checkpoint_id=None,
            checkpoint_label=None,
            checkpoint_source_dataset=None,
            checkpoint_scope="none",
            algorithm_variant_id=_digest(variant_payload),
            algorithm_variant_label=run_label,
        )

    checkpoint_payload: dict[str, Any] = {"method": method}
    descriptions: list[str] = []
    variant_descriptions: list[str] = []
    sources: list[str | None] = []
    for checkpoint_key, template_key, display_name, artifact in artifacts:
        checkpoint_payload[checkpoint_key] = artifact.raw
        variant_payload[template_key] = artifact.template
        descriptions.append(f"{display_name}: {artifact.suffix}")
        variant_descriptions.append(f"{display_name}: {artifact.variant_suffix}")
        sources.append(artifact.source_dataset)

    source_dataset = sources[0]
    if source_dataset is None or any(source != source_dataset for source in sources):
        source_dataset = None
    scope = (
        "unresolved"
        if source_dataset is None
        else "matched"
        if source_dataset == dataset_name
        else "cross_dataset"
    )
    artifact_label = " + ".join(descriptions)
    source_label = source_dataset or "unresolved"
    return CheckpointMetadata(
        checkpoint_id=_digest(checkpoint_payload),
        checkpoint_label=f"{method_label} / {source_label} / {artifact_label}",
        checkpoint_source_dataset=source_dataset,
        checkpoint_scope=scope,
        algorithm_variant_id=_digest(variant_payload),
        algorithm_variant_label=f"{method_label} / {' + '.join(variant_descriptions)}",
    )
