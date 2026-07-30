"""Lazy safetensors storage for generated teacher outputs."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import torch
from safetensors import safe_open


CACHE_FORMAT_VERSION = 2
CACHE_PAYLOAD_FILENAME = "cache.safetensors"
CACHE_MANIFEST_FILENAME = "manifest.json"
CACHE_RESOLVED_CONFIG_FILENAME = "resolved_config.json"
_ARTIFACT_FILENAMES = {
    CACHE_PAYLOAD_FILENAME,
    CACHE_MANIFEST_FILENAME,
    CACHE_RESOLVED_CONFIG_FILENAME,
}
_FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}
_DEFAULT_HEADER_RESERVE_BYTES = 2 * 1024 * 1024


class GenerationOutput(TypedDict):
    sequences: list[torch.Tensor]
    logits: list[torch.Tensor]
    text: list[str]


@dataclass(frozen=True)
class TensorMetadata:
    tensor: str
    dtype: str
    shape: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class GenerationEntryMetadata:
    sequences: tuple[TensorMetadata, ...]
    logits: tuple[TensorMetadata, ...]
    text: tuple[str, ...]
    content_digest: str

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return tuple(tensor.shape[0] for tensor in self.sequences)

    @property
    def sequence_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tensor.shape for tensor in self.sequences)

    @property
    def logit_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tensor.shape for tensor in self.logits)

    @property
    def has_logits(self) -> bool:
        return bool(self.logits)


class GenerationCacheReader(Protocol):
    provenance: Mapping[str, Any]
    provenance_digest: str

    def get(
        self,
        key: str,
        device: torch.device | str | None = None,
    ) -> GenerationOutput | None: ...

    def metadata(self, key: str) -> GenerationEntryMetadata | None: ...

    def keys(self) -> tuple[str, ...]: ...

    def __contains__(self, key: object) -> bool: ...

    def __len__(self) -> int: ...


_TORCH_TO_SAFETENSORS_DTYPE = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
for _torch_name, _safe_name in (
    ("uint16", "U16"),
    ("uint32", "U32"),
    ("uint64", "U64"),
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
):
    if hasattr(torch, _torch_name):
        _TORCH_TO_SAFETENSORS_DTYPE[getattr(torch, _torch_name)] = _safe_name


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Generation-cache metadata must be canonical JSON data.") from exc


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_semantic_key(key: Any, field: str) -> str:
    if (
        not isinstance(key, str)
        or len(key) != 64
        or any(character not in "0123456789abcdef" for character in key)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest.")
    return key


def _validate_provenance(value: Any) -> dict[str, Any]:
    provenance = _require_dict(value, "manifest.provenance")
    required = {"model_path", "tokenizer_path", "dtype", "tokenizer", "store_logits"}
    if set(provenance) != required:
        raise ValueError(
            "Generation-cache provenance must contain exactly model_path, "
            "tokenizer_path, dtype, tokenizer, and store_logits."
        )
    for field in ("model_path", "tokenizer_path", "dtype"):
        if not isinstance(provenance[field], str) or not provenance[field]:
            raise ValueError(f"Generation-cache provenance {field} must be non-empty.")
    if not isinstance(provenance["tokenizer"], dict):
        raise ValueError("Generation-cache provenance tokenizer must be an object.")
    if not isinstance(provenance["store_logits"], bool):
        raise ValueError("Generation-cache provenance store_logits must be boolean.")
    return json.loads(_canonical_json_bytes(provenance))


def generation_cache_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compatibility-relevant provenance from a resolved cache config."""
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Generation-cache config model must be an object.")
    model_path = model.get("model_path")
    tokenizer_path = model.get("tokenizer_path", model_path)
    dtype = model.get("dtype")
    tokenizer = model.get("tokenizer")
    store_logits = config.get("store_logits")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("Generation-cache provenance requires model.model_path.")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("Generation-cache provenance requires model.tokenizer_path.")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("Generation-cache provenance requires model.dtype.")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("Generation-cache provenance requires model.tokenizer.")
    if not isinstance(store_logits, bool):
        raise ValueError("Generation-cache provenance requires boolean store_logits.")
    return _validate_provenance({
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "dtype": dtype,
        "tokenizer": dict(tokenizer),
        "store_logits": store_logits,
    })


def _require_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    return value


def _parse_tensor_metadata(value: Any, field: str) -> TensorMetadata:
    data = _require_dict(value, field)
    tensor = data.get("tensor")
    dtype = data.get("dtype")
    shape = data.get("shape")
    digest = data.get("digest")
    if not isinstance(tensor, str) or not tensor:
        raise ValueError(f"{field}.tensor must be a non-empty string.")
    if dtype not in set(_TORCH_TO_SAFETENSORS_DTYPE.values()):
        raise ValueError(f"{field}.dtype is unsupported: {dtype!r}.")
    if (
        not isinstance(shape, list)
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
    ):
        raise ValueError(f"{field}.shape must contain non-negative integers.")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{field}.digest must be a SHA-256 hex digest.")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{field}.digest must be a SHA-256 hex digest.") from exc
    return TensorMetadata(tensor=tensor, dtype=dtype, shape=tuple(shape), digest=digest)


def _entry_digest_payload(
    sequences: Sequence[TensorMetadata],
    logits: Sequence[TensorMetadata],
    text: Sequence[str],
) -> dict[str, Any]:
    def tensor_payload(tensor: TensorMetadata) -> dict[str, Any]:
        return {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "digest": tensor.digest,
        }

    return {
        "sequences": [tensor_payload(tensor) for tensor in sequences],
        "logits": [tensor_payload(tensor) for tensor in logits],
        "text": list(text),
    }


def _parse_entry_metadata(value: Any, field: str) -> GenerationEntryMetadata:
    data = _require_dict(value, field)
    raw_sequences = data.get("sequences")
    raw_logits = data.get("logits")
    raw_text = data.get("text")
    content_digest = data.get("content_digest")
    if not isinstance(raw_sequences, list) or not raw_sequences:
        raise ValueError(f"{field}.sequences must be a non-empty list.")
    if not isinstance(raw_logits, list):
        raise ValueError(f"{field}.logits must be a list.")
    if not isinstance(raw_text, list) or any(not isinstance(item, str) for item in raw_text):
        raise ValueError(f"{field}.text must be a list of strings.")
    sequences = tuple(
        _parse_tensor_metadata(item, f"{field}.sequences[{index}]")
        for index, item in enumerate(raw_sequences)
    )
    logits = tuple(
        _parse_tensor_metadata(item, f"{field}.logits[{index}]")
        for index, item in enumerate(raw_logits)
    )
    text = tuple(raw_text)
    if len(text) != len(sequences):
        raise ValueError(f"{field}.text count must match sequences.")
    if logits and len(logits) != len(sequences):
        raise ValueError(f"{field}.logits count must be zero or match sequences.")
    for index, sequence in enumerate(sequences):
        if len(sequence.shape) != 1:
            raise ValueError(f"{field}.sequences[{index}] must be one-dimensional.")
        if sequence.dtype != "I64":
            raise ValueError(f"{field}.sequences[{index}] must use I64 token IDs.")
        if sequence.shape[0] == 0:
            raise ValueError(f"{field}.sequences[{index}] must not be empty.")
    for index, logit in enumerate(logits):
        if len(logit.shape) != 2:
            raise ValueError(f"{field}.logits[{index}] must be two-dimensional.")
        if logit.dtype not in _FLOAT_DTYPES:
            raise ValueError(f"{field}.logits[{index}] must use a floating dtype.")
        if logit.shape[1] == 0:
            raise ValueError(f"{field}.logits[{index}] vocabulary must not be empty.")
        if logit.shape[0] != sequences[index].shape[0]:
            raise ValueError(f"{field}.logits[{index}] length must match its sequence.")
    expected_digest = _json_digest(_entry_digest_payload(sequences, logits, text))
    if content_digest != expected_digest:
        raise ValueError(f"{field}.content_digest does not match entry metadata.")
    return GenerationEntryMetadata(
        sequences=sequences,
        logits=logits,
        text=text,
        content_digest=expected_digest,
    )


def _tensor_metadata_json(tensor: TensorMetadata) -> dict[str, Any]:
    return {
        "tensor": tensor.tensor,
        "dtype": tensor.dtype,
        "shape": list(tensor.shape),
        "digest": tensor.digest,
    }


def _entry_metadata_json(entry: GenerationEntryMetadata) -> dict[str, Any]:
    return {
        "sequences": [_tensor_metadata_json(tensor) for tensor in entry.sequences],
        "logits": [_tensor_metadata_json(tensor) for tensor in entry.logits],
        "text": list(entry.text),
        "content_digest": entry.content_digest,
    }


class SafetensorsGenerationCache:
    """Read one immutable generation-cache artifact lazily from CPU storage."""

    def __init__(self, artifact_dir: str | os.PathLike[str]):
        self.artifact_dir = Path(artifact_dir).expanduser().absolute()
        if not self.artifact_dir.is_dir():
            raise ValueError(
                f"Generation-cache path must be an artifact directory: {self.artifact_dir}"
            )
        artifact_names = {path.name for path in self.artifact_dir.iterdir()}
        if artifact_names != _ARTIFACT_FILENAMES:
            raise ValueError(
                "Generation-cache artifact must contain exactly "
                f"{sorted(_ARTIFACT_FILENAMES)}; found {sorted(artifact_names)}."
            )
        invalid_files = [
            name for name in _ARTIFACT_FILENAMES
            if not (self.artifact_dir / name).is_file()
            or (self.artifact_dir / name).is_symlink()
        ]
        if invalid_files:
            raise ValueError(
                "Generation-cache artifact entries must be regular files: "
                f"{sorted(invalid_files)}."
            )
        manifest_path = self.artifact_dir / CACHE_MANIFEST_FILENAME
        payload_path = self.artifact_dir / CACHE_PAYLOAD_FILENAME
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"Generation-cache manifest is missing: {manifest_path}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Generation-cache manifest is unreadable: {manifest_path}") from exc
        manifest = _require_dict(manifest, "Generation-cache manifest")
        if manifest.get("format_version") != CACHE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported generation-cache format version: "
                f"{manifest.get('format_version')!r}."
            )
        if manifest.get("storage") != "safetensors":
            raise ValueError("Generation-cache manifest storage must be 'safetensors'.")
        if manifest.get("payload") != CACHE_PAYLOAD_FILENAME:
            raise ValueError(
                f"Generation-cache manifest payload must be {CACHE_PAYLOAD_FILENAME!r}."
            )
        provenance = _validate_provenance(manifest.get("provenance"))
        provenance_digest = manifest.get("provenance_digest")
        if provenance_digest != _json_digest(provenance):
            raise ValueError("Generation-cache provenance digest does not match provenance.")
        raw_entries = _require_dict(manifest.get("entries"), "manifest.entries")
        if not raw_entries:
            raise ValueError("Generation-cache manifest entries must not be empty.")
        entries: dict[str, GenerationEntryMetadata] = {}
        tensor_owners: dict[str, str] = {}
        for key, raw_entry in raw_entries.items():
            _validate_semantic_key(key, "Generation-cache entry key")
            entry = _parse_entry_metadata(raw_entry, f"manifest.entries[{key!r}]")
            for tensor in (*entry.sequences, *entry.logits):
                previous = tensor_owners.get(tensor.tensor)
                if previous is not None:
                    raise ValueError(
                        f"Tensor {tensor.tensor!r} is referenced more than once "
                        f"by entries {previous!r} and {key!r}."
                    )
                tensor_owners[tensor.tensor] = key
            entries[key] = entry
        expects_logits = provenance["store_logits"]
        inconsistent_logits = [
            key for key, entry in entries.items()
            if bool(entry.logits) != expects_logits
        ]
        if inconsistent_logits:
            raise ValueError(
                "Generation-cache entries do not match provenance.store_logits: "
                f"{inconsistent_logits[:3]}."
            )
        vocab_sizes = {
            logit.shape[1]
            for entry in entries.values()
            for logit in entry.logits
        }
        if len(vocab_sizes) > 1:
            raise ValueError("Generation-cache logits must use one vocabulary size.")
        if not payload_path.is_file():
            raise ValueError(f"Generation-cache payload is missing: {payload_path}")
        try:
            with safe_open(payload_path, framework="pt", device="cpu") as payload:
                payload_metadata = payload.metadata()
                if payload_metadata != {
                    "format_version": str(CACHE_FORMAT_VERSION),
                    "provenance_digest": provenance_digest,
                }:
                    raise ValueError(
                        "Generation-cache payload metadata differs from manifest."
                    )
                payload_keys = set(payload.keys())
                expected_keys = set(tensor_owners)
                if payload_keys != expected_keys:
                    missing = sorted(expected_keys - payload_keys)
                    unexpected = sorted(payload_keys - expected_keys)
                    raise ValueError(
                        "Generation-cache payload tensor index differs from manifest: "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                for entry in entries.values():
                    for tensor in (*entry.sequences, *entry.logits):
                        tensor_slice = payload.get_slice(tensor.tensor)
                        if tuple(tensor_slice.get_shape()) != tensor.shape:
                            raise ValueError(
                                f"Tensor {tensor.tensor!r} shape differs from manifest."
                            )
                        if tensor_slice.get_dtype() != tensor.dtype:
                            raise ValueError(
                                f"Tensor {tensor.tensor!r} dtype differs from manifest."
                            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Generation-cache payload is unreadable: {payload_path}") from exc
        self._payload_path = payload_path
        self._entries = entries
        self._verified_keys: set[str] = set()
        self.provenance = provenance
        self.provenance_digest = provenance_digest

    def get(
        self,
        key: str,
        device: torch.device | str | None = None,
    ) -> GenerationOutput | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        destination = None if device is None else torch.device(device)
        with safe_open(self._payload_path, framework="pt", device="cpu") as payload:
            sequences = [payload.get_tensor(tensor.tensor) for tensor in entry.sequences]
            logits = [payload.get_tensor(tensor.tensor) for tensor in entry.logits]
        if key not in self._verified_keys:
            for tensor, metadata in zip(
                (*sequences, *logits),
                (*entry.sequences, *entry.logits),
                strict=True,
            ):
                if _tensor_digest(tensor) != metadata.digest:
                    raise ValueError(
                        f"Generation-cache tensor {metadata.tensor!r} digest mismatch."
                    )
            self._verified_keys.add(key)
        if destination is not None and destination.type != "cpu":
            sequences = [tensor.to(destination) for tensor in sequences]
            logits = [tensor.to(destination) for tensor in logits]
        return GenerationOutput(
            sequences=sequences,
            logits=logits,
            text=list(entry.text),
        )

    def metadata(self, key: str) -> GenerationEntryMetadata | None:
        return self._entries.get(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)


class CompositeGenerationCache:
    """Conflict-safe union of compatible generation-cache artifacts."""

    def __init__(self, caches: Sequence[GenerationCacheReader]):
        if not caches:
            raise ValueError("At least one generation cache is required.")
        self._caches = tuple(caches)
        self.provenance = dict(self._caches[0].provenance)
        self.provenance_digest = self._caches[0].provenance_digest
        self._owners: dict[str, GenerationCacheReader] = {}
        self._metadata: dict[str, GenerationEntryMetadata] = {}
        for index, cache in enumerate(self._caches):
            if cache.provenance_digest != self.provenance_digest:
                raise ValueError(
                    f"Generation cache at index {index} has incompatible provenance."
                )
            for key in cache.keys():
                metadata = cache.metadata(key)
                if metadata is None:
                    raise ValueError(f"Generation cache omitted metadata for key {key!r}.")
                previous = self._metadata.get(key)
                if previous is not None:
                    if previous.content_digest != metadata.content_digest:
                        raise ValueError(
                            f"Generation caches contain conflicting duplicate key {key!r}."
                        )
                    continue
                self._owners[key] = cache
                self._metadata[key] = metadata
        vocab_sizes = {
            logit.shape[1]
            for metadata in self._metadata.values()
            for logit in metadata.logits
        }
        if len(vocab_sizes) > 1:
            raise ValueError(
                "Composed generation caches must use one vocabulary size."
            )

    def get(
        self,
        key: str,
        device: torch.device | str | None = None,
    ) -> GenerationOutput | None:
        owner = self._owners.get(key)
        return None if owner is None else owner.get(key, device=device)

    def metadata(self, key: str) -> GenerationEntryMetadata | None:
        return self._metadata.get(key)

    def keys(self) -> tuple[str, ...]:
        return tuple(self._metadata)

    def __contains__(self, key: object) -> bool:
        return key in self._metadata

    def __len__(self) -> int:
        return len(self._metadata)


def load_generation_cache(
    path_or_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    cache_device: torch.device | str = "cpu",
) -> GenerationCacheReader:
    device = torch.device(cache_device)
    if device.type != "cpu":
        raise ValueError("Lazy generation-cache backing must use a CPU device.")
    if isinstance(path_or_paths, (str, os.PathLike)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)
        if not paths:
            raise ValueError("Generation-cache path list must not be empty.")
    canonical_paths = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError("Generation-cache path list resolves to duplicate artifacts.")
    caches = [SafetensorsGenerationCache(path) for path in canonical_paths]
    if len(caches) == 1:
        return caches[0]
    return CompositeGenerationCache(caches)


@dataclass(frozen=True)
class _PayloadTensor:
    metadata: TensorMetadata
    data_offset: int
    num_bytes: int


@dataclass(frozen=True)
class _PayloadEntry:
    metadata: GenerationEntryMetadata
    tensors: tuple[_PayloadTensor, ...]


def _write_all(output: Any, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = output.write(remaining)
        if written is None or written <= 0:
            raise OSError("Generation-cache payload write made no progress.")
        remaining = remaining[written:]


def _write_tensor_bytes(
    output: Any,
    tensor: torch.Tensor,
    digest: hashlib._Hash,
    *,
    block_size: int = 8 * 1024 * 1024,
) -> int:
    byte_view = tensor.view(torch.uint8).reshape(-1)
    total = byte_view.numel()
    for start in range(0, total, block_size):
        data = byte_view[start:start + block_size].numpy().tobytes()
        _write_all(output, data)
        digest.update(data)
    return total


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    byte_view = tensor.detach().to("cpu").contiguous().view(torch.uint8).reshape(-1)
    block_size = 8 * 1024 * 1024
    for start in range(0, byte_view.numel(), block_size):
        digest.update(byte_view[start:start + block_size].numpy().tobytes())
    return digest.hexdigest()


class StreamingGenerationCacheWriter:
    """Append entries once to a private safetensors payload, then backfill its header."""

    def __init__(
        self,
        work_dir: str | os.PathLike[str],
        *,
        provenance: Mapping[str, Any],
        _header_reserve_bytes: int = _DEFAULT_HEADER_RESERVE_BYTES,
    ):
        if _header_reserve_bytes <= 0 or _header_reserve_bytes % 8 != 0:
            raise ValueError("Safetensors header reserve must be a positive multiple of 8.")
        self.work_dir = Path(work_dir).expanduser().absolute()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.provenance = _validate_provenance(dict(provenance))
        self.provenance_digest = _json_digest(self.provenance)
        self._entries: dict[str, _PayloadEntry] = {}
        self._state = "open"
        self._vocab_size: int | None = None
        self._header_reserve_bytes = _header_reserve_bytes
        self._payload_tmp = self.work_dir / f".{CACHE_PAYLOAD_FILENAME}.tmp"
        self._payload = self._payload_tmp.open("xb+", buffering=0)
        _write_all(self._payload, struct.pack("<Q", self._header_reserve_bytes))
        _write_all(self._payload, b" " * self._header_reserve_bytes)
        self._data_start = self._payload.tell()

    def add(self, key: str, generation: Mapping[str, Any]) -> None:
        if self._state == "finalized":
            raise RuntimeError("Cannot add entries after cache finalization.")
        if self._state != "open":
            raise RuntimeError("Cannot add entries after generation-cache writer closure.")
        _validate_semantic_key(key, "Generation-cache entry key")
        raw_sequences = generation.get("sequences")
        raw_logits = generation.get("logits")
        raw_text = generation.get("text")
        if not isinstance(raw_sequences, list) or not raw_sequences:
            raise ValueError("Generation output sequences must be a non-empty list.")
        if not isinstance(raw_logits, list):
            raise ValueError("Generation output logits must be a list.")
        if not isinstance(raw_text, list) or any(not isinstance(item, str) for item in raw_text):
            raise ValueError("Generation output text must be a list of strings.")
        if len(raw_text) != len(raw_sequences):
            raise ValueError("Generation output text count must match sequences.")
        if raw_logits and len(raw_logits) != len(raw_sequences):
            raise ValueError("Generation output logits count must be zero or match sequences.")
        if bool(raw_logits) != self.provenance["store_logits"]:
            raise ValueError(
                "Generation output logits do not match provenance.store_logits."
            )
        if any(not isinstance(tensor, torch.Tensor) for tensor in (*raw_sequences, *raw_logits)):
            raise ValueError("Generation output payloads must be tensors.")
        for kind, raw_tensors, expected_rank in (
            ("sequence", raw_sequences, 1),
            ("logit", raw_logits, 2),
        ):
            for raw_tensor in raw_tensors:
                if raw_tensor.ndim != expected_rank:
                    raise ValueError(
                        f"Generation output {kind} tensors must have rank {expected_rank}."
                    )
                if raw_tensor.dtype not in _TORCH_TO_SAFETENSORS_DTYPE:
                    raise ValueError(f"Unsupported cache tensor dtype: {raw_tensor.dtype}.")
                if kind == "sequence":
                    if raw_tensor.dtype != torch.int64:
                        raise ValueError(
                            "Generation output sequences must use int64 token IDs."
                        )
                    if raw_tensor.numel() == 0:
                        raise ValueError("Generation output sequences must not be empty.")
                else:
                    if _TORCH_TO_SAFETENSORS_DTYPE[raw_tensor.dtype] not in _FLOAT_DTYPES:
                        raise ValueError(
                            "Generation output logits must use F16, BF16, F32, or F64."
                        )
                    if raw_tensor.shape[1] == 0:
                        raise ValueError(
                            "Generation output logits vocabulary must not be empty."
                        )
                    vocab_size = raw_tensor.shape[1]
                    if self._vocab_size is not None and vocab_size != self._vocab_size:
                        raise ValueError(
                            "Generation output logits must use one vocabulary size."
                        )
        if raw_logits:
            if len({tensor.shape[1] for tensor in raw_logits}) != 1:
                raise ValueError("Generation output logits must use one vocabulary size.")
            for index, (sequence, logit) in enumerate(zip(raw_sequences, raw_logits)):
                if logit.shape[0] != sequence.shape[0]:
                    raise ValueError(
                        f"Generation output logits[{index}] length must match its sequence."
                    )
        entry_vocab_size = raw_logits[0].shape[1] if raw_logits else None

        entry_start = self._payload.tell()
        tensors: list[_PayloadTensor] = []
        sequence_metadata: list[TensorMetadata] = []
        logit_metadata: list[TensorMetadata] = []
        try:
            for kind, raw_tensors, metadata_list in (
                ("sequence", raw_sequences, sequence_metadata),
                ("logit", raw_logits, logit_metadata),
            ):
                for index, raw_tensor in enumerate(raw_tensors):
                    tensor = raw_tensor.detach().to("cpu").contiguous()
                    dtype = _TORCH_TO_SAFETENSORS_DTYPE[tensor.dtype]
                    tensor_name = f"entry.{key}.{kind}.{index}"
                    digest = hashlib.sha256()
                    data_offset = self._payload.tell() - self._data_start
                    num_bytes = _write_tensor_bytes(self._payload, tensor, digest)
                    metadata = TensorMetadata(
                        tensor=tensor_name,
                        dtype=dtype,
                        shape=tuple(tensor.shape),
                        digest=digest.hexdigest(),
                    )
                    tensors.append(_PayloadTensor(metadata, data_offset, num_bytes))
                    metadata_list.append(metadata)
        except Exception:
            self._rollback_payload(entry_start)
            raise
        entry_metadata = GenerationEntryMetadata(
            sequences=tuple(sequence_metadata),
            logits=tuple(logit_metadata),
            text=tuple(raw_text),
            content_digest=_json_digest(
                _entry_digest_payload(sequence_metadata, logit_metadata, raw_text)
            ),
        )
        previous = self._entries.get(key)
        if previous is not None:
            self._rollback_payload(entry_start)
            if previous.metadata != entry_metadata:
                raise ValueError(f"Conflicting duplicate generation-cache key: {key!r}.")
            return
        self._entries[key] = _PayloadEntry(entry_metadata, tuple(tensors))
        if self._vocab_size is None:
            self._vocab_size = entry_vocab_size

    def _rollback_payload(self, offset: int) -> None:
        self._payload.seek(offset)
        self._payload.truncate()

    def metadata(self, key: str) -> GenerationEntryMetadata | None:
        entry = self._entries.get(key)
        return None if entry is None else entry.metadata

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def finalize(self) -> None:
        if self._state == "finalized":
            raise RuntimeError("Generation cache has already been finalized.")
        if self._state != "open":
            raise RuntimeError("Cannot finalize a closed generation-cache writer.")
        if not self._entries:
            raise ValueError("Cannot finalize an empty generation cache.")
        payload_path = self.work_dir / CACHE_PAYLOAD_FILENAME
        manifest_path = self.work_dir / CACHE_MANIFEST_FILENAME
        if payload_path.exists() or manifest_path.exists():
            raise FileExistsError("Generation-cache final files already exist in work directory.")
        manifest_tmp = self.work_dir / f".{CACHE_MANIFEST_FILENAME}.tmp"
        try:
            header: dict[str, Any] = {}
            for entry in self._entries.values():
                for tensor in entry.tensors:
                    header[tensor.metadata.tensor] = {
                        "dtype": tensor.metadata.dtype,
                        "shape": list(tensor.metadata.shape),
                        "data_offsets": [
                            tensor.data_offset,
                            tensor.data_offset + tensor.num_bytes,
                        ],
                    }
            header["__metadata__"] = {
                "format_version": str(CACHE_FORMAT_VERSION),
                "provenance_digest": self.provenance_digest,
            }
            header_bytes = _canonical_json_bytes(header)
            if len(header_bytes) > self._header_reserve_bytes:
                raise ValueError(
                    "Safetensors header exceeds reserved capacity: "
                    f"{len(header_bytes)} > {self._header_reserve_bytes} bytes."
                )
            self._payload.seek(0)
            _write_all(self._payload, struct.pack("<Q", self._header_reserve_bytes))
            _write_all(self._payload, header_bytes)
            _write_all(
                self._payload,
                b" " * (self._header_reserve_bytes - len(header_bytes)),
            )
            self._payload.flush()
            os.fsync(self._payload.fileno())
            self._payload.close()
            manifest = {
                "format_version": CACHE_FORMAT_VERSION,
                "storage": "safetensors",
                "payload": CACHE_PAYLOAD_FILENAME,
                "provenance": self.provenance,
                "provenance_digest": self.provenance_digest,
                "entries": {
                    key: _entry_metadata_json(entry.metadata)
                    for key, entry in self._entries.items()
                },
            }
            with manifest_tmp.open("xb") as output:
                _write_all(output, _canonical_json_bytes(manifest))
                _write_all(output, b"\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(self._payload_tmp, payload_path)
            os.replace(manifest_tmp, manifest_path)
            self._state = "finalized"
        except Exception:
            self._close_payload()
            self._payload_tmp.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
            payload_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            self._state = "closed"
            raise

    def abort(self) -> None:
        if self._state == "open":
            self._close_payload()
            self._payload_tmp.unlink(missing_ok=True)
            self._state = "closed"

    def _close_payload(self) -> None:
        if not self._payload.closed:
            self._payload.close()

    def __enter__(self) -> StreamingGenerationCacheWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc is not None or self._state != "finalized":
            self.abort()
