import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open as real_safe_open

import sempic.utils.generation_cache as generation_cache_module
from sempic.utils.generation_cache import (
    CACHE_MANIFEST_FILENAME,
    CACHE_PAYLOAD_FILENAME,
    CompositeGenerationCache,
    SafetensorsGenerationCache,
    StreamingGenerationCacheWriter,
    generation_cache_provenance,
    load_generation_cache,
)


PROVENANCE = {
    "model_path": "teacher",
    "tokenizer_path": "tokenizer",
    "dtype": "bfloat16",
    "tokenizer": {"pad_token_id": 0, "eos_token_id": 2},
    "store_logits": True,
}


def semantic_key(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


FIRST = semantic_key("first")
SECOND = semantic_key("second")
SHARED = semantic_key("shared")


def generation(token_id: int, *, text: str | None = None) -> dict:
    return {
        "sequences": [torch.tensor([token_id, 2], dtype=torch.long)],
        "logits": [torch.arange(10, dtype=torch.bfloat16).reshape(2, 5)],
        "text": [text or f"answer-{token_id}"],
    }


def write_artifact(path: Path, entries: dict[str, dict], provenance=PROVENANCE) -> None:
    writer = StreamingGenerationCacheWriter(
        path,
        provenance=provenance,
        _header_reserve_bytes=64 * 1024,
    )
    for key, output in entries.items():
        writer.add(key, output)
    writer.finalize()
    (path / "resolved_config.json").write_text("{}\n", encoding="utf-8")


class GenerationCacheStorageTests(unittest.TestCase):
    def test_streaming_writer_finalizes_one_payload_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            writer = StreamingGenerationCacheWriter(
                artifact,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            writer.add(FIRST, generation(3))

            self.assertEqual(writer.keys(), (FIRST,))
            self.assertTrue((artifact / ".cache.safetensors.tmp").is_file())
            self.assertFalse((artifact / ".cache_chunks").exists())
            self.assertEqual(writer.metadata(FIRST).sequence_lengths, (2,))  # type: ignore[union-attr]

            writer.finalize()
            (artifact / "resolved_config.json").write_text("{}\n", encoding="utf-8")

            self.assertEqual(
                sorted(path.name for path in artifact.iterdir()),
                [CACHE_PAYLOAD_FILENAME, CACHE_MANIFEST_FILENAME, "resolved_config.json"],
            )
            restored = SafetensorsGenerationCache(artifact)
            output = restored.get(FIRST)
            assert output is not None
            self.assertTrue(torch.equal(output["sequences"][0], torch.tensor([3, 2])))
            self.assertEqual(output["logits"][0].dtype, torch.bfloat16)
            self.assertEqual(output["text"], ["answer-3"])

    def test_reader_validates_header_without_getting_payload_and_get_is_entry_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            write_artifact(artifact, {FIRST: generation(3), SECOND: generation(4)})
            requested: list[str] = []

            class TrackingSafeOpen:
                def __init__(self, *args, **kwargs):
                    self._inner = real_safe_open(*args, **kwargs)

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, *args):
                    return self._inner.__exit__(*args)

                def keys(self):
                    return self._inner.keys()

                def metadata(self):
                    return self._inner.metadata()

                def get_slice(self, name):
                    return self._inner.get_slice(name)

                def get_tensor(self, name):
                    requested.append(name)
                    return self._inner.get_tensor(name)

            with patch(
                "sempic.utils.generation_cache.safe_open",
                side_effect=TrackingSafeOpen,
            ):
                reader = SafetensorsGenerationCache(artifact)
                self.assertEqual(requested, [])
                output = reader.get(SECOND)

            assert output is not None
            self.assertEqual(
                requested,
                [f"entry.{SECOND}.sequence.0", f"entry.{SECOND}.logit.0"],
            )
            self.assertEqual(output["text"], ["answer-4"])

    def test_loader_composes_union_and_accepts_identical_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            write_artifact(first, {SHARED: generation(3), FIRST: generation(4)})
            write_artifact(second, {SHARED: generation(3), SECOND: generation(5)})

            reader = load_generation_cache([first, second])

            self.assertIsInstance(reader, CompositeGenerationCache)
            self.assertEqual(set(reader.keys()), {SHARED, FIRST, SECOND})
            self.assertEqual(len(reader), 3)
            self.assertEqual(reader.get(SECOND)["text"], ["answer-5"])  # type: ignore[index]

    def test_composite_rejects_conflicting_duplicates_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            conflicting = root / "conflicting"
            incompatible = root / "incompatible"
            write_artifact(first, {SHARED: generation(3)})
            write_artifact(conflicting, {SHARED: generation(4)})
            write_artifact(
                incompatible,
                {semantic_key("other"): generation(5)},
                provenance={**PROVENANCE, "model_path": "other-teacher"},
            )

            with self.assertRaisesRegex(ValueError, "conflicting duplicate key"):
                load_generation_cache([first, conflicting])
            with self.assertRaisesRegex(ValueError, "incompatible provenance"):
                load_generation_cache([first, incompatible])

    def test_different_generation_settings_do_not_change_compatibility_provenance(self):
        base = {
            "model": {
                "model_path": "teacher",
                "tokenizer_path": "tokenizer",
                "dtype": "bfloat16",
                "tokenizer": {"pad_token_id": 0, "eos_token_id": 2},
            },
            "store_logits": True,
        }
        short = generation_cache_provenance({
            **base,
            "model": {**base["model"], "generation_kwargs": {"max_new_tokens": 128}},
        })
        long = generation_cache_provenance({
            **base,
            "model": {**base["model"], "generation_kwargs": {"max_new_tokens": 512}},
        })

        self.assertEqual(short, long)

    def test_writer_rejects_conflicting_duplicate_and_allows_identical_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = StreamingGenerationCacheWriter(
                temporary,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            writer.add(SHARED, generation(3))
            payload_tmp = Path(temporary) / ".cache.safetensors.tmp"
            size_after_first = payload_tmp.stat().st_size
            writer.add(SHARED, generation(3))
            self.assertEqual(len(writer), 1)
            self.assertEqual(payload_tmp.stat().st_size, size_after_first)

            with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                writer.add(SHARED, generation(4))
            self.assertEqual(payload_tmp.stat().st_size, size_after_first)
            writer.abort()

    def test_finalize_backfills_header_without_rewriting_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            writer = StreamingGenerationCacheWriter(
                artifact,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            writer.add(FIRST, generation(3))
            payload_tmp = artifact / ".cache.safetensors.tmp"
            inode_before = payload_tmp.stat().st_ino
            with payload_tmp.open("rb") as payload:
                header_size = struct.unpack("<Q", payload.read(8))[0]
                payload.seek(8 + header_size)
                data_before = payload.read()

            writer.finalize()

            payload_path = artifact / CACHE_PAYLOAD_FILENAME
            self.assertEqual(payload_path.stat().st_ino, inode_before)
            with payload_path.open("rb") as payload:
                finalized_header_size = struct.unpack("<Q", payload.read(8))[0]
                header = json.loads(payload.read(finalized_header_size))
                payload.seek(8 + finalized_header_size)
                self.assertEqual(payload.read(), data_before)
                data_size = payload_path.stat().st_size - 8 - finalized_header_size
            offsets = sorted(
                value["data_offsets"]
                for name, value in header.items()
                if name != "__metadata__"
            )
            self.assertEqual(offsets[0][0], 0)
            self.assertTrue(all(left[1] == right[0] for left, right in zip(offsets, offsets[1:])))
            self.assertEqual(offsets[-1][1], data_size)
            with real_safe_open(payload_path, framework="pt", device="cpu") as payload:
                self.assertEqual(
                    set(payload.keys()),
                    {f"entry.{FIRST}.sequence.0", f"entry.{FIRST}.logit.0"},
                )

    def test_header_capacity_failure_cleans_private_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            writer = StreamingGenerationCacheWriter(
                artifact,
                provenance=PROVENANCE,
                _header_reserve_bytes=8,
            )
            writer.add(FIRST, generation(3))

            with self.assertRaisesRegex(ValueError, "header exceeds reserved capacity"):
                writer.finalize()

            self.assertEqual(list(artifact.iterdir()), [])

    def test_partial_tensor_write_rolls_back_to_previous_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            writer = StreamingGenerationCacheWriter(
                artifact,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            writer.add(FIRST, generation(3))
            payload_tmp = artifact / ".cache.safetensors.tmp"
            size_after_first = payload_tmp.stat().st_size

            def fail_after_partial_write(output, tensor, digest, **kwargs):
                output.write(b"partial")
                raise OSError("injected write failure")

            with patch.object(
                generation_cache_module,
                "_write_tensor_bytes",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    writer.add(SECOND, generation(4))

            self.assertEqual(payload_tmp.stat().st_size, size_after_first)
            self.assertEqual(writer.keys(), (FIRST,))
            writer.add(SECOND, generation(4))
            writer.finalize()
            with real_safe_open(
                artifact / CACHE_PAYLOAD_FILENAME,
                framework="pt",
                device="cpu",
            ) as payload:
                self.assertEqual(len(payload.keys()), 4)

    def test_context_error_removes_private_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                with StreamingGenerationCacheWriter(
                    artifact,
                    provenance=PROVENANCE,
                    _header_reserve_bytes=64 * 1024,
                ) as writer:
                    writer.add(FIRST, generation(3))
                    raise RuntimeError("injected failure")

            self.assertEqual(list(artifact.iterdir()), [])

    def test_write_all_handles_short_writes(self):
        class ShortWriter:
            def __init__(self):
                self.data = bytearray()

            def write(self, data):
                written = min(3, len(data))
                self.data.extend(data[:written])
                return written

        output = ShortWriter()
        generation_cache_module._write_all(output, b"0123456789")
        self.assertEqual(output.data, b"0123456789")

    def test_failed_first_entry_does_not_commit_vocabulary_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = StreamingGenerationCacheWriter(
                temporary,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            with patch.object(
                generation_cache_module,
                "_write_tensor_bytes",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    writer.add(FIRST, generation(3))

            different_vocab = generation(4)
            different_vocab["logits"] = [torch.zeros(2, 7, dtype=torch.bfloat16)]
            writer.add(SECOND, different_vocab)
            writer.abort()

    def test_abort_and_finalize_failure_close_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = StreamingGenerationCacheWriter(
                temporary,
                provenance=PROVENANCE,
                _header_reserve_bytes=64 * 1024,
            )
            writer.add(FIRST, generation(3))
            writer.abort()
            writer.abort()
            with self.assertRaisesRegex(RuntimeError, "writer closure"):
                writer.add(SECOND, generation(4))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                writer.finalize()

        with tempfile.TemporaryDirectory() as temporary:
            writer = StreamingGenerationCacheWriter(
                temporary,
                provenance=PROVENANCE,
                _header_reserve_bytes=8,
            )
            writer.add(FIRST, generation(3))
            with self.assertRaisesRegex(ValueError, "header exceeds reserved capacity"):
                writer.finalize()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                writer.finalize()

    def test_writer_rejects_semantically_invalid_tensor_types_and_logits_mode(self):
        invalid_outputs = (
            {
                "sequences": [torch.tensor([1.0])],
                "logits": [torch.zeros(1, 2)],
                "text": ["x"],
            },
            {
                "sequences": [torch.tensor([], dtype=torch.long)],
                "logits": [torch.zeros(0, 2)],
                "text": ["x"],
            },
            {
                "sequences": [torch.tensor([1])],
                "logits": [torch.ones(1, 2, dtype=torch.long)],
                "text": ["x"],
            },
            {
                "sequences": [torch.tensor([1])],
                "logits": [],
                "text": ["x"],
            },
        )
        for output in invalid_outputs:
            with self.subTest(output=output), tempfile.TemporaryDirectory() as temporary:
                writer = StreamingGenerationCacheWriter(
                    temporary,
                    provenance=PROVENANCE,
                    _header_reserve_bytes=64 * 1024,
                )
                with self.assertRaises(ValueError):
                    writer.add(FIRST, output)
                writer.abort()

    def test_manifest_tampering_and_non_cpu_backing_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            write_artifact(artifact, {FIRST: generation(3)})
            manifest_path = artifact / CACHE_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text())
            manifest["entries"][FIRST]["sequences"][0]["shape"] = [3]
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "length|content_digest"):
                SafetensorsGenerationCache(artifact)
            with self.assertRaisesRegex(ValueError, "CPU device"):
                load_generation_cache(artifact, cache_device="cuda:0")

    def test_payload_corruption_is_detected_when_entry_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            write_artifact(artifact, {FIRST: generation(3)})
            payload_path = artifact / CACHE_PAYLOAD_FILENAME
            with payload_path.open("r+b") as payload:
                payload.seek(-1, 2)
                value = payload.read(1)
                payload.seek(-1, 2)
                payload.write(bytes([value[0] ^ 1]))

            reader = SafetensorsGenerationCache(artifact)
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                reader.get(FIRST)

    def test_provenance_excludes_dataset_and_runtime_fields(self):
        config = {
            "model": {
                "model_path": "teacher",
                "tokenizer_path": "tokenizer",
                "dtype": "bfloat16",
                "device": "cuda:0",
                "generation_kwargs": {"max_new_tokens": 8},
                "tokenizer": {"pad_token_id": 0},
            },
            "store_logits": True,
            "data_configs": [{"dataset_name": "hotpot_qa"}],
            "output_dir": "/tmp/cache",
        }

        self.assertEqual(
            generation_cache_provenance(config),
            PROVENANCE | {"tokenizer": {"pad_token_id": 0}},
        )


if __name__ == "__main__":
    unittest.main()
