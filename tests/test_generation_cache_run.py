import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import GenerationConfig

import run_generation_cache
from sempic.utils.generation_cache import load_generation_cache
from sempic.utils.generation_cache_run import (
    _rename_directory_noreplace,
    generate_cache_artifact,
    load_generation_cache_config,
)


def config_dict(output_dir: Path, **overrides):
    config = {
        "model": {
            "model_path": "teacher",
            "generation_kwargs": {"max_new_tokens": 3},
        },
        "data_configs": [
            {
                "dataset_name": "biography",
                "num_samples": 1,
                "num_data_strs": 1,
                "num_shots": 0,
                "subset": "1k",
            }
        ],
        "store_logits": True,
        "output_dir": str(output_dir),
    }
    config.update(overrides)
    return config


class GenerationCacheConfigTests(unittest.TestCase):
    def test_defaults_are_normalized(self):
        config = load_generation_cache_config(config_dict(Path("artifact")))

        self.assertEqual(config["model"]["tokenizer_path"], "teacher")
        self.assertEqual(config["model"]["dtype"], "bfloat16")
        self.assertEqual(config["model"]["device"], "cuda:0")
        self.assertEqual(config["gen_batch_size"], 1)
        self.assertEqual(config["cache_device"], "cpu")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["data_configs"][0]["split"], "train")
        self.assertEqual(config["data_configs"][0]["seed"], 42)
        self.assertEqual(config["data_configs"][0]["template"], "default")

    def test_rejects_invalid_owned_fields(self):
        cases = (
            ({"store_logits": "true"}, "store_logits"),
            ({"gen_batch_size": 0}, "gen_batch_size"),
            ({"cache_device": "not a device"}, "cache_device"),
            ({"data_configs": []}, "data_configs"),
            ({"seed": True}, "seed"),
        )
        for override, message in cases:
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    load_generation_cache_config(config_dict(Path("artifact"), **override))

    def test_cli_accepts_exactly_one_config(self):
        with self.assertRaises(SystemExit):
            run_generation_cache.parse_args([])
        with self.assertRaises(SystemExit):
            run_generation_cache.parse_args(["one.json", "two.json"])
        self.assertEqual(
            run_generation_cache.parse_args(["one.json"]).config_file,
            Path("one.json"),
        )


class GenerationCacheArtifactTests(unittest.TestCase):
    def test_atomic_publish_does_not_replace_existing_empty_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "cache.pt").write_text("source")

            with self.assertRaises(FileExistsError):
                _rename_directory_noreplace(source, destination)

            self.assertTrue(source.is_dir())
            self.assertEqual(list(destination.iterdir()), [])

    def _run_patches(self, *, store_logits=True):
        tokenizer = SimpleNamespace(
            padding_side="right",
            pad_token_id=7,
            eos_token_id=9,
        )
        model = SimpleNamespace(
            generation_config=GenerationConfig(
                max_new_tokens=5,
                eos_token_id=9,
            ),
            eval=mock.Mock(),
        )

        def build_cache(**kwargs):
            self.assertIs(kwargs["store_logits"], store_logits)
            kwargs["generation_sink"](SAMPLE_KEY, {
                "sequences": [torch.tensor([1])],
                "logits": [torch.zeros(1, 2)] if store_logits else [],
                "text": ["x"],
            })
            return mock.Mock(), True

        return (
            mock.patch(
                "sempic.utils.generation_cache_run._load_resources",
                return_value=(model, tokenizer),
            ),
            mock.patch(
                "sempic.utils.generation_cache_run._load_samples",
                return_value=[mock.sentinel.sample],
            ),
            mock.patch(
                "sempic.utils.generation_cache_run.build_generation_cache",
                side_effect=build_cache,
            ),
        )

    def test_success_publishes_exact_three_files_and_resolved_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifact"
            config = load_generation_cache_config(config_dict(output_dir))
            patches = self._run_patches()

            with patches[0], patches[1], patches[2] as build:
                published = generate_cache_artifact(config)

            self.assertEqual(published, output_dir.absolute())
            self.assertEqual(
                sorted(path.name for path in published.iterdir()),
                ["cache.safetensors", "manifest.json", "resolved_config.json"],
            )
            cache = load_generation_cache(published)
            self.assertEqual(cache.keys(), (SAMPLE_KEY,))
            self.assertEqual(cache.get(SAMPLE_KEY)["text"], ["x"])
            resolved = json.loads((published / "resolved_config.json").read_text())
            self.assertEqual(resolved["output_dir"], str(output_dir.absolute()))
            self.assertEqual(
                resolved["cache_path"], str(output_dir.absolute())
            )
            self.assertEqual(resolved["model"]["tokenizer"]["pad_token_id"], 7)
            self.assertEqual(resolved["model"]["tokenizer"]["padding_side"], "right")
            self.assertEqual(resolved["model"]["generation_kwargs"]["max_new_tokens"], 3)
            self.assertEqual(build.call_args.kwargs["batch_size"], 1)

    def test_store_logits_false_is_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifact"
            config = load_generation_cache_config(
                config_dict(output_dir, store_logits=False)
            )
            patches = self._run_patches(store_logits=False)
            with patches[0], patches[1], patches[2]:
                generate_cache_artifact(config)

    def test_existing_target_fails_before_resources_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifact"
            output_dir.mkdir()
            config = load_generation_cache_config(config_dict(output_dir))
            with mock.patch(
                "sempic.utils.generation_cache_run._load_resources"
            ) as load_resources:
                with self.assertRaises(FileExistsError):
                    generate_cache_artifact(config)
            load_resources.assert_not_called()

    def test_dangling_target_symlink_fails_before_resources_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifact"
            os.symlink(Path(directory) / "missing", output_dir)
            config = load_generation_cache_config(config_dict(output_dir))
            with mock.patch(
                "sempic.utils.generation_cache_run._load_resources"
            ) as load_resources:
                with self.assertRaises(FileExistsError):
                    generate_cache_artifact(config)
            load_resources.assert_not_called()

    def test_failure_removes_only_private_temporary_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "artifact"
            unrelated = root / ".artifact.unrelated.tmp"
            unrelated.mkdir()
            config = load_generation_cache_config(config_dict(output_dir))
            with mock.patch(
                "sempic.utils.generation_cache_run._load_resources",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    generate_cache_artifact(config)

            self.assertFalse(output_dir.exists())
            self.assertTrue(unrelated.is_dir())
            self.assertEqual(
                [path for path in root.iterdir() if path.name != unrelated.name],
                [],
            )


if __name__ == "__main__":
    unittest.main()
SAMPLE_KEY = hashlib.sha256(b"sample").hexdigest()
