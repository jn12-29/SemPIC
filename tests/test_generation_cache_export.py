import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.export_generation_cache_texts import (
    export_cache,
    load_artifact_config,
    max_new_tokens_from_config,
)
from sempic.prompt import TokenSpan, TokenizedPrompt
from sempic.utils.generation_cache import StreamingGenerationCacheWriter


class FakeTokenizer:
    def decode(self, token_ids, **kwargs):
        del kwargs
        return ",".join(str(int(token_id)) for token_id in token_ids)


class GenerationCacheExportTests(unittest.TestCase):
    def test_export_uses_semantic_keys_and_artifact_provenance(self):
        cache_key = hashlib.sha256(b"semantic-key").hexdigest()
        generation = {
            "sequences": [torch.tensor([7, 8], dtype=torch.long)],
            "logits": [],
            "text": ["answer"],
        }
        prompt = TokenizedPrompt(
            input_ids=torch.tensor([1, 2], dtype=torch.long),
            parts=(TokenSpan(kind="inline", start=0, end=2),),
        )
        config = {
            "model": {
                "model_path": "teacher",
                "tokenizer_path": "tokenizer",
                "dtype": "bfloat16",
                "device": "cuda:0",
                "generation_kwargs": {"max_new_tokens": 2},
            },
            "data_configs": [{
                "dataset_name": "biography",
                "num_samples": 1,
                "num_data_strs": 2,
                "num_shots": 0,
                "subset": "10k",
                "split": "train",
                "seed": 0,
                "data_kwargs": {},
                "template": "tokenizer_chat",
                "template_kwargs": {},
            }],
            "store_logits": True,
            "gen_batch_size": 1,
            "cache_device": "cpu",
            "seed": 42,
            "output_dir": "artifact",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = Path(temp_dir) / "artifact"
            artifact_dir.mkdir()
            with StreamingGenerationCacheWriter(
                artifact_dir,
                provenance={
                    "model_path": "teacher",
                    "tokenizer_path": "tokenizer",
                    "dtype": "bfloat16",
                    "tokenizer": {"pad_token_id": 0, "eos_token_id": 2},
                    "store_logits": False,
                },
            ) as writer:
                writer.add(cache_key, generation)
                writer.finalize()
            (artifact_dir / "resolved_config.json").write_text(json.dumps(config))

            resolved_config = load_artifact_config(artifact_dir)
            max_new_tokens = max_new_tokens_from_config(resolved_config)
            summary = export_cache(
                artifact_dir,
                resolved_config,
                [(cache_key, prompt), (cache_key, prompt)],
                max_new_tokens,
                FakeTokenizer(),
                Path(temp_dir) / "exports",
                "teacher",
            )

            self.assertEqual(max_new_tokens, 2)
            self.assertEqual(
                resolved_config["model"]["tokenizer_path"], "tokenizer"
            )
            self.assertEqual(summary["mapping_mode"], "semantic-sample-v2")
            records = [
                json.loads(line)
                for line in Path(summary["all_texts"]).read_text().splitlines()
            ]
            self.assertEqual(records[0]["cache_key"], cache_key)
            self.assertTrue(records[0]["reached_max_length"])

    def test_max_new_tokens_must_be_a_positive_integer(self):
        config = {
            "model": {"generation_kwargs": {"max_new_tokens": True}},
        }
        with self.assertRaisesRegex(ValueError, "positive integer"):
            max_new_tokens_from_config(config)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
