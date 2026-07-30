import json
import tempfile
import unittest
from pathlib import Path

from plot_scripts.evidence_config_match import (
    behavioral_projection,
    compare_behavioral_configs,
    load_completed_result,
    measurement_id,
    sha256_value,
)


def example_config() -> dict:
    return {
        "model": {
            "model_path": "./models/example",
            "dtype": "bfloat16",
            "device": "cuda:0",
            "generation_kwargs": {
                "max_new_tokens": 32,
                "stop_strings": ["<|im_end|>"],
                "do_sample": False,
                "use_cache": True,
            },
        },
        "dataset": {
            "dataset_name": "biography",
            "num_samples": 100,
            "split": "test",
            "seed": 42,
            "template": "tokenizer_chat",
            "template_kwargs": {},
        },
        "cache_comb": {"method": "kvpacket", "kwargs": {}},
        "packet_wrapper": {"path": "./artifacts/packet_wrapper.pt"},
        "lora": {"path": None},
        "compress": None,
        "quantization": None,
        "seed": 42,
        "logging": {"level": "INFO"},
        "debug_dump": {"enabled": False},
        "run_suffix": "first-run",
    }


class EvidenceConfigMatchTest(unittest.TestCase):
    def test_ignores_only_declared_nonbehavioral_fields_and_normalizes_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = example_config()
            candidate = json.loads(json.dumps(expected))
            candidate["model"]["model_path"] = str(root / "models" / "example")
            candidate["packet_wrapper"]["path"] = str(
                root / "artifacts" / "packet_wrapper.pt"
            )
            candidate["logging"] = {"level": "DEBUG"}
            candidate["debug_dump"] = {"enabled": True}
            candidate["run_suffix"] = "another-run"

            comparison = compare_behavioral_configs(expected, candidate, root)

            self.assertTrue(comparison.matched)
            projection = behavioral_projection(expected, root)
            self.assertNotIn("logging", projection)
            self.assertEqual(
                projection["model"]["model_path"], str(root / "models" / "example")
            )

    def test_generation_and_artifact_differences_are_behavioral(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = example_config()
            candidate = json.loads(json.dumps(expected))
            del candidate["model"]["generation_kwargs"]["stop_strings"]
            candidate["packet_wrapper"]["path"] = "./artifacts/other.pt"

            comparison = compare_behavioral_configs(expected, candidate, directory)

            self.assertFalse(comparison.matched)
            self.assertEqual(
                {difference.field for difference in comparison.differences},
                {
                    "model.generation_kwargs.stop_strings",
                    "packet_wrapper.path",
                },
            )

    def test_unknown_config_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = example_config()
            candidate = json.loads(json.dumps(expected))
            candidate["new_runtime_behavior"] = True

            comparison = compare_behavioral_configs(expected, candidate, directory)

            self.assertEqual(comparison.status, "incompatible")
            self.assertEqual(comparison.differences[0].field, "new_runtime_behavior")

    def test_completed_result_requires_finite_f1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_result.json"
            path.write_text(
                json.dumps({"config": example_config(), "result": {"f1": "bad"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite numeric F1"):
                load_completed_result(path)

    def test_measurement_id_is_method_normalized(self):
        config_hash = sha256_value({"config": "same"})
        result_hash = sha256_value({"result": "same"})
        full_id = measurement_id(
            method_key="full_recompute",
            config_projection_sha256=config_hash,
            result_payload_sha256=result_hash,
        )
        kv_id = measurement_id(
            method_key="kvpacket",
            config_projection_sha256=config_hash,
            result_payload_sha256=result_hash,
        )
        self.assertNotEqual(full_id, kv_id)


if __name__ == "__main__":
    unittest.main()
