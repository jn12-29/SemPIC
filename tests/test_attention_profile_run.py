import tempfile
import unittest
from pathlib import Path

from sempic.attention_metrics.profile_identity import validate_query_pass_identity
from sempic.attention_metrics.profile_run import (
    build_partition_identity,
    group_profile_configs,
    snapshot_artifact,
)
from sempic.attention_metrics.spec import QueryPassSpec


def config(model, method, *, dataset="fixture", eval_seed=13):
    wrapper = method in {"kvpacket", "sempic_kvpacket"}
    lora = method in {"sempic", "sempic_kvpacket"}
    return {
        "model": {"model_path": str(model), "dtype": "float32", "device": "cpu"},
        "dataset": {"dataset_name": dataset, "split": "test", "seed": 42},
        "seed": eval_seed,
        "cache_comb": {"method": "no_recompute" if method == "vanilla_pic" else method, "kwargs": {}},
        "packet_wrapper": {"path": str(model / "packet.pt") if wrapper else None},
        "lora": {"path": str(model / "lora") if lora else None},
        "compress": None,
        "quantization": None,
    }


class ProfileRunContractTests(unittest.TestCase):
    def test_identity_is_query_pass_granular_and_snapshot_is_constant_size(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            for name in ("a", "b", "c"):
                (model / name).write_text(name)
            group = [
                ("full.json", config(model, "full_recompute")),
                ("vanilla.json", config(model, "vanilla_pic")),
            ]
            query_pass = QueryPassSpec(
                "gold_answer", "gold_answer_literal_tokens",
                ("attention_profile", "pic_retrieval"),
            )
            identity = build_partition_identity(
                group, query_pass=query_pass, model_id="model",
                dataset_id="fixture", max_samples=2,
            )
            validate_query_pass_identity(identity)
            self.assertEqual(identity["query_pass_id"], "gold_answer")
            self.assertEqual(identity["eval_seed"], 13)
            self.assertNotEqual(identity["eval_seed"], identity["dataset_config"]["seed"])
            self.assertEqual(identity["query_spec"], query_pass.to_dict())
            self.assertEqual(identity["artifact_snapshots"][0]["files"], [])
            self.assertEqual(snapshot_artifact("model", model)["files"], [])

    def test_grouping_requires_full_and_unique_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            with self.assertRaises(ValueError):
                group_profile_configs([("vanilla.json", config(model, "vanilla_pic"))])
            with self.assertRaises(ValueError):
                group_profile_configs([
                    ("full-a.json", config(model, "full_recompute")),
                    ("full-b.json", config(model, "full_recompute")),
                ])
            with self.assertRaises(ValueError):
                group_profile_configs([
                    ("full.json", config(model, "full_recompute", eval_seed=13)),
                    ("vanilla.json", config(model, "vanilla_pic", eval_seed=17)),
                ])


if __name__ == "__main__":
    unittest.main()
