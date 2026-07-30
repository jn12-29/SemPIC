import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sempic.attention_metrics.basis import (
    LayerAttentionBasis,
    PhysicalChunkScope,
    PhysicalLayout,
)
from sempic.attention_metrics.profile_collection import collect_query_pass_partition
from sempic.attention_metrics.profile_storage import load_partition
from sempic.attention_metrics.processing import process_partitions
from sempic.attention_metrics.spec import QueryPassSpec
from tests.test_attention_profiles import identity, method


class QueryPassCollectionTests(unittest.TestCase):
    def test_one_method_forward_fans_out_to_all_reducers_and_resumes(self):
        methods = ("full_recompute", "vanilla_pic")
        partition_identity = identity([method(key) for key in methods])
        partition_identity["query_spec"]["reducers"].append(
            "raw_attention_profile"
        )
        query_pass = QueryPassSpec.from_dict(partition_identity["query_spec"])
        calls = []
        layout = PhysicalLayout(2, (
            PhysicalChunkScope("chunk", "digest", 0, 2, 0, 2),
        ))

        def stream(method_key, entry, spec, sink):
            calls.append((entry, method_key, spec.query_pass_id))
            probabilities = torch.tensor([[[0.4, 0.6]], [[0.7, 0.3]]])
            sink(LayerAttentionBasis(
                layer_index=0,
                scaled_masked_logits=probabilities.log(),
                attention_probabilities=probabilities,
                physical_values=torch.ones((1, 2, 3)),
                query_to_kv_head=torch.tensor([0, 0]),
                keep_mask=torch.ones((1, 1, 2), dtype=torch.bool),
                layout=layout,
            ))
            return SimpleNamespace(
                layer_count=1, query_count=1, query_head_count=2, layout=layout,
                sample_id=f"sample-{entry}",
                canonical_token_digest="canonical",
                query_target_digest="target",
            )

        def identify(index, entry, spec):
            return {
                "sample_id": f"sample-{entry}",
                "canonical_token_digest": "canonical",
                "query_target_digest": "target",
                "chunks": [{
                    "chunk_id": "chunk",
                    "token_digest": "digest",
                    "token_length": 2,
                }],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = collect_query_pass_partition(
                partition_identity=partition_identity,
                query_pass=query_pass,
                entries=[0, 1],
                identify_sample=identify,
                stream_method=stream,
                layer_count=1,
                query_head_count=2,
                partition_path=root / "statistics.pt",
                work_dir=root / ".work",
            )
            artifact = load_partition(path)
            self.assertEqual(len(artifact["samples"]), 2)
            self.assertEqual(len(calls), 4)
            self.assertEqual(
                set(artifact["samples"][0]["chunks"][0]["reducer_outputs"]),
                {
                    "attention_profile",
                    "pic_retrieval",
                    "raw_attention_profile",
                },
            )
            torch.testing.assert_close(
                artifact["samples"][0]["chunks"][0]["reducer_outputs"]
                ["raw_attention_profile"]["vanilla_pic"]["raw"],
                torch.tensor([[0.55, 0.45]]),
            )
            processed = process_partitions([artifact], {
                "position_mode": "auto",
                "num_position_bins": 2,
                "edge_ratios": ["0.2"],
            })
            self.assertIn("attention_profile", processed["metric_specs"])
            self.assertIn("retrieval_nrmse", processed["metric_specs"])
            collect_query_pass_partition(
                partition_identity=partition_identity,
                query_pass=query_pass,
                entries=[0, 1],
                identify_sample=identify,
                stream_method=stream,
                layer_count=1,
                query_head_count=2,
                partition_path=path,
                work_dir=root / ".work",
            )
            self.assertEqual(len(calls), 4)
            path.unlink()
            def identify_changed_chunks(index, entry, spec):
                value = identify(index, entry, spec)
                value["chunks"][0]["chunk_id"] = "changed-boundary"
                return value

            with self.assertRaisesRegex(ValueError, "ContextBlock boundaries"):
                collect_query_pass_partition(
                    partition_identity=partition_identity,
                    query_pass=query_pass,
                    entries=[0, 1],
                    identify_sample=identify_changed_chunks,
                    stream_method=stream,
                    layer_count=1,
                    query_head_count=2,
                    partition_path=path,
                    work_dir=root / ".work",
                )


if __name__ == "__main__":
    unittest.main()
