import copy
import unittest

import torch

from sempic.attention_metrics.processed_storage import validate_processed_metrics
from sempic.attention_metrics.processing import process_partitions
from sempic.attention_metrics.profiles import make_partition

from tests.test_attention_profiles import identity, method


METHODS = [method("full_recompute"), method("vanilla_pic")]
LAYER_COUNT = 2
HEAD_COUNT = 2
QUERY_COUNT = 2


def config(position_mode="auto"):
    return {
        "position_mode": position_mode,
        "num_position_bins": 4,
        "edge_ratios": ["0.25"],
    }


def _retrieval(nrmse, *, cosine=0.25, cosine_count=2, mass_error=0.2):
    shape = (LAYER_COUNT, HEAD_COUNT)
    return {
        "query_count": QUERY_COUNT,
        "squared_error_sum": torch.full(shape, nrmse * nrmse, dtype=torch.float32),
        "reference_energy_sum": torch.ones(shape, dtype=torch.float32),
        "cosine_distance_sum": torch.full(
            shape, cosine * cosine_count, dtype=torch.float32
        ),
        "cosine_valid_count": torch.full(
            shape, cosine_count, dtype=torch.int64
        ),
        "absolute_mass_error_sum": torch.full(
            shape, mass_error * QUERY_COUNT, dtype=torch.float32
        ),
        "full_scope_mass_sum": torch.ones(shape, dtype=torch.float32),
        "candidate_scope_mass_sum": torch.ones(shape, dtype=torch.float32),
    }


def _chunk(chunk_id, raw_error, *, length=4, nrmse=1.0):
    profile_shape = (LAYER_COUNT, length)
    retrieval_shape = (LAYER_COUNT, HEAD_COUNT)
    layouts = {
        key: {"pic_start": 1, "pic_end": length + 1, "scope_start": 1, "scope_end": length + 1}
        for key in ("full_recompute", "vanilla_pic")
    }
    return {
        "chunk_id": chunk_id,
        "token_digest": f"digest:{chunk_id}",
        "token_length": length,
        "method_layouts": layouts,
        "reducer_outputs": {
            "attention_profile": {
                "full_recompute": {
                    "raw": torch.full(profile_shape, 0.2, dtype=torch.float32),
                    "chunk_conditional": torch.full(
                        profile_shape, 1 / length, dtype=torch.float32
                    ),
                },
                "vanilla_pic": {
                    "raw_absolute_error": torch.full(
                        profile_shape, raw_error, dtype=torch.float32
                    ),
                    "chunk_conditional_absolute_error": torch.full(
                        profile_shape, raw_error / 2, dtype=torch.float32
                    ),
                },
            },
            "pic_retrieval": {
                "full_recompute": {
                    "query_count": QUERY_COUNT,
                    "reference_energy_sum": torch.ones(
                        retrieval_shape, dtype=torch.float32
                    ),
                    "scope_mass_sum": torch.ones(
                        retrieval_shape, dtype=torch.float32
                    ),
                },
                "vanilla_pic": _retrieval(nrmse),
            },
        },
    }


def _sample(index, chunks):
    return {
        "sample_index": index,
        "sample_id": f"sample:{index}",
        "canonical_token_digest": f"prompt:{index}",
        "query_target_digest": f"answer:{index}",
        "query_count": QUERY_COUNT,
        "chunks": chunks,
    }


def partition(*, ragged=False):
    second_length = 3 if ragged else 4
    return make_partition(
        partition_identity=identity(METHODS),
        layer_count=LAYER_COUNT,
        query_head_count=HEAD_COUNT,
        samples=[
            _sample(0, [
                _chunk("a", 0.1, nrmse=1.0),
                _chunk("b", 0.3, nrmse=3.0, length=second_length),
            ]),
            _sample(1, [_chunk("c", 0.6, nrmse=6.0)]),
        ],
    )


def _record(artifact, metric, view, method, **facets):
    matches = [
        record
        for record in artifact["records"]
        if record["metric_key"] == metric
        and record["view_key"] == view
        and record["method_key"] == method
        and record["facets"] == facets
    ]
    if len(matches) != 1:
        raise AssertionError(f"Expected one record, found {len(matches)}")
    return matches[0]


class AttentionProcessingTests(unittest.TestCase):
    def test_emits_strict_generic_records_without_full_retrieval_errors(self):
        artifact = process_partitions([partition()], config())
        self.assertIs(validate_processed_metrics(artifact), artifact)
        self.assertEqual(
            set(artifact),
            {
                "processing_config",
                "processing_fingerprint",
                "source_partitions",
                "metric_specs",
                "records",
            },
        )
        self.assertEqual(
            set(artifact["metric_specs"]),
            {
                "attention_profile",
                "attention_absolute_deviation",
                "retrieval_nrmse",
                "retrieval_cosine_distance",
                "attention_mass_error",
            },
        )
        retrieval = {
            "retrieval_nrmse",
            "retrieval_cosine_distance",
            "attention_mass_error",
        }
        self.assertFalse(any(
            record["metric_key"] in retrieval
            and record["method_key"] == "full_recompute"
            for record in artifact["records"]
        ))
        for metric in retrieval:
            self.assertEqual(
                {
                    record["view_key"]
                    for record in artifact["records"]
                    if record["metric_key"] == metric
                },
                {"layer_head_heatmap", "layer_curve", "global_bar"},
            )

    def test_chunks_are_equal_weight_within_sample_then_samples_define_sem(self):
        artifact = process_partitions([partition()], config())
        attention = _record(
            artifact,
            "attention_absolute_deviation",
            "global_bar",
            "vanilla_pic",
            attention_view="raw",
        )
        self.assertAlmostEqual(attention["mean"].item(), 0.4)
        self.assertAlmostEqual(attention["sem"].item(), 0.2)
        self.assertEqual(attention["count"].item(), 2)

        nrmse = _record(
            artifact, "retrieval_nrmse", "global_bar", "vanilla_pic"
        )
        # sample 0 is the equal-chunk mean (1 + 3) / 2; sample 1 is 6.
        self.assertAlmostEqual(nrmse["mean"].item(), 4.0)
        self.assertAlmostEqual(nrmse["sem"].item(), 2.0)

    def test_zero_reference_energy_and_cosine_valid_count_are_missing(self):
        data = partition()
        first = data["samples"][0]["chunks"][0]["reducer_outputs"][
            "pic_retrieval"
        ]["vanilla_pic"]
        second = data["samples"][1]["chunks"][0]["reducer_outputs"][
            "pic_retrieval"
        ]["vanilla_pic"]
        first["reference_energy_sum"][0, 0] = 0
        first["squared_error_sum"][0, 0] = 4
        first["cosine_valid_count"][0, 1] = 0
        first["cosine_distance_sum"][0, 1] = 0
        second["reference_energy_sum"][0, 0] = 0
        second["squared_error_sum"][0, 0] = 0
        second["cosine_valid_count"][0, 1] = 0
        second["cosine_distance_sum"][0, 1] = 0

        artifact = process_partitions([data], config())
        nrmse = _record(
            artifact, "retrieval_nrmse", "layer_head_heatmap", "vanilla_pic"
        )
        cosine = _record(
            artifact,
            "retrieval_cosine_distance",
            "layer_head_heatmap",
            "vanilla_pic",
        )
        # Sample 0's other chunk remains valid, so it contributes one equal-chunk value.
        self.assertEqual(nrmse["count"][0, 0].item(), 1)
        self.assertAlmostEqual(nrmse["mean"][0, 0].item(), 3.0)
        self.assertEqual(cosine["count"][0, 1].item(), 1)
        self.assertAlmostEqual(cosine["mean"][0, 1].item(), 0.25)

        only_invalid = copy.deepcopy(data)
        for sample in only_invalid["samples"]:
            for chunk in sample["chunks"]:
                stats = chunk["reducer_outputs"]["pic_retrieval"]["vanilla_pic"]
                stats["reference_energy_sum"][0, 0] = 0
                stats["cosine_valid_count"][0, 1] = 0
                stats["cosine_distance_sum"][0, 1] = 0
        invalid_artifact = process_partitions([only_invalid], config())
        invalid_nrmse = _record(
            invalid_artifact, "retrieval_nrmse", "layer_head_heatmap", "vanilla_pic"
        )
        invalid_cosine = _record(
            invalid_artifact,
            "retrieval_cosine_distance",
            "layer_head_heatmap",
            "vanilla_pic",
        )
        self.assertEqual(invalid_nrmse["count"][0, 0].item(), 0)
        self.assertTrue(torch.isnan(invalid_nrmse["mean"][0, 0]))
        self.assertEqual(invalid_cosine["count"][0, 1].item(), 0)
        self.assertTrue(torch.isnan(invalid_cosine["mean"][0, 1]))

    def test_cosine_and_mass_use_their_query_denominators(self):
        data = partition()
        for sample in data["samples"]:
            for chunk in sample["chunks"]:
                stats = chunk["reducer_outputs"]["pic_retrieval"]["vanilla_pic"]
                stats["cosine_distance_sum"].fill_(1.5)
                stats["cosine_valid_count"].fill_(2)
                stats["absolute_mass_error_sum"].fill_(0.6)
        artifact = process_partitions([data], config())
        cosine = _record(
            artifact, "retrieval_cosine_distance", "global_bar", "vanilla_pic"
        )
        mass = _record(
            artifact, "attention_mass_error", "global_bar", "vanilla_pic"
        )
        self.assertAlmostEqual(cosine["mean"].item(), 0.75)
        self.assertAlmostEqual(mass["mean"].item(), 0.3)

    def test_position_modes_and_directed_regions_remain_distinct(self):
        equal = partition()
        values = torch.tensor(
            [[0.1, 0.2, 0.4, 0.8], [0.1, 0.2, 0.4, 0.8]],
            dtype=torch.float32,
        )
        equal["samples"] = [equal["samples"][0]]
        equal["samples"][0]["chunks"] = [equal["samples"][0]["chunks"][0]]
        equal["samples"][0]["chunks"][0]["reducer_outputs"]["attention_profile"][
            "vanilla_pic"
        ]["raw_absolute_error"] = values
        artifact = process_partitions([equal], config())
        heatmap = _record(
            artifact,
            "attention_absolute_deviation",
            "layer_position_heatmap",
            "vanilla_pic",
            attention_view="raw",
            position_mode="absolute",
        )
        self.assertEqual(heatmap["coordinates"]["position_bin"], [0, 1, 2, 3])
        expected = {"prefix": 0.1, "interior": 0.3, "suffix": 0.8}
        for region, value in expected.items():
            record = _record(
                artifact,
                "attention_absolute_deviation",
                "global_bar",
                "vanilla_pic",
                attention_view="raw",
                edge_ratio="0.25",
                region=region,
            )
            self.assertAlmostEqual(record["mean"].item(), value)

        normalized = process_partitions([partition(ragged=True)], config())
        normalized_heatmap = _record(
            normalized,
            "attention_absolute_deviation",
            "layer_position_heatmap",
            "vanilla_pic",
            attention_view="raw",
            position_mode="normalized",
        )
        self.assertEqual(len(normalized_heatmap["coordinates"]["position_bin"]), 4)
        with self.assertRaisesRegex(ValueError, "equal chunk lengths"):
            process_partitions([partition(ragged=True)], config("absolute"))

    def test_raw_and_conditional_attention_are_both_materialized(self):
        artifact = process_partitions([partition()], config())
        keys = {
            (record["metric_key"], record["method_key"], record["facets"].get("attention_view"))
            for record in artifact["records"]
            if record["view_key"] == "layer_position_heatmap"
        }
        self.assertIn(("attention_profile", "full_recompute", "raw"), keys)
        self.assertIn(
            ("attention_profile", "full_recompute", "chunk_conditional"), keys
        )
        self.assertIn(
            ("attention_absolute_deviation", "vanilla_pic", "raw"), keys
        )
        self.assertIn(
            (
                "attention_absolute_deviation",
                "vanilla_pic",
                "chunk_conditional",
            ),
            keys,
        )


if __name__ == "__main__":
    unittest.main()
