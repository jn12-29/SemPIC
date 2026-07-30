import math
import unittest

from plot_scripts.build_multimodel_paper_evidence import (
    _f1_point,
    _parse_assignments,
    _ratios_from_means,
)
from plot_scripts.multimodel_paper_data import (
    DATASET_ORDER,
    MODEL_ORDER,
    SCHEMA_NAME,
    validate_plot_data,
)


def _profile() -> list[dict]:
    return [
        {
            "start": index / 20,
            "end": (index + 1) / 20,
            "mean": 2.0 if index < 2 else 1.0,
            "sem": 0.001,
            "count": 100,
        }
        for index in range(20)
    ]


def _bundle() -> dict:
    sources = [
        {"source_id": "source", "path": "/tmp/source", "sha256": "a" * 64},
        {
            "source_id": "f1_authority",
            "path": "/tmp/f1.csv",
            "sha256": "b" * 64,
        },
    ]
    models = []
    for model_id in MODEL_ORDER:
        points = []
        for dataset_id in DATASET_ORDER:
            points.append(
                {
                    "dataset_id": dataset_id,
                    "f1": {
                        "full": 0.9,
                        "no_cache": 0.1,
                        "no_recompute": 0.2,
                        "kvpacket": 0.55,
                        "sempic": 0.83,
                        "joint": 0.84,
                    },
                    "kv_recovery": 0.5,
                    "sempic_recovery": 0.9,
                    "f1_change": 0.63,
                    "kv_pre_ratio": 0.1,
                    "kv_interior_ratio": 0.9,
                    "kv_rint": 0.9,
                    "sempic_rint": 0.7,
                    "sink_profile": _profile(),
                    "sink_ratio": 2.0,
                    "source_ids": {
                        "f1": "f1_authority",
                        "boundary_attention": "source",
                        "interior_attention": "source",
                        "sink_attention": "source",
                    },
                }
            )
        models.append({"model_id": model_id, "display_name": model_id, "points": points})
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": 1,
        "sources": sources,
        "models": models,
    }


class MultimodelPaperEvidenceTest(unittest.TestCase):
    def test_contract_accepts_complete_fixed_matrix(self):
        bundle = _bundle()
        self.assertIs(validate_plot_data(bundle), bundle)

    def test_contract_rejects_noninteger_schema_version(self):
        for value in (True, 1.0):
            with self.subTest(value=value):
                bundle = _bundle()
                bundle["schema_version"] = value
                with self.assertRaisesRegex(ValueError, "version 1"):
                    validate_plot_data(bundle)

    def test_contract_rejects_zero_based_or_noncontiguous_profile_bins(self):
        bundle = _bundle()
        bundle["models"][0]["points"][0]["sink_profile"][1]["start"] = 0.06
        with self.assertRaisesRegex(ValueError, "Invalid sink profile bin"):
            validate_plot_data(bundle)

    def test_contract_requires_boundary_and_validation_interior_ratio_identity(self):
        bundle = _bundle()
        bundle["models"][0]["points"][0]["kv_interior_ratio"] = 0.8
        with self.assertRaisesRegex(ValueError, "must be identical"):
            validate_plot_data(bundle)

    def test_contract_recomputes_visible_f1_derivatives(self):
        for field in ("kv_recovery", "sempic_recovery", "f1_change"):
            with self.subTest(field=field):
                bundle = _bundle()
                bundle["models"][0]["points"][0][field] += 0.1
                with self.assertRaisesRegex(ValueError, "does not recompute"):
                    validate_plot_data(bundle)

    def test_contract_requires_full100_profiles_and_f1_authority_source(self):
        bundle = _bundle()
        bundle["models"][0]["points"][0]["sink_profile"][0]["count"] = 20
        with self.assertRaisesRegex(ValueError, "Invalid sink profile bin"):
            validate_plot_data(bundle)

        bundle = _bundle()
        bundle["models"][0]["points"][0]["source_ids"]["f1"] = "source"
        with self.assertRaisesRegex(ValueError, "f1_authority"):
            validate_plot_data(bundle)

    def test_contract_recomputes_sink_ratio_from_profile(self):
        bundle = _bundle()
        bundle["models"][0]["points"][0]["sink_ratio"] = 1.0
        with self.assertRaisesRegex(ValueError, "does not recompute"):
            validate_plot_data(bundle)

    def test_assignment_parser_requires_all_datasets_once(self):
        values = [f"{dataset}=/tmp/{dataset}.pt" for dataset in DATASET_ORDER]
        parsed = _parse_assignments(values, "--partition")
        self.assertEqual(tuple(parsed), DATASET_ORDER)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _parse_assignments(values + [values[0]], "--partition")

    def test_attention_ratios_use_vanilla_as_denominator(self):
        means = {}
        for dataset in DATASET_ORDER:
            means[(dataset, "vanilla_pic", "prefix")] = 4.0
            means[(dataset, "vanilla_pic", "interior")] = 2.0
            means[(dataset, "kvpacket", "prefix")] = 1.0
            means[(dataset, "kvpacket", "interior")] = 1.5
            means[(dataset, "sempic", "interior")] = 1.0
        ratios = _ratios_from_means(means, "test")
        self.assertEqual(ratios["biography"]["kv_pre_ratio"], 0.25)
        self.assertEqual(ratios["biography"]["kv_rint"], 0.75)
        self.assertEqual(ratios["biography"]["sempic_rint"], 0.5)

    def test_recovery_and_f1_change_are_derived_from_authority(self):
        class Entry:
            values = {
                "full_f1": 0.9,
                "no_cache_f1": 0.1,
                "no_recompute_f1": 0.2,
                "kvpacket_f1": 0.55,
                "sempic_f1": 0.83,
                "joint_f1": 0.84,
            }

        point = _f1_point(Entry())
        self.assertTrue(math.isclose(point["kv_recovery"], 0.5))
        self.assertTrue(math.isclose(point["sempic_recovery"], 0.9))
        self.assertTrue(math.isclose(point["f1_change"], 0.63))


if __name__ == "__main__":
    unittest.main()
