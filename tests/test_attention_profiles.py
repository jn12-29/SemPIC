import copy
import tempfile
import unittest
from pathlib import Path

import torch

from sempic.attention_metrics.profile_identity import (
    normalize_method_key,
    runtime_fingerprint,
)
from sempic.attention_metrics.profile_storage import (
    load_checkpoint,
    load_partition,
    make_checkpoint,
    save_checkpoint,
    save_partition,
)
from sempic.attention_metrics.profiles import (
    make_partition,
    query_pass_partition_fingerprint,
    validate_partition,
)


MODEL_CONFIG = {"model_path": "/fixture/model"}
LAYER_COUNT = 2
QUERY_HEAD_COUNT = 3


def method(key: str) -> dict[str, object]:
    resolved_method_config = {
        "cache_comb": {"method": key, "kwargs": {}},
        "packet_wrapper": {
            "path": "/fixture/packet_wrapper.pt"
            if key in ("kvpacket", "sempic_kvpacket")
            else None
        },
        "lora": {
            "path": "/fixture/lora"
            if key in ("sempic", "sempic_kvpacket")
            else None
        },
        "compress": None,
        "quantization": None,
    }
    return {
        "method_key": key,
        "runtime_fingerprint": runtime_fingerprint(
            MODEL_CONFIG, resolved_method_config
        ),
        "resolved_method_config": resolved_method_config,
        "source_config": f"{key}.json",
    }


def identity(methods: list[dict[str, object]]) -> dict[str, object]:
    configured_paths = {MODEL_CONFIG["model_path"]}
    for method_record in methods:
        resolved = method_record["resolved_method_config"]
        for artifact_name in ("packet_wrapper", "lora"):
            path = resolved[artifact_name]["path"]
            if path is not None:
                configured_paths.add(path)
    return {
        "model_config": MODEL_CONFIG,
        "dataset_config": {"dataset_name": "fixture"},
        "eval_seed": 17,
        "artifact_snapshots": [
            {
                "artifact_key": f"artifact-{index}",
                "canonical_path": path,
                "files": [],
            }
            for index, path in enumerate(sorted(configured_paths))
        ],
        "model_id": "model",
        "dataset_id": "fixture",
        "query_pass_id": "literal_answer",
        "query_spec": {
            "kind": "gold_answer_literal_tokens",
            "query_pass_id": "literal_answer",
            "reducers": ["attention_profile", "pic_retrieval"],
        },
        "methods": methods,
        "max_samples": 2,
        "dataset_iteration": {"seed": 42, "split": "test"},
    }


def _full_retrieval(query_count: int) -> dict[str, object]:
    shape = (LAYER_COUNT, QUERY_HEAD_COUNT)
    return {
        "query_count": query_count,
        "reference_energy_sum": torch.full(shape, 2.0),
        "scope_mass_sum": torch.full(shape, 1.0),
    }


def _candidate_retrieval(query_count: int) -> dict[str, object]:
    shape = (LAYER_COUNT, QUERY_HEAD_COUNT)
    return {
        "query_count": query_count,
        "squared_error_sum": torch.full(shape, 0.5),
        "reference_energy_sum": torch.full(shape, 2.0),
        "cosine_distance_sum": torch.full(shape, 0.25),
        "cosine_valid_count": torch.full(shape, query_count, dtype=torch.int64),
        "absolute_mass_error_sum": torch.full(shape, 0.1),
        "full_scope_mass_sum": torch.full(shape, 1.0),
        "candidate_scope_mass_sum": torch.full(shape, 0.9),
    }


def sample(index: int, methods: tuple[str, ...]) -> dict[str, object]:
    query_count = 2
    token_length = 4
    profile_shape = (LAYER_COUNT, token_length)
    layouts = {}
    attention = {}
    retrieval = {}
    for key in methods:
        if key in ("kvpacket", "sempic_kvpacket"):
            layouts[key] = {
                "pic_start": 2,
                "pic_end": 6,
                "scope_start": 1,
                "scope_end": 7,
            }
        else:
            layouts[key] = {
                "pic_start": 1,
                "pic_end": 5,
                "scope_start": 1,
                "scope_end": 5,
            }
        if key == "full_recompute":
            attention[key] = {
                "raw": torch.full(profile_shape, 0.2),
                "chunk_conditional": torch.full(profile_shape, 0.25),
            }
            retrieval[key] = _full_retrieval(query_count)
        else:
            attention[key] = {
                "raw_absolute_error": torch.full(profile_shape, 0.01),
                "chunk_conditional_absolute_error": torch.full(
                    profile_shape, 0.02
                ),
            }
            retrieval[key] = _candidate_retrieval(query_count)
    return {
        "sample_index": index,
        "sample_id": f"sample:{index}",
        "canonical_token_digest": f"prompt-{index}",
        "query_target_digest": f"answer-{index}",
        "query_count": query_count,
        "chunks": [
            {
                "chunk_id": "part-0:0:4",
                "token_digest": "chunk",
                "token_length": token_length,
                "method_layouts": layouts,
                "reducer_outputs": {
                    "attention_profile": attention,
                    "pic_retrieval": retrieval,
                },
            }
        ],
    }


class AttentionQueryPassSchemaTests(unittest.TestCase):
    def setUp(self):
        self.method_records = [
            method("full_recompute"),
            method("vanilla_pic"),
            method("kvpacket"),
            method("sempic"),
        ]
        self.identity = identity(self.method_records)
        self.methods = tuple(record["method_key"] for record in self.method_records)
        self.artifact = make_partition(
            partition_identity=self.identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            samples=[sample(0, self.methods)],
        )

    def test_partition_and_complete_sample_checkpoint_round_trip(self):
        checkpoint = make_checkpoint(
            partition_identity=self.identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            sample=self.artifact["samples"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = load_partition(save_partition(root / "statistics.pt", self.artifact))
            loaded_checkpoint = load_checkpoint(
                save_checkpoint(
                    root / "sample_000000.pt",
                    checkpoint,
                    partition_identity=self.identity,
                    layer_count=LAYER_COUNT,
                    query_head_count=QUERY_HEAD_COUNT,
                    expected_index=0,
                ),
                partition_identity=self.identity,
                layer_count=LAYER_COUNT,
                query_head_count=QUERY_HEAD_COUNT,
                expected_index=0,
            )
        self.assertEqual(
            loaded["partition_identity"]["query_pass_id"], "literal_answer"
        )
        self.assertEqual(
            tuple(
                loaded_checkpoint["sample"]["chunks"][0]["reducer_outputs"][
                    "attention_profile"
                ]
            ),
            self.methods,
        )
        torch.testing.assert_close(
            loaded_checkpoint["sample"]["chunks"][0]["reducer_outputs"][
                "pic_retrieval"
            ]["sempic"]["squared_error_sum"],
            torch.full((LAYER_COUNT, QUERY_HEAD_COUNT), 0.5),
        )

    def test_raw_attention_profile_round_trip_and_strict_validation(self):
        raw_identity = copy.deepcopy(self.identity)
        raw_identity["query_spec"]["reducers"].append("raw_attention_profile")
        raw_sample = sample(0, self.methods)
        raw_sample["chunks"][0]["reducer_outputs"]["raw_attention_profile"] = {
            key: {"raw": torch.full((LAYER_COUNT, 4), 0.1)}
            for key in self.methods
        }
        artifact = make_partition(
            partition_identity=raw_identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            samples=[raw_sample],
        )
        self.assertEqual(artifact["schema_version"], 1)
        checkpoint = make_checkpoint(
            partition_identity=raw_identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            sample=raw_sample,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loaded = load_partition(save_partition(root / "statistics.pt", artifact))
            loaded_checkpoint = load_checkpoint(
                save_checkpoint(
                    root / "sample_000000.pt",
                    checkpoint,
                    partition_identity=raw_identity,
                    layer_count=LAYER_COUNT,
                    query_head_count=QUERY_HEAD_COUNT,
                    expected_index=0,
                ),
                partition_identity=raw_identity,
                layer_count=LAYER_COUNT,
                query_head_count=QUERY_HEAD_COUNT,
                expected_index=0,
            )
        for value in (loaded, loaded_checkpoint):
            sample_value = (
                value["sample"] if "sample" in value else value["samples"][0]
            )
            torch.testing.assert_close(
                sample_value["chunks"][0]["reducer_outputs"]
                ["raw_attention_profile"]["sempic"]["raw"],
                torch.full((LAYER_COUNT, 4), 0.1),
            )

        mutations = (
            lambda tensor: tensor.to(torch.float64),
            lambda tensor: tensor[:, :3],
            lambda tensor: tensor.fill_(-0.1),
            lambda tensor: tensor.fill_(1.1),
            lambda tensor: tensor.fill_(torch.nan),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(artifact)
                record = invalid["samples"][0]["chunks"][0]["reducer_outputs"][
                    "raw_attention_profile"
                ]["sempic"]
                record["raw"] = mutation(record["raw"])
                with self.assertRaises(ValueError):
                    validate_partition(invalid)

        for mutation in (
            lambda records: records.pop("sempic"),
            lambda records: records["sempic"].update(unknown=torch.zeros(1)),
        ):
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(artifact)
                records = invalid["samples"][0]["chunks"][0][
                    "reducer_outputs"
                ]["raw_attention_profile"]
                mutation(records)
                with self.assertRaises(ValueError):
                    validate_partition(invalid)

    def test_intermediate_basis_fields_are_rejected(self):
        for field in ("logits", "values", "retrieval_vector"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.artifact)
                invalid["samples"][0]["chunks"][0]["reducer_outputs"][field] = (
                    torch.zeros(1)
                )
                with self.assertRaisesRegex(ValueError, "unknown fields"):
                    validate_partition(invalid)

    def test_all_methods_and_both_reducers_are_required(self):
        mutations = (
            lambda chunk: chunk["reducer_outputs"].pop("pic_retrieval"),
            lambda chunk: chunk["reducer_outputs"]["attention_profile"].pop(
                "sempic"
            ),
            lambda chunk: chunk["reducer_outputs"]["pic_retrieval"].pop(
                "vanilla_pic"
            ),
            lambda chunk: chunk["method_layouts"].pop("kvpacket"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(self.artifact)
                mutation(invalid["samples"][0]["chunks"][0])
                with self.assertRaises(ValueError):
                    validate_partition(invalid)

    def test_tensor_dtype_shape_device_and_count_are_strict(self):
        mutations = (
            lambda chunk: chunk["reducer_outputs"]["attention_profile"][
                "full_recompute"
            ].update(raw=torch.zeros(LAYER_COUNT, 4, dtype=torch.float64)),
            lambda chunk: chunk["reducer_outputs"]["pic_retrieval"][
                "sempic"
            ].update(squared_error_sum=torch.zeros(LAYER_COUNT, 1)),
            lambda chunk: chunk["reducer_outputs"]["pic_retrieval"][
                "sempic"
            ].update(cosine_valid_count=torch.zeros(LAYER_COUNT, QUERY_HEAD_COUNT)),
            lambda chunk: chunk["reducer_outputs"]["pic_retrieval"][
                "sempic"
            ].update(
                cosine_valid_count=torch.full(
                    (LAYER_COUNT, QUERY_HEAD_COUNT), 3, dtype=torch.int64
                )
            ),
            lambda chunk: chunk["reducer_outputs"]["pic_retrieval"][
                "sempic"
            ].update(query_count=True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(self.artifact)
                mutation(invalid["samples"][0]["chunks"][0])
                with self.assertRaises(ValueError):
                    validate_partition(invalid)

    def test_tensor_subclasses_and_non_finite_statistics_are_rejected(self):
        class UnsafeTensor(torch.Tensor):
            pass

        invalid = copy.deepcopy(self.artifact)
        invalid["samples"][0]["chunks"][0]["reducer_outputs"][
            "attention_profile"
        ]["full_recompute"]["raw"] = torch.zeros(LAYER_COUNT, 4).as_subclass(
            UnsafeTensor
        )
        with self.assertRaisesRegex(ValueError, "CPU float32"):
            validate_partition(invalid)

        invalid = copy.deepcopy(self.artifact)
        invalid["samples"][0]["chunks"][0]["reducer_outputs"][
            "pic_retrieval"
        ]["sempic"]["squared_error_sum"][0, 0] = torch.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_partition(invalid)

    def test_layout_scope_contract_is_method_specific(self):
        invalid = copy.deepcopy(self.artifact)
        invalid["samples"][0]["chunks"][0]["method_layouts"]["sempic"][
            "scope_start"
        ] = 0
        with self.assertRaisesRegex(ValueError, "Wrapper-free"):
            validate_partition(invalid)

        invalid = copy.deepcopy(self.artifact)
        invalid["samples"][0]["chunks"][0]["method_layouts"]["kvpacket"].update(
            scope_start=3
        )
        with self.assertRaisesRegex(ValueError, "contain"):
            validate_partition(invalid)

        invalid = copy.deepcopy(self.artifact)
        invalid["samples"][0]["chunks"][0]["method_layouts"]["kvpacket"].update(
            scope_start=2
        )
        with self.assertRaisesRegex(ValueError, "head and tail"):
            validate_partition(invalid)

    def test_identity_and_fingerprint_tampering_are_rejected(self):
        for mutation in (
            lambda value: value.update(partition_fingerprint="wrong"),
            lambda value: value["partition_identity"].update(model_id="wrong"),
            lambda value: value["partition_identity"].pop("query_spec"),
            lambda value: value["partition_identity"].update(eval_seed=-1),
            lambda value: value["partition_identity"].update(
                query_pass_id="shifted_prediction"
            ),
        ):
            with self.subTest(mutation=mutation):
                invalid = copy.deepcopy(self.artifact)
                mutation(invalid)
                with self.assertRaises(ValueError):
                    validate_partition(invalid)

    def test_legacy_identity_without_eval_seed_remains_readable(self):
        legacy = copy.deepcopy(self.artifact)
        legacy_identity = legacy["partition_identity"]
        legacy_identity.pop("eval_seed")
        legacy["partition_fingerprint"] = query_pass_partition_fingerprint(
            legacy_identity,
            legacy["layer_count"],
            legacy["query_head_count"],
        )

        validated = validate_partition(legacy)

        self.assertNotIn("eval_seed", validated["partition_identity"])

    def test_checkpoint_filename_and_sample_index_must_match(self):
        checkpoint = make_checkpoint(
            partition_identity=self.identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            sample=self.artifact["samples"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "filename"):
                save_checkpoint(
                    Path(directory) / "sample_000001.pt",
                    checkpoint,
                    partition_identity=self.identity,
                    layer_count=LAYER_COUNT,
                    query_head_count=QUERY_HEAD_COUNT,
                    expected_index=0,
                )

    def test_failed_checkpoint_validation_does_not_replace_complete_sample(self):
        checkpoint = make_checkpoint(
            partition_identity=self.identity,
            layer_count=LAYER_COUNT,
            query_head_count=QUERY_HEAD_COUNT,
            sample=self.artifact["samples"][0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample_000000.pt"
            save_checkpoint(
                path,
                checkpoint,
                partition_identity=self.identity,
                layer_count=LAYER_COUNT,
                query_head_count=QUERY_HEAD_COUNT,
                expected_index=0,
            )
            invalid = copy.deepcopy(checkpoint)
            invalid["sample"]["chunks"][0]["reducer_outputs"].pop("pic_retrieval")
            with self.assertRaises(ValueError):
                save_checkpoint(
                    path,
                    invalid,
                    partition_identity=self.identity,
                    layer_count=LAYER_COUNT,
                    query_head_count=QUERY_HEAD_COUNT,
                    expected_index=0,
                )
            loaded = load_checkpoint(
                path,
                partition_identity=self.identity,
                layer_count=LAYER_COUNT,
                query_head_count=QUERY_HEAD_COUNT,
                expected_index=0,
            )
        self.assertIn(
            "pic_retrieval", loaded["sample"]["chunks"][0]["reducer_outputs"]
        )

    def test_method_alias_is_input_only(self):
        self.assertEqual(normalize_method_key("no_recompute"), "vanilla_pic")
        invalid = copy.deepcopy(self.artifact)
        invalid["partition_identity"]["methods"][1]["method_key"] = "no_recompute"
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_partition(invalid)


if __name__ == "__main__":
    unittest.main()
