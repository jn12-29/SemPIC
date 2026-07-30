import json
import tempfile
import unittest
from pathlib import Path

from sempic.attention_metrics.spec import AttentionAnalysisConfig, QueryPassSpec


class QueryPassSpecTests(unittest.TestCase):
    def test_paper_config_has_three_independent_query_passes(self):
        config = AttentionAnalysisConfig.from_file(
            "attention_config/paper_attention.json"
        )
        self.assertEqual(
            [item.query_pass_id for item in config.query_passes],
            ["terminal_query", "gold_answer", "shifted_prediction"],
        )
        self.assertTrue(all(
            item.reducers == ("attention_profile", "pic_retrieval")
            for item in config.query_passes
        ))

    def test_sink_config_has_only_shifted_prediction_and_raw_profiles(self):
        config = AttentionAnalysisConfig.from_file(
            "attention_config/paper_attention_sink.json"
        )
        self.assertEqual(len(config.query_passes), 1)
        query_pass = config.query_passes[0]
        self.assertEqual(query_pass.query_pass_id, "shifted_prediction")
        self.assertEqual(
            query_pass.reducers,
            (
                "attention_profile",
                "pic_retrieval",
                "raw_attention_profile",
            ),
        )

    def test_query_pass_fields_and_reducers_are_strict(self):
        value = {
            "query_pass_id": "gold_answer",
            "kind": "gold_answer_literal_tokens",
            "reducers": ["attention_profile", "pic_retrieval"],
        }
        self.assertEqual(QueryPassSpec.from_dict(value).to_dict(), value)
        with self.assertRaises(ValueError):
            QueryPassSpec.from_dict({**value, "unknown": True})
        with self.assertRaises(ValueError):
            QueryPassSpec.from_dict({**value, "reducers": ["attention_profile"] * 2})

        raw_value = {
            **value,
            "reducers": [
                "attention_profile",
                "pic_retrieval",
                "raw_attention_profile",
            ],
        }
        self.assertEqual(QueryPassSpec.from_dict(raw_value).to_dict(), raw_value)
        for reducers in (["raw_attention_profile"], [
            "attention_profile", "raw_attention_profile"
        ]):
            with self.subTest(reducers=reducers):
                with self.assertRaisesRegex(
                    ValueError, "requires attention_profile and pic_retrieval"
                ):
                    QueryPassSpec.from_dict({**value, "reducers": reducers})

    def test_analysis_rejects_duplicate_query_pass_ids(self):
        query_pass = {
            "query_pass_id": "gold_answer",
            "kind": "gold_answer_literal_tokens",
            "reducers": ["attention_profile"],
        }
        with self.assertRaises(ValueError):
            AttentionAnalysisConfig.from_dict({
                "schema_version": 2,
                "query_passes": [query_pass, query_pass],
            })


if __name__ == "__main__":
    unittest.main()
