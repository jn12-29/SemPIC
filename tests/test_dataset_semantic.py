import random
import unittest
from unittest import mock

from datasets import Dataset, DatasetDict

from sempic.dataset import biography, hotpot_qa, musique, niah
from sempic.prompt import ContextBlock


def context_contents(entry):
    return [
        part.content
        for part in entry["prompt"].parts
        if isinstance(part, ContextBlock)
    ]


class DatasetSemanticTests(unittest.TestCase):
    def test_biography_exposes_documents_query_shots_and_attribute(self):
        train_rows = []
        test_rows = []
        for row in range(4):
            train_row = {
                attribute: [f"document-{row}-{attribute}"]
                for attribute in biography.BIO_ATTRIBUTES
            }
            test_row = {
                attribute: [f"question-{row}-{attribute}"]
                for attribute in biography.BIO_ATTRIBUTES
            }
            test_row.update({
                "name": f"person-{row}",
                "labels": {
                    attribute: f"answer-{row}-{attribute}"
                    for attribute in biography.BIO_ATTRIBUTES
                },
            })
            train_rows.append(train_row)
            test_rows.append(test_row)
        dataset = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        })

        with (
            mock.patch.object(biography, "load_dataset", return_value=dataset),
            mock.patch.object(biography, "filter_bio_dataset", return_value=dataset),
        ):
            entry = next(biography.bio_ret_eval_generator(
                num_samples=1,
                num_data_strs=1,
                num_shots=1,
                split="train",
                seed=3,
                cache_dataset=False,
                question_type="completion",
            ))

        semantic = entry["semantic"]
        attribute = semantic["task"]["attribute"]
        self.assertEqual(semantic["documents"], context_contents(entry))
        self.assertRegex(semantic["query"], rf"^question-\d+-{attribute}$")
        self.assertEqual(len(semantic["shots"]), 1)
        shot = semantic["shots"][0]
        self.assertEqual(shot["task"].keys(), {"attribute"})
        shot_attribute = shot["task"]["attribute"]
        self.assertRegex(shot["query"], rf"^question-\d+-{shot_attribute}$")
        shot_row = shot["query"].split("-")[1]
        self.assertEqual(shot["documents"], [" ".join(
            f"document-{shot_row}-{item_attribute}"
            for item_attribute in biography.BIO_ATTRIBUTES
        )])
        self.assertEqual(shot["answer"], f"answer-{shot_row}-{shot_attribute}")
        self.assertIn(shot["answer"], entry["prompt"].parts[0].content)
        self.assertIn(attribute, biography.BIO_ATTRIBUTES)

        with (
            mock.patch.object(biography, "load_dataset", return_value=dataset),
            mock.patch.object(biography, "filter_bio_dataset", return_value=dataset),
        ):
            qa_entry = next(biography.bio_ret_eval_generator(
                num_samples=1,
                num_data_strs=1,
                num_shots=0,
                split="train",
                seed=3,
                cache_dataset=False,
                question_type="QA",
            ))

        self.assertNotIn("Respond with the answer only", qa_entry["semantic"]["query"])
        self.assertIn("Respond with the answer only", qa_entry["query"])

    def test_hotpot_payload_excludes_instruction_and_cot_controls(self):
        dataset = Dataset.from_list([
            {
                "question": f"question-{row}",
                "answer": f"answer-{row}",
                "level": "hard",
                "context": {
                    "title": [f"title-{row}-0", f"title-{row}-1"],
                    "sentences": [
                        [f"document-{row}-0"],
                        [f"document-{row}-1"],
                    ],
                },
            }
            for row in range(3)
        ])

        def generate(**controls):
            with mock.patch.object(hotpot_qa, "load_dataset", return_value=dataset):
                return next(hotpot_qa.hotpot_qa_ret_eval_generator(
                    num_samples=1,
                    num_data_strs=0,
                    num_shots=1,
                    seed=5,
                    **controls,
                ))

        plain = generate(add_inst=False, add_cot=False)
        instructed = generate(add_inst=True, add_cot=True)

        self.assertEqual(plain["semantic"], instructed["semantic"])
        self.assertNotEqual(plain["prompt"], instructed["prompt"])
        semantic = plain["semantic"]
        row = semantic["query"].split("-")[1]
        self.assertEqual(
            semantic["documents"],
            [f"document-{row}-0", f"document-{row}-1"],
        )
        self.assertEqual(semantic["documents"], context_contents(plain))
        self.assertEqual(semantic["task"], {})
        self.assertEqual(len(semantic["shots"]), 1)
        shot = semantic["shots"][0]
        shot_row = shot["query"].split("-")[1]
        self.assertEqual(shot["documents"], [f"document-{shot_row}-0", f"document-{shot_row}-1"])
        self.assertEqual(shot["answer"], f"answer-{shot_row}")

    def test_hotpot_defaults_to_hard_and_filters_few_shots(self):
        levels = ["easy", "hard", "medium", "hard", "easy", "hard"]
        dataset = Dataset.from_list([
            {
                "question": f"question-{row}",
                "answer": f"answer-{row}",
                "level": level,
                "context": {
                    "title": [f"title-{row}"],
                    "sentences": [[f"document-{row}"]],
                },
            }
            for row, level in enumerate(levels)
        ])

        def generate(num_samples, num_shots=0, **data_kwargs):
            with mock.patch.object(hotpot_qa, "load_dataset", return_value=dataset):
                return list(hotpot_qa.hotpot_qa_ret_eval_generator(
                    num_samples=num_samples,
                    num_data_strs=0,
                    num_shots=num_shots,
                    seed=5,
                    add_inst=False,
                    add_cot=False,
                    **data_kwargs,
                ))

        all_entries = generate(6, difficulty=["easy", "medium", "hard"])
        original_random_state = random.getstate()
        self.addCleanup(random.setstate, original_random_state)
        random.seed(123)
        global_random_state = random.getstate()
        generate(3)
        self.assertEqual(random.getstate(), global_random_state)
        hard_entries = generate(3)
        expected_hard_queries = [
            entry["query"]
            for entry in all_entries
            if levels[int(entry["query"].split("-")[1])] == "hard"
        ]
        self.assertEqual(
            [entry["query"] for entry in hard_entries],
            expected_hard_queries,
        )
        easy_entries = generate(2, difficulty=["easy"])
        self.assertEqual(
            [entry["query"] for entry in easy_entries],
            [
                entry["query"]
                for entry in all_entries
                if levels[int(entry["query"].split("-")[1])] == "easy"
            ],
        )
        medium_hard_entries = generate(4, difficulty=["medium", "hard"])
        self.assertEqual(
            [entry["query"] for entry in medium_hard_entries],
            [
                entry["query"]
                for entry in all_entries
                if levels[int(entry["query"].split("-")[1])] in {"medium", "hard"}
            ],
        )

        with self.assertWarnsRegex(UserWarning, "greater than dataset size"):
            entries_with_shot = generate(4, num_shots=1)
        self.assertEqual(len(entries_with_shot), 2)
        for entry in entries_with_shot:
            row = int(entry["query"].split("-")[1])
            self.assertEqual(levels[row], "hard")
            self.assertEqual(len(entry["semantic"]["shots"]), 1)
            shot_row = int(entry["semantic"]["shots"][0]["query"].split("-")[1])
            self.assertEqual(levels[shot_row], "hard")

    def test_hotpot_validates_difficulty(self):
        dataset = Dataset.from_list([{
            "question": "question-0",
            "answer": "answer-0",
            "level": "hard",
            "context": {
                "title": ["title-0"],
                "sentences": [["document-0"]],
            },
        }])

        for difficulty in (None, [], "hard", ["unknown"], ["hard", 1]):
            with self.subTest(difficulty=difficulty):
                with mock.patch.object(hotpot_qa, "load_dataset", return_value=dataset):
                    with self.assertRaises(ValueError):
                        list(hotpot_qa.hotpot_qa_ret_eval_generator(
                            num_samples=1,
                            num_data_strs=0,
                            num_shots=0,
                            seed=5,
                            difficulty=difficulty,
                        ))

    def test_musique_payload_uses_raw_question_and_ordered_paragraphs(self):
        dataset = Dataset.from_list([
            {
                "question": f"question-{row}",
                "answer": f"answer-{row}",
                "paragraphs": [
                    {"paragraph_text": f"document-{row}-0"},
                    {"paragraph_text": f"document-{row}-1"},
                ],
            }
            for row in range(3)
        ])

        def generate(**controls):
            with mock.patch.object(musique, "load_dataset", return_value=dataset):
                return next(musique.musique_ret_eval_generator(
                    num_samples=1,
                    num_data_strs=0,
                    num_shots=1,
                    seed=7,
                    cache_dataset=False,
                    **controls,
                ))

        entry = generate(add_inst=True, add_cot=True)
        plain = generate(add_inst=False, add_cot=False)

        semantic = entry["semantic"]
        self.assertEqual(semantic, plain["semantic"])
        self.assertNotEqual(entry["prompt"], plain["prompt"])
        row = semantic["query"].split("-")[1]
        self.assertEqual(
            semantic["documents"],
            [f"document-{row}-0", f"document-{row}-1"],
        )
        self.assertEqual(semantic["documents"], context_contents(entry))
        self.assertNotIn("thinking step by step", semantic["query"])
        self.assertEqual(semantic["task"], {})
        self.assertEqual(len(semantic["shots"]), 1)
        shot = semantic["shots"][0]
        shot_row = shot["query"].split("-")[1]
        self.assertEqual(shot["documents"], [f"document-{shot_row}-0", f"document-{shot_row}-1"])
        self.assertEqual(shot["answer"], f"answer-{shot_row}")

    def test_niah_payload_uses_raw_context_and_intentionally_has_no_shots(self):
        dataset = Dataset.from_list([
            {
                "context": f"raw context {row}",
                "question": f"raw question {row}",
                "answer": [f"answer-{row}"],
            }
            for row in range(4)
        ])

        with (
            mock.patch.object(niah, "load_dataset", return_value=dataset),
            mock.patch.object(
                niah,
                "split_with_nltk",
                return_value=["rendered chunk 0", "rendered chunk 1"],
            ),
        ):
            entry = next(niah.niah_ret_eval_generator(
                num_samples=1,
                num_data_strs=0,
                num_shots=1,
                split="test",
                seed=11,
                cache_dataset=False,
                chunk_size=4,
            ))

        semantic = entry["semantic"]
        row = semantic["query"].rsplit(" ", 1)[1]
        self.assertEqual(semantic["documents"], [f"raw context {row}"])
        self.assertEqual(semantic["query"], f"raw question {row}")
        self.assertNotIn("Provide a short answer", semantic["query"])
        self.assertEqual(semantic["shots"], [])
        self.assertEqual(semantic["task"], {})
        self.assertEqual(context_contents(entry), ["rendered chunk 0", "rendered chunk 1"])

    def test_niah_rejects_unknown_split(self):
        dataset = Dataset.from_list([{
            "context": "raw context",
            "question": "raw question",
            "answer": ["answer"],
        }])

        with mock.patch.object(niah, "load_dataset", return_value=dataset) as load_dataset:
            with self.assertRaisesRegex(ValueError, "Unknown split: trian"):
                list(niah.niah_ret_eval_generator(
                    num_samples=1,
                    num_data_strs=0,
                    num_shots=0,
                    split="trian",
                    seed=11,
                    cache_dataset=False,
                ))
        load_dataset.assert_not_called()

    def test_same_documents_with_different_queries_are_distinct_payloads(self):
        first = {
            "documents": ["document"],
            "query": "first question",
            "shots": [],
            "task": {},
        }
        second = {**first, "query": "second question"}

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
