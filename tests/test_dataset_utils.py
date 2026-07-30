import unittest
from unittest import mock

from sempic.dataset.utils import split_with_nltk


class SplitWithNltkTests(unittest.TestCase):
    def test_returns_empty_list_for_empty_sentence_output(self):
        with mock.patch(
            "sempic.dataset.utils.nltk.sent_tokenize",
            return_value=[],
        ):
            chunks = split_with_nltk("ignored", max_chunk_size=100)

        self.assertEqual(chunks, [])

    def test_does_not_duplicate_first_sentence(self):
        with mock.patch(
            "sempic.dataset.utils.nltk.sent_tokenize",
            return_value=["First sentence.", "Second sentence."],
        ):
            chunks = split_with_nltk("ignored", max_chunk_size=100)

        self.assertEqual(chunks, ["First sentence. Second sentence."])

    def test_splits_when_next_sentence_exceeds_chunk_size(self):
        with mock.patch(
            "sempic.dataset.utils.nltk.sent_tokenize",
            return_value=["First sentence.", "Second sentence."],
        ):
            chunks = split_with_nltk("ignored", max_chunk_size=20)

        self.assertEqual(chunks, ["First sentence.", "Second sentence."])

    def test_downloads_punkt_tab_for_current_nltk_lookup_error(self):
        error = LookupError("Resource punkt_tab not found.")

        with mock.patch(
            "sempic.dataset.utils.nltk.sent_tokenize",
            side_effect=[error, ["Only sentence."]],
        ), mock.patch("sempic.dataset.utils.nltk.download") as download:
            chunks = split_with_nltk("ignored", max_chunk_size=100)

        self.assertEqual(chunks, ["Only sentence."])
        download.assert_called_once_with("punkt_tab", quiet=True)

    def test_downloads_punkt_for_older_nltk_lookup_error(self):
        error = LookupError("Resource punkt not found.")

        with mock.patch(
            "sempic.dataset.utils.nltk.sent_tokenize",
            side_effect=[error, ["Only sentence."]],
        ), mock.patch("sempic.dataset.utils.nltk.download") as download:
            chunks = split_with_nltk("ignored", max_chunk_size=100)

        self.assertEqual(chunks, ["Only sentence."])
        download.assert_called_once_with("punkt", quiet=True)


if __name__ == "__main__":
    unittest.main()
