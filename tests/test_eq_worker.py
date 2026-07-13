from __future__ import annotations

import unittest

import torch

from src.workers.eq_worker import (
    calculate_eq_score,
    pretokenize_eq_dataset,
    select_eq_reference,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return "prompt"

    def __call__(self, prompt, *, return_tensors):
        del prompt, return_tensors
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones((1, 2), dtype=torch.long),
        }


class EqReferenceTests(unittest.TestCase):
    def setUp(self):
        self.normalized = {
            "emotion1_score": 2.307692307692308,
            "emotion2_score": 3.076923076923077,
            "emotion3_score": 4.615384615384616,
            "emotion4_score": 0.0,
        }
        self.fullscale = {
            "emotion1_score": 3,
            "emotion2_score": 4,
            "emotion3_score": 6,
            "emotion4_score": 0,
        }

    def test_select_eq_reference_prefers_fullscale_labels(self):
        sample = {
            "reference_answer": self.normalized,
            "reference_answer_fullscale": self.fullscale,
        }

        self.assertIs(select_eq_reference(sample), self.fullscale)

    def test_select_eq_reference_supports_legacy_normalized_only_data(self):
        self.assertIs(
            select_eq_reference({"reference_answer": self.normalized}),
            self.normalized,
        )

    def test_exact_fullscale_prediction_receives_full_score(self):
        sample = {
            "reference_answer": self.normalized,
            "reference_answer_fullscale": self.fullscale,
        }

        score = calculate_eq_score(
            self.fullscale,
            select_eq_reference(sample),
        )

        self.assertEqual(score, 1.0)

    def test_pretokenize_eq_dataset_stores_fullscale_reference(self):
        dataset = {
            "q1": {
                "prompt": "Example EQ prompt",
                "reference_answer": self.normalized,
                "reference_answer_fullscale": self.fullscale,
            }
        }

        tokenized = pretokenize_eq_dataset(
            dataset,
            FakeTokenizer(),
            torch.device("cpu"),
        )

        self.assertIs(tokenized["q1"]["reference"], self.fullscale)


if __name__ == "__main__":
    unittest.main()
