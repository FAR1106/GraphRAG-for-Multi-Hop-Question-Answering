"""tests/test_evaluation.py — Unit tests for evaluation metrics."""
from __future__ import annotations

import pytest
from evaluation.metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    supporting_fact_recall,
    compute_metrics,
)


class TestNormalizeAnswer:
    def test_lowercases(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_removes_articles(self):
        assert normalize_answer("the cat") == "cat"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_removes_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1

    def test_article_ignored(self):
        assert exact_match("the United States", "United States") == 1

    def test_mismatch(self):
        assert exact_match("London", "Paris") == 0

    def test_empty_prediction(self):
        assert exact_match("", "Paris") == 0


class TestTokenF1:
    def test_perfect_match(self):
        result = token_f1("hello world", "hello world")
        assert result["f1"] == 1.0

    def test_no_overlap(self):
        result = token_f1("apple orange", "cat dog")
        assert result["f1"] == 0.0

    def test_partial_match(self):
        result = token_f1("the cat sat on the mat", "the cat")
        assert 0.0 < result["f1"] < 1.0

    def test_empty_prediction(self):
        result = token_f1("", "some answer")
        assert result["f1"] == 0.0


class TestSupportingFactRecall:
    def test_all_retrieved(self):
        gold_facts = [["Doc1", 0], ["Doc2", 1]]
        retrieved = ["Doc1", "Doc2", "Doc3"]
        assert supporting_fact_recall(gold_facts, retrieved) == 1.0

    def test_none_retrieved(self):
        gold_facts = [["Doc1", 0], ["Doc2", 1]]
        retrieved = ["Doc3"]
        assert supporting_fact_recall(gold_facts, retrieved) == 0.0

    def test_partial(self):
        gold_facts = [["Doc1", 0], ["Doc2", 1]]
        retrieved = ["Doc1"]
        assert supporting_fact_recall(gold_facts, retrieved) == 0.5

    def test_empty_gold_facts(self):
        # vacuously true
        assert supporting_fact_recall([], ["Doc1"]) == 1.0


class TestComputeMetrics:
    def _make_pred(self, gold, pred, gold_docs, retrieved_docs):
        return {
            "gold_answer": gold,
            "predicted_answer": pred,
            "gold_supporting_facts": [[d, 0] for d in gold_docs],
            "retrieved_titles": retrieved_docs,
            "latency_s": 1.0,
        }

    def test_all_correct(self):
        preds = [
            self._make_pred("Paris", "Paris", ["Doc1"], ["Doc1"]),
            self._make_pred("London", "London", ["Doc2"], ["Doc2"]),
        ]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["supporting_fact_recall"] == 1.0

    def test_all_wrong(self):
        preds = [self._make_pred("Paris", "Rome", ["Doc1"], [])]
        metrics = compute_metrics(preds)
        assert metrics["exact_match"] == 0.0

    def test_empty_input(self):
        metrics = compute_metrics([])
        assert "error" in metrics

    def test_latency_reported(self):
        preds = [self._make_pred("X", "X", [], [])]
        metrics = compute_metrics(preds)
        assert "latency_median_s" in metrics
        assert metrics["latency_median_s"] >= 0
