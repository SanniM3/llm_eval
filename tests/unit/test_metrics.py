"""Tests for evaluation metrics."""

import pytest

from evalab.evaluation.accuracy import (
    AccuracySuite,
    exact_match,
    normalize_answer,
    token_f1,
)
from evalab.evaluation.calibration import compute_ece, extract_confidence


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_lowercase(self):
        assert normalize_answer("HELLO") == "hello"

    def test_remove_punctuation(self):
        assert normalize_answer("Hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the quick brown fox") == "quick brown fox"
        assert normalize_answer("a cat") == "cat"
        assert normalize_answer("an apple") == "apple"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestExactMatch:
    """Tests for exact match scoring."""

    def test_exact_match_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self):
        assert exact_match("PARIS", "paris") == 1.0

    def test_exact_match_with_articles(self):
        assert exact_match("the capital is Paris", "Paris") == 0.0
        assert exact_match("Paris", "the Paris") == 1.0  # Article removed

    def test_exact_match_different(self):
        assert exact_match("London", "Paris") == 0.0


class TestTokenF1:
    """Tests for token-level F1 scoring."""

    def test_perfect_match(self):
        precision, recall, f1 = token_f1("the cat sat", "the cat sat")
        assert f1 == 1.0

    def test_partial_match(self):
        precision, recall, f1 = token_f1("the cat", "the cat sat on the mat")
        # "cat" matches (after removing articles)
        assert 0 < f1 < 1

    def test_no_match(self):
        precision, recall, f1 = token_f1("dog", "cat")
        assert f1 == 0.0

    def test_empty_prediction(self):
        precision, recall, f1 = token_f1("", "answer")
        assert f1 == 0.0

    def test_both_empty(self):
        precision, recall, f1 = token_f1("", "")
        assert f1 == 1.0


class TestAccuracySuite:
    """Tests for AccuracySuite."""

    def test_evaluate_exact_match(self):
        suite = AccuracySuite()
        results = suite.evaluate("Paris", {"answer": "Paris"})

        em_result = next(r for r in results if r.name == "exact_match")
        assert em_result.value == 1.0

    def test_evaluate_with_aliases(self):
        suite = AccuracySuite(check_aliases=True)
        results = suite.evaluate("4", {"answer": "four", "aliases": ["4"]})

        em_result = next(r for r in results if r.name == "exact_match")
        assert em_result.value == 1.0

    def test_evaluate_wrong_answer(self):
        suite = AccuracySuite()
        results = suite.evaluate("London", {"answer": "Paris"})

        em_result = next(r for r in results if r.name == "exact_match")
        assert em_result.value == 0.0


class TestExtractConfidence:
    """Tests for confidence extraction."""

    def test_extract_confidence_colon(self):
        assert extract_confidence("Answer: Yes\nConfidence: 0.85") == 0.85

    def test_extract_confidence_equals(self):
        assert extract_confidence("confidence=0.9") == 0.9

    def test_extract_confidence_percentage(self):
        assert extract_confidence("I am 90% confident") == 0.9

    def test_extract_confidence_not_found(self):
        assert extract_confidence("Just a regular answer") is None

    def test_extract_confidence_clamp(self):
        # Values > 1 should be treated as percentages
        assert extract_confidence("Confidence: 95") == 0.95


class TestComputeECE:
    """Tests for Expected Calibration Error."""

    def test_perfect_calibration(self):
        # All confident and correct
        confidences = [1.0, 1.0, 1.0]
        correctness = [True, True, True]
        ece, _ = compute_ece(confidences, correctness, num_bins=10)
        assert ece == 0.0

    def test_overconfident(self):
        # High confidence but wrong
        confidences = [0.9, 0.9, 0.9]
        correctness = [False, False, False]
        ece, _ = compute_ece(confidences, correctness, num_bins=10)
        assert ece > 0.8  # Should be close to 0.9

    def test_empty_inputs(self):
        ece, bins = compute_ece([], [], num_bins=10)
        assert ece == 0.0
        assert bins == []
