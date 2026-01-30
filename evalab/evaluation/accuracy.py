"""Accuracy evaluation suite for QA and classification."""

import re
import string
from collections import Counter
from typing import Any

from evalab.evaluation.base import EvaluationSuite, MetricResult


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.

    Applies standard SQuAD-style normalization:
    - Lowercase
    - Remove punctuation
    - Remove articles
    - Remove extra whitespace
    """
    text = text.lower()

    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)

    # Remove articles
    articles = {"a", "an", "the"}
    words = text.split()
    words = [w for w in words if w not in articles]

    # Collapse whitespace
    return " ".join(words)


def exact_match(prediction: str, reference: str) -> float:
    """
    Compute exact match score.

    Args:
        prediction: Model output
        reference: Ground truth

    Returns:
        1.0 if match, 0.0 otherwise
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> tuple[float, float, float]:
    """
    Compute token-level precision, recall, and F1.

    Uses SQuAD-style tokenization and normalization.

    Args:
        prediction: Model output
        reference: Ground truth

    Returns:
        Tuple of (precision, recall, f1)
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        if not pred_tokens and not ref_tokens:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    # Count matching tokens
    common = pred_counter & ref_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0, 0.0, 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def extract_answer(text: str, pattern: str | None = None) -> str:
    """
    Extract answer from model output.

    Handles common formats like "Answer: X" or "The answer is X".

    Args:
        text: Model output text
        pattern: Optional regex pattern

    Returns:
        Extracted answer text
    """
    text = text.strip()

    if pattern:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Try common patterns
    patterns = [
        r"^(?:answer|response):\s*(.+)",
        r"^(?:the answer is|answer is)\s*(.+)",
        r"^(.+?)(?:\.|$)",  # First sentence
    ]

    for p in patterns:
        match = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

    return text


class AccuracySuite(EvaluationSuite):
    """
    Accuracy evaluation suite.

    Computes:
    - Exact match (EM)
    - Token F1 (SQuAD-style)
    - Accuracy (for classification)
    - Macro F1 (for multi-class)
    """

    def __init__(
        self,
        extract_pattern: str | None = None,
        check_aliases: bool = True,
    ):
        """
        Initialize accuracy suite.

        Args:
            extract_pattern: Optional regex to extract answer from output
            check_aliases: Whether to check alias answers
        """
        super().__init__(name="accuracy")
        self.extract_pattern = extract_pattern
        self.check_aliases = check_aliases

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate prediction accuracy.

        Args:
            prediction: Model output
            reference: Ground truth (string or dict with 'answer' and optional 'aliases')
            **kwargs: Additional context

        Returns:
            List of accuracy metrics
        """
        # Extract answer from prediction
        pred_answer = extract_answer(prediction, self.extract_pattern)

        # Handle reference format
        if isinstance(reference, dict):
            ref_answer = reference.get("answer", "")
            aliases = reference.get("aliases", [])
        else:
            ref_answer = reference
            aliases = []

        # Compute against main answer
        em = exact_match(pred_answer, ref_answer)
        precision, recall, f1 = token_f1(pred_answer, ref_answer)

        # Check aliases if not already matched
        if em == 0.0 and self.check_aliases and aliases:
            for alias in aliases:
                alias_em = exact_match(pred_answer, alias)
                if alias_em > em:
                    em = alias_em
                    _, _, f1 = token_f1(pred_answer, alias)
                    break

        return [
            MetricResult(
                name="exact_match",
                value=em,
                details={"prediction": pred_answer, "reference": ref_answer},
            ),
            MetricResult(
                name="token_f1",
                value=f1,
                details={"precision": precision, "recall": recall},
            ),
        ]

    def get_metric_names(self) -> list[str]:
        return ["exact_match", "token_f1"]


class ClassificationAccuracySuite(EvaluationSuite):
    """
    Accuracy suite for classification tasks.

    Computes:
    - Accuracy
    - Per-class metrics when given full dataset
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize classification accuracy suite.

        Args:
            normalize: Whether to normalize labels
        """
        super().__init__(name="classification_accuracy")
        self.normalize = normalize

    def _normalize_label(self, label: str) -> str:
        """Normalize a label for comparison."""
        if self.normalize:
            return label.strip().lower()
        return label.strip()

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate classification accuracy.

        Args:
            prediction: Predicted label
            reference: True label (string or dict with 'label')
            **kwargs: Additional context

        Returns:
            List of accuracy metrics
        """
        pred_label = self._normalize_label(prediction)

        # Handle reference format
        if isinstance(reference, dict):
            ref_label = self._normalize_label(reference.get("label", ""))
            secondary = [self._normalize_label(l) for l in reference.get("secondary_labels", [])]
        else:
            ref_label = self._normalize_label(reference)
            secondary = []

        # Check match
        is_correct = pred_label == ref_label or pred_label in secondary
        accuracy = 1.0 if is_correct else 0.0

        return [
            MetricResult(
                name="accuracy",
                value=accuracy,
                details={
                    "prediction": pred_label,
                    "reference": ref_label,
                    "secondary_labels": secondary,
                },
            ),
        ]

    def get_metric_names(self) -> list[str]:
        return ["accuracy"]


def compute_macro_f1(
    predictions: list[str],
    references: list[str],
    labels: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute macro F1 score across all classes.

    Args:
        predictions: List of predicted labels
        references: List of true labels
        labels: Optional list of all possible labels

    Returns:
        Dictionary with per-class and macro metrics
    """
    # Normalize
    predictions = [p.strip().lower() for p in predictions]
    references = [r.strip().lower() for r in references]

    # Get all labels
    if labels is None:
        labels = sorted(set(predictions) | set(references))

    # Compute per-class metrics
    per_class = {}
    for label in labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}

    # Compute macro averages
    macro_precision = sum(c["precision"] for c in per_class.values()) / len(labels)
    macro_recall = sum(c["recall"] for c in per_class.values()) / len(labels)
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(labels)

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
