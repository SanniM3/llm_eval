"""Calibration evaluation suite."""

import logging
import re
from typing import Any

import numpy as np

from evalab.evaluation.base import EvaluationSuite, MetricResult

logger = logging.getLogger(__name__)


def extract_confidence(text: str) -> float | None:
    """
    Extract confidence score from model output.

    Looks for patterns like "Confidence: 0.8", "confidence=0.9", etc.

    Args:
        text: Model output

    Returns:
        Confidence value (0-1) or None if not found
    """
    patterns = [
        r"confidence[:\s=]+(\d*\.?\d+)",
        r"(\d*\.?\d+)%?\s*confident",
        r"certainty[:\s=]+(\d*\.?\d+)",
        r"probability[:\s=]+(\d*\.?\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            value = float(match.group(1))
            # Handle percentage
            if value > 1.0:
                value = value / 100.0
            return min(1.0, max(0.0, value))

    return None


def compute_ece(
    confidences: list[float],
    correctness: list[bool],
    num_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Compute Expected Calibration Error.

    ECE measures the gap between predicted confidence and actual accuracy.

    Args:
        confidences: List of confidence scores (0-1)
        correctness: List of boolean correctness indicators
        num_bins: Number of bins for calibration

    Returns:
        Tuple of (ECE value, bin data for reliability diagram)
    """
    if not confidences or len(confidences) != len(correctness):
        return 0.0, []

    confidences = np.array(confidences)
    correctness = np.array(correctness, dtype=float)

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_data = []

    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        # Find samples in this bin
        if i == num_bins - 1:
            # Last bin includes upper boundary
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_confidences = confidences[mask]
        bin_correctness = correctness[mask]

        bin_size = len(bin_confidences)
        if bin_size > 0:
            avg_confidence = float(np.mean(bin_confidences))
            avg_accuracy = float(np.mean(bin_correctness))
            gap = abs(avg_accuracy - avg_confidence)

            ece += (bin_size / total_samples) * gap

            bin_data.append({
                "bin_idx": i,
                "lower": lower,
                "upper": upper,
                "count": bin_size,
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "gap": gap,
            })
        else:
            bin_data.append({
                "bin_idx": i,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "avg_confidence": None,
                "avg_accuracy": None,
                "gap": None,
            })

    return float(ece), bin_data


def compute_self_consistency(
    samples: list[str],
    normalize: bool = True,
) -> float:
    """
    Compute self-consistency from multiple samples.

    Returns the proportion of samples that agree with the most common answer.

    Args:
        samples: List of model outputs
        normalize: Whether to normalize outputs

    Returns:
        Consistency score (0-1)
    """
    if not samples:
        return 0.0

    if normalize:
        samples = [s.strip().lower() for s in samples]

    # Count occurrences
    from collections import Counter

    counts = Counter(samples)
    most_common_count = counts.most_common(1)[0][1]

    return most_common_count / len(samples)


class CalibrationSuite(EvaluationSuite):
    """
    Calibration evaluation suite.

    Computes:
    - Expected Calibration Error (ECE)
    - Mean confidence
    - Confidence-accuracy correlation
    - Self-consistency (optional)
    """

    def __init__(
        self,
        num_bins: int = 10,
        extract_confidence: bool = True,
        llm_backend: Any | None = None,
        num_samples: int = 5,
    ):
        """
        Initialize calibration suite.

        Args:
            num_bins: Number of bins for ECE
            extract_confidence: Whether to extract confidence from output
            llm_backend: LLM backend for self-consistency sampling
            num_samples: Number of samples for self-consistency
        """
        super().__init__(name="calibration")
        self.num_bins = num_bins
        self.extract_confidence = extract_confidence
        self.llm_backend = llm_backend
        self.num_samples = num_samples

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate calibration for a single example.

        Note: ECE is computed at the dataset level. This returns per-example
        confidence for later aggregation.

        Args:
            prediction: Model output
            reference: Ground truth
            **kwargs:
                - confidence: Pre-computed confidence (0-1)
                - is_correct: Whether prediction is correct
                - samples: List of samples for self-consistency

        Returns:
            List of calibration-related metrics
        """
        results = []

        # Get confidence
        confidence = kwargs.get("confidence")
        if confidence is None and self.extract_confidence:
            confidence = extract_confidence(prediction)

        if confidence is not None:
            results.append(
                MetricResult(
                    name="confidence",
                    value=confidence,
                )
            )

        # Self-consistency
        samples = kwargs.get("samples", [])
        if samples:
            consistency = compute_self_consistency(samples)
            results.append(
                MetricResult(
                    name="self_consistency",
                    value=consistency,
                    details={"num_samples": len(samples)},
                )
            )

        return results

    def evaluate_dataset(
        self,
        predictions: list[str],
        references: list[str | dict[str, Any]],
        confidences: list[float],
        correctness: list[bool],
    ) -> list[MetricResult]:
        """
        Compute dataset-level calibration metrics.

        Args:
            predictions: All model outputs
            references: All ground truths
            confidences: Confidence scores for each prediction
            correctness: Whether each prediction is correct

        Returns:
            List of calibration metrics
        """
        results = []

        # Filter out None confidences
        valid_pairs = [
            (c, correct)
            for c, correct in zip(confidences, correctness)
            if c is not None
        ]

        if not valid_pairs:
            return [
                MetricResult(
                    name="ece",
                    value=0.0,
                    details={"error": "no valid confidence scores"},
                )
            ]

        confs, corrects = zip(*valid_pairs)
        confs = list(confs)
        corrects = list(corrects)

        # Compute ECE
        ece, bin_data = compute_ece(confs, corrects, self.num_bins)
        results.append(
            MetricResult(
                name="ece",
                value=ece,
                details={
                    "bins": bin_data,
                    "num_bins": self.num_bins,
                    "num_samples": len(confs),
                },
            )
        )

        # Mean confidence
        mean_conf = float(np.mean(confs))
        results.append(
            MetricResult(
                name="mean_confidence",
                value=mean_conf,
            )
        )

        # Confidence-accuracy correlation
        if len(set(corrects)) > 1:  # Need variance in both
            corr = float(np.corrcoef(confs, corrects)[0, 1])
            if not np.isnan(corr):
                results.append(
                    MetricResult(
                        name="confidence_accuracy_corr",
                        value=corr,
                    )
                )

        # Overall accuracy
        accuracy = float(np.mean(corrects))
        results.append(
            MetricResult(
                name="overall_accuracy",
                value=accuracy,
            )
        )

        return results

    def get_metric_names(self) -> list[str]:
        return [
            "confidence",
            "self_consistency",
            "ece",
            "mean_confidence",
            "confidence_accuracy_corr",
        ]
