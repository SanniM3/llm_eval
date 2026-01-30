"""Base classes for evaluation suites."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """
    Result of a metric computation for a single example.

    Attributes:
        name: Metric name
        value: Scalar metric value
        details: Optional detailed breakdown
    """

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
        }


@dataclass
class EvaluationResult:
    """
    Result of evaluating a single example.

    Contains all metrics computed for the example.
    """

    example_id: str
    metrics: list[MetricResult]
    error: str | None = None

    def get_metric(self, name: str) -> MetricResult | None:
        """Get a specific metric by name."""
        for m in self.metrics:
            if m.name == name:
                return m
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "error": self.error,
        }


class EvaluationSuite(ABC):
    """
    Abstract base class for evaluation suites.

    Each suite computes one or more related metrics.
    """

    def __init__(self, name: str):
        """
        Initialize evaluation suite.

        Args:
            name: Suite identifier
        """
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate a single prediction against reference.

        Args:
            prediction: Model output
            reference: Ground truth (string or dict depending on task)
            **kwargs: Additional context (e.g., retrieved_context, input)

        Returns:
            List of MetricResult objects
        """
        pass

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> list[list[MetricResult]]:
        """
        Evaluate a batch of predictions.

        Default implementation calls evaluate() for each item.
        Override for optimized batch processing.

        Args:
            predictions: List of model outputs
            references: List of ground truths
            **kwargs: Additional context

        Returns:
            List of metric lists, one per example
        """
        results = []
        for pred, ref in zip(predictions, references):
            results.append(self.evaluate(pred, ref, **kwargs))
        return results

    def get_metric_names(self) -> list[str]:
        """
        Get names of metrics this suite produces.

        Returns:
            List of metric names
        """
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
