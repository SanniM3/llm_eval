"""Cost and latency evaluation suite."""

import logging
from typing import Any

import numpy as np

from evalab.config.defaults import DEFAULT_PRICING
from evalab.evaluation.base import EvaluationSuite, MetricResult

logger = logging.getLogger(__name__)


class CostLatencySuite(EvaluationSuite):
    """
    Cost and latency evaluation suite.

    Tracks:
    - Token usage (input/output)
    - Latency distribution (p50, p95, p99)
    - Estimated cost
    - Throughput
    """

    def __init__(
        self,
        pricing_table: dict[str, dict[str, float]] | None = None,
        model_name: str | None = None,
    ):
        """
        Initialize cost/latency suite.

        Args:
            pricing_table: Dict of model -> {input: price, output: price} per 1M tokens
            model_name: Model name for cost estimation
        """
        super().__init__(name="cost_latency")
        self.pricing = pricing_table or DEFAULT_PRICING
        self.model_name = model_name

    def _get_model_pricing(self, model: str | None = None) -> dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Dict with input and output prices per 1M tokens
        """
        model = model or self.model_name

        if model:
            # Try exact match
            if model in self.pricing:
                return self.pricing[model]

            # Try prefix match (e.g., "gpt-4o-mini-2024-07-18" -> "gpt-4o-mini")
            for key in self.pricing:
                if model.startswith(key):
                    return self.pricing[key]

        return self.pricing.get("default", {"input": 1.0, "output": 3.0})

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """
        Estimate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name

        Returns:
            Estimated cost in USD
        """
        pricing = self._get_model_pricing(model)

        # Prices are per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate cost/latency for a single generation.

        Args:
            prediction: Model output (not used directly)
            reference: Reference (not used)
            **kwargs:
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - latency_ms: Generation latency in milliseconds
                - model: Model name

        Returns:
            List of cost/latency metrics
        """
        input_tokens = kwargs.get("input_tokens", 0)
        output_tokens = kwargs.get("output_tokens", 0)
        latency_ms = kwargs.get("latency_ms", 0.0)
        model = kwargs.get("model", self.model_name)

        results = []

        # Token counts
        results.append(MetricResult(name="input_tokens", value=float(input_tokens)))
        results.append(MetricResult(name="output_tokens", value=float(output_tokens)))
        results.append(MetricResult(name="total_tokens", value=float(input_tokens + output_tokens)))

        # Latency
        results.append(MetricResult(name="latency_ms", value=latency_ms))

        # Cost
        cost = self.estimate_cost(input_tokens, output_tokens, model)
        results.append(MetricResult(name="estimated_cost_usd", value=cost))

        # Tokens per second
        if latency_ms > 0:
            tokens_per_sec = (output_tokens / latency_ms) * 1000
            results.append(MetricResult(name="output_tokens_per_sec", value=tokens_per_sec))

        return results

    def evaluate_dataset(
        self,
        all_input_tokens: list[int],
        all_output_tokens: list[int],
        all_latencies: list[float],
        model: str | None = None,
        total_duration_sec: float | None = None,
    ) -> list[MetricResult]:
        """
        Compute dataset-level cost/latency metrics.

        Args:
            all_input_tokens: Input tokens per example
            all_output_tokens: Output tokens per example
            all_latencies: Latencies per example (ms)
            model: Model name
            total_duration_sec: Total run duration

        Returns:
            List of aggregated metrics
        """
        results = []

        if not all_latencies:
            return results

        # Token totals
        total_input = sum(all_input_tokens)
        total_output = sum(all_output_tokens)
        total_tokens = total_input + total_output

        results.append(MetricResult(name="total_input_tokens", value=float(total_input)))
        results.append(MetricResult(name="total_output_tokens", value=float(total_output)))
        results.append(MetricResult(name="total_tokens", value=float(total_tokens)))

        # Token averages
        results.append(
            MetricResult(
                name="mean_input_tokens",
                value=float(np.mean(all_input_tokens)),
            )
        )
        results.append(
            MetricResult(
                name="mean_output_tokens",
                value=float(np.mean(all_output_tokens)),
            )
        )

        # Latency statistics
        latencies = np.array(all_latencies)
        results.append(MetricResult(name="mean_latency_ms", value=float(np.mean(latencies))))
        results.append(MetricResult(name="std_latency_ms", value=float(np.std(latencies))))
        results.append(MetricResult(name="min_latency_ms", value=float(np.min(latencies))))
        results.append(MetricResult(name="max_latency_ms", value=float(np.max(latencies))))
        results.append(MetricResult(name="p50_latency_ms", value=float(np.percentile(latencies, 50))))
        results.append(MetricResult(name="p95_latency_ms", value=float(np.percentile(latencies, 95))))
        results.append(MetricResult(name="p99_latency_ms", value=float(np.percentile(latencies, 99))))

        # Cost
        total_cost = self.estimate_cost(total_input, total_output, model)
        results.append(MetricResult(name="total_cost_usd", value=total_cost))
        results.append(
            MetricResult(
                name="mean_cost_usd",
                value=total_cost / len(all_latencies),
            )
        )

        # Throughput
        if total_duration_sec and total_duration_sec > 0:
            requests_per_sec = len(all_latencies) / total_duration_sec
            tokens_per_sec = total_output / total_duration_sec

            results.append(MetricResult(name="throughput_req_per_sec", value=requests_per_sec))
            results.append(MetricResult(name="throughput_tokens_per_sec", value=tokens_per_sec))

        return results

    def get_metric_names(self) -> list[str]:
        return [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "latency_ms",
            "estimated_cost_usd",
            "output_tokens_per_sec",
            "total_input_tokens",
            "total_output_tokens",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "total_cost_usd",
            "throughput_req_per_sec",
        ]


def format_cost(cost_usd: float) -> str:
    """
    Format cost as human-readable string.

    Args:
        cost_usd: Cost in USD

    Returns:
        Formatted string
    """
    if cost_usd < 0.01:
        return f"${cost_usd:.4f}"
    elif cost_usd < 1.0:
        return f"${cost_usd:.3f}"
    else:
        return f"${cost_usd:.2f}"


def format_latency(latency_ms: float) -> str:
    """
    Format latency as human-readable string.

    Args:
        latency_ms: Latency in milliseconds

    Returns:
        Formatted string
    """
    if latency_ms < 1000:
        return f"{latency_ms:.1f}ms"
    else:
        return f"{latency_ms / 1000:.2f}s"
