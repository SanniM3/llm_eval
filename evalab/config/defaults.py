"""Default configuration values and pricing tables."""

# Default embedding model for dense retrieval
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Pricing table for cost estimation (per 1M tokens)
# Updated as of 2025
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    # Default fallback for unknown models (conservative estimate)
    "default": {"input": 1.00, "output": 3.00},
}

# Default supported task types
SUPPORTED_TASKS = ["qa", "rag_qa", "summarization", "classification"]

# Default metric aggregation functions
DEFAULT_AGGREGATIONS = {
    "mean": "Average across all examples",
    "std": "Standard deviation",
    "min": "Minimum value",
    "max": "Maximum value",
    "p50": "50th percentile (median)",
    "p95": "95th percentile",
    "p99": "99th percentile",
}

# Default perturbation types for robustness evaluation
DEFAULT_PERTURBATIONS = [
    "paraphrase",  # Paraphrase the question using LLM
    "distractor",  # Add distractor sentence to context
    "drop_top_k",  # Drop top-1 retrieved document
    "typos",  # Add minor typos to input
]

# Metric descriptions
METRIC_DESCRIPTIONS = {
    # Accuracy metrics
    "exact_match": "Exact string match between prediction and reference",
    "token_f1": "Token-level F1 score (SQuAD-style)",
    "accuracy": "Classification accuracy",
    "macro_f1": "Macro-averaged F1 score",
    # Semantic metrics
    "bert_score_f1": "BERTScore F1",
    "bert_score_precision": "BERTScore Precision",
    "bert_score_recall": "BERTScore Recall",
    "semantic_similarity": "Cosine similarity of sentence embeddings",
    # Faithfulness metrics
    "faithfulness_score": "Overall faithfulness score (0-1)",
    "unsupported_claim_rate": "Rate of claims not supported by context",
    "citation_precision": "Precision of cited documents",
    "contradiction_rate": "Rate of contradicted claims",
    # Robustness metrics
    "output_consistency": "Semantic similarity under perturbation",
    "decision_flip_rate": "Rate of changed predictions under perturbation",
    "faithfulness_delta": "Change in faithfulness under perturbation",
    # Calibration metrics
    "ece": "Expected Calibration Error",
    "confidence_accuracy_corr": "Correlation between confidence and correctness",
    "mean_confidence": "Average model confidence",
    # Cost/Latency metrics
    "total_input_tokens": "Total input tokens",
    "total_output_tokens": "Total output tokens",
    "mean_latency_ms": "Mean latency in milliseconds",
    "p50_latency_ms": "Median latency",
    "p95_latency_ms": "95th percentile latency",
    "p99_latency_ms": "99th percentile latency",
    "estimated_cost_usd": "Estimated cost in USD",
    "throughput_req_per_sec": "Requests per second",
}
