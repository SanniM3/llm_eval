"""Error taxonomy and classification for attribution."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of errors in LLM outputs."""

    RETRIEVAL_FAILURE = "retrieval_failure"  # No supporting evidence retrieved
    GENERATION_FAILURE = "generation_failure"  # Evidence present but ignored/contradicted
    PROMPT_FAILURE = "prompt_failure"  # Invalid format, incomplete response
    DATA_AMBIGUITY = "data_ambiguity"  # Reference uncertain or multi-answer
    CORRECT = "correct"  # No error


@dataclass
class ErrorAttribution:
    """Attribution result for a single example."""

    example_id: str
    error_type: ErrorType
    is_correct: bool
    confidence: float  # Confidence in the attribution
    details: dict[str, Any] = field(default_factory=dict)
    evidence: str | None = None  # Supporting evidence for attribution

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "example_id": self.example_id,
            "error_type": self.error_type.value,
            "is_correct": self.is_correct,
            "confidence": self.confidence,
            "details": self.details,
            "evidence": self.evidence,
        }


class ErrorClassifier:
    """
    Classifies errors using heuristic rules.

    Attribution heuristics:
    1. If answer incorrect AND retrieved context lacks supporting evidence -> retrieval_failure
    2. If evidence exists (high similarity) but output unsupported -> generation_failure
    3. If format invalid -> prompt_failure
    4. If reference has multiple valid answers -> data_ambiguity
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize error classifier.

        Args:
            similarity_threshold: Threshold for evidence support
            embedding_model: Model for similarity computation
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._embedder = None

    def _get_embedder(self) -> Any:
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embedder = self._get_embedder()
        embeddings = embedder.encode([text1, text2], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    def _check_evidence_support(
        self,
        reference_answer: str,
        context: str,
    ) -> tuple[bool, float, str]:
        """
        Check if reference answer is supported by context.

        Args:
            reference_answer: Ground truth answer
            context: Retrieved context

        Returns:
            Tuple of (is_supported, max_similarity, best_matching_text)
        """
        if not context:
            return False, 0.0, ""

        embedder = self._get_embedder()

        # Split context into sentences
        import re

        context_sentences = [s.strip() for s in re.split(r"[.!?]+", context) if s.strip()]

        if not context_sentences:
            return False, 0.0, ""

        # Compute similarities
        ref_embedding = embedder.encode([reference_answer], normalize_embeddings=True)[0]
        context_embeddings = embedder.encode(context_sentences, normalize_embeddings=True)

        similarities = np.dot(context_embeddings, ref_embedding)
        max_idx = np.argmax(similarities)
        max_sim = float(similarities[max_idx])

        is_supported = max_sim >= self.similarity_threshold

        return is_supported, max_sim, context_sentences[max_idx]

    def _check_format_validity(self, prediction: str, task_type: str) -> bool:
        """
        Check if prediction has valid format.

        Args:
            prediction: Model output
            task_type: Task type

        Returns:
            True if format is valid
        """
        prediction = prediction.strip()

        # Empty response is always invalid
        if not prediction:
            return False

        # Check for common format issues
        if task_type == "classification":
            # Should be a short label
            if len(prediction.split()) > 5:
                return False

        if task_type == "qa" or task_type == "rag_qa":
            # Check for refusal patterns
            refusal_patterns = [
                "i cannot",
                "i don't have",
                "i'm unable",
                "as an ai",
                "i apologize",
            ]
            lower_pred = prediction.lower()
            for pattern in refusal_patterns:
                if pattern in lower_pred:
                    return False

        return True

    def _check_reference_ambiguity(
        self,
        reference: dict[str, Any] | str,
    ) -> bool:
        """
        Check if reference has ambiguity.

        Args:
            reference: Ground truth

        Returns:
            True if reference is ambiguous
        """
        if isinstance(reference, dict):
            aliases = reference.get("aliases", [])
            secondary_labels = reference.get("secondary_labels", [])

            # Multiple valid answers suggest ambiguity
            if len(aliases) > 2 or len(secondary_labels) > 1:
                return True

        return False

    def classify(
        self,
        example_id: str,
        prediction: str,
        reference: str | dict[str, Any],
        is_correct: bool,
        context: str | None = None,
        task_type: str = "qa",
    ) -> ErrorAttribution:
        """
        Classify the error type for an example.

        Args:
            example_id: Example identifier
            prediction: Model output
            reference: Ground truth
            is_correct: Whether prediction is correct
            context: Retrieved context (for RAG tasks)
            task_type: Task type

        Returns:
            ErrorAttribution with classified error type
        """
        # If correct, no error
        if is_correct:
            return ErrorAttribution(
                example_id=example_id,
                error_type=ErrorType.CORRECT,
                is_correct=True,
                confidence=1.0,
            )

        # Get reference answer
        if isinstance(reference, dict):
            ref_answer = reference.get("answer") or reference.get("label", "")
        else:
            ref_answer = str(reference)

        details: dict[str, Any] = {}
        evidence = None

        # Check format validity
        if not self._check_format_validity(prediction, task_type):
            return ErrorAttribution(
                example_id=example_id,
                error_type=ErrorType.PROMPT_FAILURE,
                is_correct=False,
                confidence=0.9,
                details={"reason": "invalid_format"},
                evidence=f"Prediction: {prediction[:100]}...",
            )

        # Check reference ambiguity
        if self._check_reference_ambiguity(reference):
            details["ambiguous_reference"] = True

        # For RAG tasks, check retrieval quality
        if context and task_type in ("rag_qa", "qa"):
            is_supported, max_sim, best_match = self._check_evidence_support(ref_answer, context)

            details["evidence_similarity"] = max_sim
            details["evidence_supported"] = is_supported

            if not is_supported:
                # Retrieval failure - evidence not in context
                return ErrorAttribution(
                    example_id=example_id,
                    error_type=ErrorType.RETRIEVAL_FAILURE,
                    is_correct=False,
                    confidence=0.8,
                    details=details,
                    evidence=f"Max similarity to reference: {max_sim:.3f}",
                )
            else:
                # Evidence exists but model failed -> generation failure
                # Also check if output is supported by context
                output_supported, out_sim, _ = self._check_evidence_support(prediction, context)
                details["output_similarity"] = out_sim

                if not output_supported:
                    return ErrorAttribution(
                        example_id=example_id,
                        error_type=ErrorType.GENERATION_FAILURE,
                        is_correct=False,
                        confidence=0.8,
                        details=details,
                        evidence=f"Evidence exists (sim={max_sim:.3f}) but output not grounded (sim={out_sim:.3f})",
                    )

        # Check for data ambiguity
        if details.get("ambiguous_reference"):
            return ErrorAttribution(
                example_id=example_id,
                error_type=ErrorType.DATA_AMBIGUITY,
                is_correct=False,
                confidence=0.6,
                details=details,
            )

        # Default to generation failure
        return ErrorAttribution(
            example_id=example_id,
            error_type=ErrorType.GENERATION_FAILURE,
            is_correct=False,
            confidence=0.7,
            details=details,
        )

    def classify_batch(
        self,
        examples: list[dict[str, Any]],
    ) -> list[ErrorAttribution]:
        """
        Classify errors for a batch of examples.

        Args:
            examples: List of dicts with keys:
                - example_id, prediction, reference, is_correct, context, task_type

        Returns:
            List of ErrorAttribution objects
        """
        return [
            self.classify(
                example_id=ex["example_id"],
                prediction=ex["prediction"],
                reference=ex["reference"],
                is_correct=ex["is_correct"],
                context=ex.get("context"),
                task_type=ex.get("task_type", "qa"),
            )
            for ex in examples
        ]

    def summarize_attributions(
        self,
        attributions: list[ErrorAttribution],
    ) -> dict[str, Any]:
        """
        Summarize error attributions.

        Args:
            attributions: List of attributions

        Returns:
            Summary statistics
        """
        total = len(attributions)
        if total == 0:
            return {"total": 0}

        error_counts = {}
        for attr in attributions:
            error_type = attr.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        correct_count = error_counts.get("correct", 0)
        error_count = total - correct_count

        return {
            "total": total,
            "correct": correct_count,
            "errors": error_count,
            "accuracy": correct_count / total,
            "error_distribution": {
                k: v / total for k, v in error_counts.items() if k != "correct"
            },
            "error_counts": error_counts,
        }
