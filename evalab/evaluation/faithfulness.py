"""Faithfulness and grounding evaluation suite."""

import json
import logging
import re
from typing import Any

import numpy as np

from evalab.evaluation.base import EvaluationSuite, MetricResult

logger = logging.getLogger(__name__)


class FaithfulnessSuite(EvaluationSuite):
    """
    Faithfulness evaluation suite for RAG outputs.

    Computes:
    - Unsupported claim rate (using embedding similarity)
    - Citation precision (how many citations are accurate)
    - Faithfulness score (overall grounding quality)
    - LLM-judge faithfulness (optional, more accurate but slower)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_llm_judge: bool = False,
        judge_model: str | None = None,
        judge_samples: int = 3,
    ):
        """
        Initialize faithfulness suite.

        Args:
            similarity_threshold: Threshold for claim support (0-1)
            embedding_model: Model for similarity computation
            use_llm_judge: Whether to use LLM as judge
            judge_model: Model to use as judge (if use_llm_judge)
            judge_samples: Number of judge samples for voting
        """
        super().__init__(name="faithfulness")
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.use_llm_judge = use_llm_judge
        self.judge_model = judge_model
        self.judge_samples = judge_samples

        self._embedder = None
        self._llm_backend = None

    def _get_embedder(self) -> Any:
        """Lazy load sentence transformer."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _compute_claim_support(
        self,
        claims: list[str],
        context: str,
    ) -> list[dict[str, Any]]:
        """
        Check if claims are supported by context using embeddings.

        Args:
            claims: List of claim sentences
            context: Retrieved context text

        Returns:
            List of dicts with claim support info
        """
        if not claims or not context:
            return []

        embedder = self._get_embedder()

        # Split context into sentences
        context_sentences = self._split_into_sentences(context)
        if not context_sentences:
            return [
                {"claim": c, "supported": False, "max_similarity": 0.0, "best_match": ""}
                for c in claims
            ]

        # Encode all texts
        claim_embeddings = embedder.encode(claims, normalize_embeddings=True)
        context_embeddings = embedder.encode(context_sentences, normalize_embeddings=True)

        results = []
        for i, claim in enumerate(claims):
            # Compute similarity with all context sentences
            similarities = np.dot(context_embeddings, claim_embeddings[i])
            max_idx = np.argmax(similarities)
            max_sim = float(similarities[max_idx])

            results.append({
                "claim": claim,
                "supported": max_sim >= self.similarity_threshold,
                "max_similarity": max_sim,
                "best_match": context_sentences[max_idx],
            })

        return results

    def _extract_citations(self, text: str) -> list[int]:
        """
        Extract citation numbers from text.

        Looks for patterns like [1], [Doc 2], (Source 3), etc.

        Args:
            text: Response text

        Returns:
            List of cited numbers
        """
        patterns = [
            r"\[(\d+)\]",  # [1]
            r"\[Doc\s*(\d+)\]",  # [Doc 1]
            r"\[Source\s*(\d+)\]",  # [Source 1]
            r"\((\d+)\)",  # (1)
        ]

        citations = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.update(int(m) for m in matches)

        return sorted(citations)

    def _compute_citation_precision(
        self,
        response: str,
        context_chunks: list[dict[str, Any]],
        reference_facts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Compute citation precision.

        Args:
            response: Model response
            context_chunks: Retrieved context chunks
            reference_facts: Optional reference supporting facts

        Returns:
            Dict with citation metrics
        """
        cited_indices = self._extract_citations(response)

        if not cited_indices:
            return {
                "num_citations": 0,
                "valid_citations": 0,
                "precision": 0.0,
            }

        # Check if citations are within range
        valid = [i for i in cited_indices if 1 <= i <= len(context_chunks)]

        return {
            "num_citations": len(cited_indices),
            "valid_citations": len(valid),
            "precision": len(valid) / len(cited_indices) if cited_indices else 0.0,
            "cited_indices": cited_indices,
        }

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate faithfulness of prediction to context.

        Args:
            prediction: Model output
            reference: Reference answer (not used for faithfulness)
            **kwargs: Must include 'context' or 'chunks'

        Returns:
            List of faithfulness metrics
        """
        # Get context
        context = kwargs.get("context", "")
        chunks = kwargs.get("chunks", [])

        if not context and chunks:
            context = "\n\n".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in chunks
            )

        if not context:
            logger.warning("No context provided for faithfulness evaluation")
            return [
                MetricResult(name="faithfulness_score", value=0.0, details={"error": "no context"})
            ]

        results = []

        # Split prediction into claims
        claims = self._split_into_sentences(prediction)

        # Compute claim support
        claim_support = self._compute_claim_support(claims, context)

        # Calculate metrics
        if claim_support:
            supported_count = sum(1 for c in claim_support if c["supported"])
            unsupported_rate = 1.0 - (supported_count / len(claim_support))
            faithfulness_score = supported_count / len(claim_support)

            avg_similarity = np.mean([c["max_similarity"] for c in claim_support])
        else:
            supported_count = 0
            unsupported_rate = 1.0
            faithfulness_score = 0.0
            avg_similarity = 0.0

        results.append(
            MetricResult(
                name="faithfulness_score",
                value=faithfulness_score,
                details={
                    "supported_claims": supported_count,
                    "total_claims": len(claim_support),
                    "threshold": self.similarity_threshold,
                },
            )
        )

        results.append(
            MetricResult(
                name="unsupported_claim_rate",
                value=unsupported_rate,
                details={"claim_support": claim_support},
            )
        )

        results.append(
            MetricResult(
                name="avg_claim_similarity",
                value=float(avg_similarity),
            )
        )

        # Citation precision
        if chunks:
            citation_info = self._compute_citation_precision(prediction, chunks)
            results.append(
                MetricResult(
                    name="citation_precision",
                    value=citation_info["precision"],
                    details=citation_info,
                )
            )

        return results

    def get_metric_names(self) -> list[str]:
        return [
            "faithfulness_score",
            "unsupported_claim_rate",
            "avg_claim_similarity",
            "citation_precision",
        ]


class LLMJudgeFaithfulness(EvaluationSuite):
    """
    LLM-as-judge faithfulness evaluation.

    Uses an LLM to evaluate claim support with higher accuracy
    but at the cost of additional API calls.
    """

    def __init__(
        self,
        llm_backend: Any,
        judge_template: str | None = None,
        num_samples: int = 3,
        temperature: float = 0.3,
    ):
        """
        Initialize LLM judge.

        Args:
            llm_backend: LLM backend for judge calls
            judge_template: Optional custom judge template
            num_samples: Number of samples for voting
            temperature: Sampling temperature
        """
        super().__init__(name="llm_judge_faithfulness")
        self.llm = llm_backend
        self.num_samples = num_samples
        self.temperature = temperature

        # Default judge template
        self.judge_template = judge_template or """You are a faithfulness evaluator. Determine if the response is faithful to the provided context.

Context:
{context}

Response:
{response}

Evaluate each claim in the response:
1. Is it SUPPORTED by the context?
2. Is it NOT_FOUND in the context?
3. Does it CONTRADICT the context?

Respond in JSON format:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED|NOT_FOUND|CONTRADICTED"}}], "faithfulness_score": 0.0-1.0}}"""

    def _parse_judge_response(self, response: str) -> dict[str, Any]:
        """Parse JSON from judge response."""
        # Try to extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback
        return {"claims": [], "faithfulness_score": 0.5}

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate faithfulness using LLM judge.

        Args:
            prediction: Model output
            reference: Reference (not used)
            **kwargs: Must include 'context'

        Returns:
            List of faithfulness metrics
        """
        context = kwargs.get("context", "")
        if not context:
            return [MetricResult(name="llm_faithfulness", value=0.0)]

        # Generate multiple judge samples
        scores = []
        all_claims = []

        for _ in range(self.num_samples):
            prompt = self.judge_template.format(context=context, response=prediction)
            result = self.llm.generate(prompt, temperature=self.temperature)

            if not result.is_error:
                parsed = self._parse_judge_response(result.text)
                scores.append(parsed.get("faithfulness_score", 0.5))
                all_claims.append(parsed.get("claims", []))

        if not scores:
            return [MetricResult(name="llm_faithfulness", value=0.0, details={"error": "no valid responses"})]

        # Use median score (more robust than mean)
        final_score = float(np.median(scores))

        return [
            MetricResult(
                name="llm_faithfulness",
                value=final_score,
                details={
                    "all_scores": scores,
                    "num_samples": len(scores),
                },
            )
        ]

    def get_metric_names(self) -> list[str]:
        return ["llm_faithfulness"]
