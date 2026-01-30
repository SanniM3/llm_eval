"""Semantic similarity evaluation suite."""

import logging
from typing import Any

import numpy as np

from evalab.evaluation.base import EvaluationSuite, MetricResult

logger = logging.getLogger(__name__)


class SemanticSuite(EvaluationSuite):
    """
    Semantic similarity evaluation suite.

    Computes:
    - BERTScore (precision, recall, F1)
    - Sentence embedding cosine similarity
    """

    def __init__(
        self,
        use_bertscore: bool = True,
        use_embeddings: bool = True,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
        embedding_model: str = "all-MiniLM-L6-v2",
        bertscore_batch_size: int = 32,
    ):
        """
        Initialize semantic suite.

        Args:
            use_bertscore: Whether to compute BERTScore
            use_embeddings: Whether to compute embedding similarity
            bertscore_model: Model for BERTScore
            embedding_model: Sentence transformer model
            bertscore_batch_size: Batch size for BERTScore
        """
        super().__init__(name="semantic")
        self.use_bertscore = use_bertscore
        self.use_embeddings = use_embeddings
        self.bertscore_model = bertscore_model
        self.embedding_model = embedding_model
        self.bertscore_batch_size = bertscore_batch_size

        # Lazy loading
        self._scorer = None
        self._embedder = None

    def _get_bertscore_scorer(self) -> Any:
        """Lazy load BERTScore scorer."""
        if self._scorer is None:
            from bert_score import BERTScorer

            self._scorer = BERTScorer(
                model_type=self.bertscore_model,
                batch_size=self.bertscore_batch_size,
                lang="en",
                rescale_with_baseline=True,
            )
            logger.info(f"Loaded BERTScore model: {self.bertscore_model}")
        return self._scorer

    def _get_embedder(self) -> Any:
        """Lazy load sentence transformer."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded embedding model: {self.embedding_model}")
        return self._embedder

    def _compute_bertscore(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[dict[str, float]]:
        """
        Compute BERTScore for a batch.

        Args:
            predictions: Model outputs
            references: Ground truths

        Returns:
            List of dicts with P, R, F1
        """
        scorer = self._get_bertscore_scorer()

        P, R, F1 = scorer.score(predictions, references)

        results = []
        for p, r, f1 in zip(P.tolist(), R.tolist(), F1.tolist()):
            results.append({
                "precision": p,
                "recall": r,
                "f1": f1,
            })

        return results

    def _compute_embedding_similarity(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[float]:
        """
        Compute cosine similarity of embeddings.

        Args:
            predictions: Model outputs
            references: Ground truths

        Returns:
            List of similarity scores
        """
        embedder = self._get_embedder()

        # Encode all texts
        pred_embeddings = embedder.encode(predictions, normalize_embeddings=True)
        ref_embeddings = embedder.encode(references, normalize_embeddings=True)

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = []
        for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
            sim = float(np.dot(pred_emb, ref_emb))
            similarities.append(sim)

        return similarities

    def evaluate(
        self,
        prediction: str,
        reference: str | dict[str, Any],
        **kwargs: Any,
    ) -> list[MetricResult]:
        """
        Evaluate semantic similarity.

        Args:
            prediction: Model output
            reference: Ground truth (string or dict with 'answer'/'summary')
            **kwargs: Additional context

        Returns:
            List of semantic metrics
        """
        # Handle reference format
        if isinstance(reference, dict):
            ref_text = reference.get("answer") or reference.get("summary", "")
        else:
            ref_text = reference

        results = []

        # Compute BERTScore
        if self.use_bertscore:
            bert_results = self._compute_bertscore([prediction], [ref_text])[0]
            results.append(
                MetricResult(
                    name="bert_score_f1",
                    value=bert_results["f1"],
                    details=bert_results,
                )
            )
            results.append(
                MetricResult(
                    name="bert_score_precision",
                    value=bert_results["precision"],
                )
            )
            results.append(
                MetricResult(
                    name="bert_score_recall",
                    value=bert_results["recall"],
                )
            )

        # Compute embedding similarity
        if self.use_embeddings:
            similarities = self._compute_embedding_similarity([prediction], [ref_text])
            results.append(
                MetricResult(
                    name="semantic_similarity",
                    value=similarities[0],
                )
            )

        return results

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str | dict[str, Any]],
        **kwargs: Any,
    ) -> list[list[MetricResult]]:
        """
        Evaluate a batch of predictions (optimized).

        Args:
            predictions: List of model outputs
            references: List of ground truths
            **kwargs: Additional context

        Returns:
            List of metric lists
        """
        # Extract reference texts
        ref_texts = []
        for ref in references:
            if isinstance(ref, dict):
                ref_texts.append(ref.get("answer") or ref.get("summary", ""))
            else:
                ref_texts.append(ref)

        # Compute batch metrics
        bert_results = None
        similarities = None

        if self.use_bertscore:
            bert_results = self._compute_bertscore(predictions, ref_texts)

        if self.use_embeddings:
            similarities = self._compute_embedding_similarity(predictions, ref_texts)

        # Build results
        all_results = []
        for i in range(len(predictions)):
            results = []

            if bert_results:
                results.append(
                    MetricResult(
                        name="bert_score_f1",
                        value=bert_results[i]["f1"],
                        details=bert_results[i],
                    )
                )
                results.append(
                    MetricResult(
                        name="bert_score_precision",
                        value=bert_results[i]["precision"],
                    )
                )
                results.append(
                    MetricResult(
                        name="bert_score_recall",
                        value=bert_results[i]["recall"],
                    )
                )

            if similarities:
                results.append(
                    MetricResult(
                        name="semantic_similarity",
                        value=similarities[i],
                    )
                )

            all_results.append(results)

        return all_results

    def get_metric_names(self) -> list[str]:
        names = []
        if self.use_bertscore:
            names.extend(["bert_score_f1", "bert_score_precision", "bert_score_recall"])
        if self.use_embeddings:
            names.append("semantic_similarity")
        return names
