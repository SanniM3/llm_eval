"""Hybrid retrieval combining BM25 and dense retrieval."""

import logging
import time
from typing import Any

from evalab.retrieval.base import BaseRetriever, RetrievalResult, RetrievedChunk
from evalab.retrieval.bm25 import BM25Retriever
from evalab.retrieval.dense import DenseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining BM25 and dense retrieval.

    Uses weighted score fusion to combine results from both retrievers.
    """

    def __init__(
        self,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        dense_model: str = "all-MiniLM-L6-v2",
        normalize_scores: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_weight: Weight for BM25 scores
            dense_weight: Weight for dense scores
            dense_model: Sentence transformer model name
            normalize_scores: Whether to normalize scores before fusion
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        super().__init__(name="hybrid")

        if abs(bm25_weight + dense_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights don't sum to 1.0 (bm25={bm25_weight}, dense={dense_weight})"
            )

        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.normalize_scores = normalize_scores

        # Initialize sub-retrievers
        self._bm25 = BM25Retriever(k1=k1, b=b)
        self._dense = DenseRetriever(model_name=dense_model)

        self._chunks: list[dict[str, Any]] = []

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Min-max normalize scores to [0, 1].

        Args:
            scores: Dictionary of chunk_id -> score

        Returns:
            Normalized scores
        """
        if not scores:
            return scores

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val - min_val < 1e-9:
            # All scores are the same
            return {k: 1.0 for k in scores}

        return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build indices for both retrievers.

        Args:
            chunks: List of chunk dictionaries
        """
        self._chunks = chunks

        logger.info("Building BM25 index...")
        self._bm25.index(chunks)

        logger.info("Building dense index...")
        self._dense.index(chunks)

        self._is_indexed = True
        logger.info("Hybrid index ready")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve chunks using hybrid fusion.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with ranked chunks
        """
        if not self._is_indexed:
            raise RuntimeError("Retriever not indexed. Call index() first.")

        start_time = time.time()

        # Get results from both retrievers (fetch more than top_k for fusion)
        fetch_k = min(top_k * 3, len(self._chunks))

        bm25_result = self._bm25.retrieve(query, top_k=fetch_k)
        dense_result = self._dense.retrieve(query, top_k=fetch_k)

        # Collect scores by chunk_id
        bm25_scores = {c.chunk_id: c.score for c in bm25_result.chunks}
        dense_scores = {c.chunk_id: c.score for c in dense_result.chunks}

        # Normalize if requested
        if self.normalize_scores:
            bm25_scores = self._normalize_scores(bm25_scores)
            dense_scores = self._normalize_scores(dense_scores)

        # Fuse scores
        all_chunk_ids = set(bm25_scores.keys()) | set(dense_scores.keys())
        fused_scores: dict[str, float] = {}

        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)
            fused_scores[chunk_id] = (
                self.bm25_weight * bm25_score + self.dense_weight * dense_score
            )

        # Sort by fused score and get top_k
        sorted_chunk_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:top_k]

        # Build chunk lookup
        chunk_lookup = {c["chunk_id"]: c for c in self._chunks}

        # Build results
        retrieved_chunks = []
        for chunk_id in top_chunk_ids:
            chunk = chunk_lookup.get(chunk_id)
            if chunk:
                retrieved_chunks.append(
                    RetrievedChunk(
                        doc_id=chunk["doc_id"],
                        chunk_id=chunk_id,
                        text=chunk["text"],
                        score=fused_scores[chunk_id],
                        title=chunk.get("title"),
                        metadata={
                            **chunk.get("metadata", {}),
                            "bm25_score": bm25_scores.get(chunk_id, 0.0),
                            "dense_score": dense_scores.get(chunk_id, 0.0),
                        },
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            retriever_type="hybrid",
            latency_ms=latency_ms,
            metadata={
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "bm25_latency_ms": bm25_result.latency_ms,
                "dense_latency_ms": dense_result.latency_ms,
            },
        )

    def clear_index(self) -> None:
        """Clear both indices."""
        self._bm25.clear_index()
        self._dense.clear_index()
        self._chunks = []
        self._is_indexed = False


def create_retriever(
    retriever_type: str,
    **kwargs: Any,
) -> BaseRetriever:
    """
    Factory function to create a retriever.

    Args:
        retriever_type: One of "bm25", "dense", "hybrid"
        **kwargs: Arguments passed to the retriever constructor

    Returns:
        Retriever instance
    """
    if retriever_type == "bm25":
        return BM25Retriever(**kwargs)
    elif retriever_type == "dense":
        return DenseRetriever(**kwargs)
    elif retriever_type == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
