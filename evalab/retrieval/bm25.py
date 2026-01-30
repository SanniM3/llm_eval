"""BM25 sparse retrieval."""

import re
import time
from typing import Any

from rank_bm25 import BM25Okapi

from evalab.retrieval.base import BaseRetriever, RetrievalResult, RetrievedChunk


class BM25Retriever(BaseRetriever):
    """
    BM25-based sparse retriever.

    Uses the Okapi BM25 algorithm for lexical matching.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        lowercase: bool = True,
        remove_stopwords: bool = False,
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            lowercase: Whether to lowercase tokens
            remove_stopwords: Whether to remove English stopwords
        """
        super().__init__(name="bm25")
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []
        self._tokenized_corpus: list[list[str]] = []

        # Simple stopwords list
        self._stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "this", "that",
            "these", "those", "it", "its", "as", "if", "when", "than", "so",
        }

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()

        # Simple word tokenization
        tokens = re.findall(r"\b\w+\b", text)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        return tokens

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        self._chunks = chunks
        self._tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]

        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve chunks using BM25.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with ranked chunks
        """
        if not self._is_indexed or self._bm25 is None:
            raise RuntimeError("Retriever not indexed. Call index() first.")

        start_time = time.time()

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Build results
        retrieved_chunks = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                chunk = self._chunks[idx]
                retrieved_chunks.append(
                    RetrievedChunk(
                        doc_id=chunk["doc_id"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=float(scores[idx]),
                        title=chunk.get("title"),
                        metadata=chunk.get("metadata", {}),
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            retriever_type="bm25",
            latency_ms=latency_ms,
            metadata={"k1": self.k1, "b": self.b},
        )

    def clear_index(self) -> None:
        """Clear the BM25 index."""
        self._bm25 = None
        self._chunks = []
        self._tokenized_corpus = []
        self._is_indexed = False
