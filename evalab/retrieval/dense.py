"""Dense retrieval using sentence transformers and FAISS."""

import logging
import time
from typing import Any

import numpy as np

from evalab.retrieval.base import BaseRetriever, RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """
    Dense retriever using sentence transformers and FAISS.

    Encodes queries and documents into dense vectors and performs
    approximate nearest neighbor search.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        use_gpu: bool = False,
    ):
        """
        Initialize dense retriever.

        Args:
            model_name: Sentence transformer model name
            normalize_embeddings: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            use_gpu: Whether to use GPU for encoding/indexing
        """
        super().__init__(name="dense")
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self._model = None
        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None

    def _load_model(self) -> None:
        """Lazily load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if self.use_gpu else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Loaded sentence transformer: {self.model_name}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode

        Returns:
            Numpy array of embeddings
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 100,
        )

        return np.array(embeddings, dtype=np.float32)

    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build FAISS index from chunks.

        Args:
            chunks: List of chunk dictionaries
        """
        import faiss

        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        # Encode all chunks
        logger.info(f"Encoding {len(texts)} chunks...")
        self._embeddings = self._encode(texts)

        # Build FAISS index
        dim = self._embeddings.shape[1]

        if self.use_gpu:
            # GPU index
            res = faiss.StandardGpuResources()
            self._index = faiss.GpuIndexFlatIP(res, dim)
        else:
            # CPU index with inner product (for normalized embeddings = cosine similarity)
            self._index = faiss.IndexFlatIP(dim)

        self._index.add(self._embeddings)
        self._is_indexed = True

        logger.info(f"Built FAISS index with {self._index.ntotal} vectors (dim={dim})")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve chunks using dense similarity.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with ranked chunks
        """
        if not self._is_indexed or self._index is None:
            raise RuntimeError("Retriever not indexed. Call index() first.")

        start_time = time.time()

        # Encode query
        query_embedding = self._encode([query])

        # Search index
        scores, indices = self._index.search(query_embedding, top_k)

        # Build results
        retrieved_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for invalid results
                chunk = self._chunks[idx]
                retrieved_chunks.append(
                    RetrievedChunk(
                        doc_id=chunk["doc_id"],
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=float(score),
                        title=chunk.get("title"),
                        metadata=chunk.get("metadata", {}),
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query,
            chunks=retrieved_chunks,
            retriever_type="dense",
            latency_ms=latency_ms,
            metadata={"model": self.model_name},
        )

    def clear_index(self) -> None:
        """Clear the FAISS index."""
        self._index = None
        self._chunks = []
        self._embeddings = None
        self._is_indexed = False

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.

        Useful for computing similarities outside of retrieval.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return self._encode([text])[0]

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if self.normalize_embeddings:
            # Already normalized, just dot product
            return float(np.dot(emb1, emb2))
        else:
            # Compute cosine similarity
            return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
