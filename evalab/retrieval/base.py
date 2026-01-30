"""Base classes for retrieval system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""

    doc_id: str
    chunk_id: str
    text: str
    score: float
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "title": self.title,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    chunks: list[RetrievedChunk]
    retriever_type: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_chunk(self) -> RetrievedChunk | None:
        """Get the top-scoring chunk."""
        return self.chunks[0] if self.chunks else None

    @property
    def doc_ids(self) -> list[str]:
        """Get unique document IDs."""
        return list(dict.fromkeys(c.doc_id for c in self.chunks))

    def get_context(self, separator: str = "\n\n") -> str:
        """
        Combine chunks into a single context string.

        Args:
            separator: Text to insert between chunks

        Returns:
            Combined context text
        """
        return separator.join(c.text for c in self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "query": self.query,
            "chunks": [c.to_dict() for c in self.chunks],
            "retriever_type": self.retriever_type,
            "latency_ms": self.latency_ms,
            "num_chunks": len(self.chunks),
            "doc_ids": self.doc_ids,
            "metadata": self.metadata,
        }


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, name: str = "base"):
        """
        Initialize retriever.

        Args:
            name: Retriever identifier
        """
        self.name = name
        self._is_indexed = False

    @abstractmethod
    def index(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build the retrieval index from chunks.

        Args:
            chunks: List of chunk dictionaries with keys:
                - chunk_id: Unique chunk identifier
                - doc_id: Parent document ID
                - text: Chunk text content
                - title: Optional document title
                - metadata: Optional metadata dict
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult with ranked chunks
        """
        pass

    def is_indexed(self) -> bool:
        """Check if the retriever has been indexed."""
        return self._is_indexed

    def clear_index(self) -> None:
        """Clear the retrieval index."""
        self._is_indexed = False
