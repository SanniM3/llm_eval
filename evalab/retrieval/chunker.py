"""Text chunking utilities."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A text chunk from a document."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "title": self.title,
            "metadata": self.metadata,
        }


class Chunker:
    """
    Text chunker with configurable size and overlap.

    Supports character-based and sentence-aware chunking.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        respect_sentences: bool = True,
        min_chunk_size: int = 50,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            respect_sentences: Try to break at sentence boundaries
            min_chunk_size: Minimum chunk size (avoid tiny trailing chunks)
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self.min_chunk_size = min_chunk_size

        # Sentence boundary pattern
        self._sentence_pattern = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def _find_sentence_boundary(self, text: str, target_pos: int) -> int:
        """
        Find the nearest sentence boundary to the target position.

        Args:
            text: Text to search
            target_pos: Target position

        Returns:
            Position of nearest sentence boundary
        """
        if not self.respect_sentences:
            return target_pos

        # Search for sentence boundaries near target
        search_start = max(0, target_pos - 100)
        search_end = min(len(text), target_pos + 100)
        search_text = text[search_start:search_end]

        best_pos = target_pos
        best_dist = float("inf")

        for match in self._sentence_pattern.finditer(search_text):
            abs_pos = search_start + match.start()
            dist = abs(abs_pos - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_pos = abs_pos

        return best_pos

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            title: Optional document title
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects
        """
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        text = text.strip()
        text_len = len(text)

        if text_len <= self.chunk_size:
            # Text fits in a single chunk
            return [
                Chunk(
                    chunk_id=f"{doc_id}_0",
                    doc_id=doc_id,
                    text=text,
                    start_char=0,
                    end_char=text_len,
                    title=title,
                    metadata=metadata or {},
                )
            ]

        start = 0
        chunk_idx = 0

        while start < text_len:
            # Calculate end position
            end = start + self.chunk_size

            if end >= text_len:
                # Last chunk
                end = text_len
            else:
                # Try to find a good break point
                end = self._find_sentence_boundary(text, end)

            # Extract chunk text
            chunk_text = text[start:end].strip()

            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_text) >= self.min_chunk_size or start + self.chunk_size >= text_len:
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_{chunk_idx}",
                        doc_id=doc_id,
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                        title=title,
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1

            # Move to next chunk with overlap
            start = end - self.overlap

            # Avoid infinite loop
            if start >= text_len - self.min_chunk_size:
                break

        return chunks

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries with keys:
                - doc_id: Document identifier
                - text: Document content
                - title: Optional title
                - metadata: Optional metadata

        Returns:
            List of all chunks from all documents
        """
        all_chunks: list[Chunk] = []

        for doc in documents:
            chunks = self.chunk_text(
                text=doc.get("text", ""),
                doc_id=doc["doc_id"],
                title=doc.get("title"),
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        return all_chunks

    def __repr__(self) -> str:
        return f"Chunker(size={self.chunk_size}, overlap={self.overlap})"
