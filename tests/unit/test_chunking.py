"""Tests for text chunking."""

import pytest

from evalab.retrieval.chunker import Chunk, Chunker


class TestChunker:
    """Tests for Chunker class."""

    def test_single_chunk_short_text(self):
        chunker = Chunker(chunk_size=1000, overlap=100)
        text = "This is a short text."

        chunks = chunker.chunk_text(text, doc_id="doc1")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].doc_id == "doc1"
        assert chunks[0].chunk_id == "doc1_0"

    def test_multiple_chunks(self):
        chunker = Chunker(chunk_size=50, overlap=10, respect_sentences=False)
        text = "A" * 100  # 100 characters

        chunks = chunker.chunk_text(text, doc_id="doc1")

        assert len(chunks) >= 2

    def test_overlap_preserved(self):
        chunker = Chunker(chunk_size=50, overlap=20, respect_sentences=False)
        text = "ABCDEFGHIJ" * 10  # 100 characters

        chunks = chunker.chunk_text(text, doc_id="doc1")

        # Check that consecutive chunks overlap
        if len(chunks) >= 2:
            end_of_first = chunks[0].text[-20:]
            start_of_second = chunks[1].text[:20]
            # There should be some overlap
            assert len(set(end_of_first) & set(start_of_second)) > 0

    def test_empty_text(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("", doc_id="doc1")
        assert len(chunks) == 0

    def test_whitespace_only(self):
        chunker = Chunker()
        chunks = chunker.chunk_text("   \n\t  ", doc_id="doc1")
        assert len(chunks) == 0

    def test_chunk_metadata(self):
        chunker = Chunker()
        metadata = {"source": "test", "page": 1}

        chunks = chunker.chunk_text(
            "Test text",
            doc_id="doc1",
            title="Test Doc",
            metadata=metadata,
        )

        assert chunks[0].title == "Test Doc"
        assert chunks[0].metadata == metadata

    def test_chunk_documents(self):
        chunker = Chunker(chunk_size=100)
        docs = [
            {"doc_id": "doc1", "text": "First document text."},
            {"doc_id": "doc2", "text": "Second document text."},
        ]

        chunks = chunker.chunk_documents(docs)

        assert len(chunks) >= 2
        doc_ids = {c.doc_id for c in chunks}
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, overlap=100)  # overlap must be < chunk_size

    def test_chunk_positions(self):
        chunker = Chunker(chunk_size=50, overlap=10, respect_sentences=False)
        text = "0123456789" * 10  # 100 chars

        chunks = chunker.chunk_text(text, doc_id="doc1")

        for chunk in chunks:
            # Verify positions match actual text
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_to_dict(self):
        chunk = Chunk(
            chunk_id="doc1_0",
            doc_id="doc1",
            text="Test text",
            start_char=0,
            end_char=9,
            title="Test",
            metadata={"key": "value"},
        )

        d = chunk.to_dict()

        assert d["chunk_id"] == "doc1_0"
        assert d["doc_id"] == "doc1"
        assert d["text"] == "Test text"
        assert d["title"] == "Test"
        assert d["metadata"] == {"key": "value"}
