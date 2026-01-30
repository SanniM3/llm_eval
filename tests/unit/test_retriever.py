"""Tests for retrieval system."""

import pytest

from evalab.retrieval.base import RetrievalResult, RetrievedChunk
from evalab.retrieval.bm25 import BM25Retriever


class TestBM25Retriever:
    """Tests for BM25 retriever."""

    @pytest.fixture
    def sample_chunks(self):
        return [
            {"chunk_id": "c1", "doc_id": "d1", "text": "Python is a programming language."},
            {"chunk_id": "c2", "doc_id": "d1", "text": "Java is also a programming language."},
            {"chunk_id": "c3", "doc_id": "d2", "text": "The weather is sunny today."},
            {"chunk_id": "c4", "doc_id": "d2", "text": "It might rain tomorrow."},
        ]

    def test_index_and_retrieve(self, sample_chunks):
        retriever = BM25Retriever()
        retriever.index(sample_chunks)

        assert retriever.is_indexed()

        result = retriever.retrieve("What is Python?", top_k=2)

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) <= 2
        assert result.retriever_type == "bm25"

    def test_retrieve_relevance(self, sample_chunks):
        retriever = BM25Retriever()
        retriever.index(sample_chunks)

        result = retriever.retrieve("programming language", top_k=2)

        # Top results should be about programming
        top_texts = [c.text for c in result.chunks]
        assert any("programming" in t for t in top_texts)

    def test_retrieve_before_index(self):
        retriever = BM25Retriever()

        with pytest.raises(RuntimeError):
            retriever.retrieve("test query")

    def test_clear_index(self, sample_chunks):
        retriever = BM25Retriever()
        retriever.index(sample_chunks)

        assert retriever.is_indexed()

        retriever.clear_index()

        assert not retriever.is_indexed()

    def test_empty_query(self, sample_chunks):
        retriever = BM25Retriever()
        retriever.index(sample_chunks)

        result = retriever.retrieve("", top_k=2)

        # Should return results but with low/zero scores
        assert isinstance(result, RetrievalResult)


class TestRetrievalResult:
    """Tests for RetrievalResult class."""

    def test_get_context(self):
        chunks = [
            RetrievedChunk(doc_id="d1", chunk_id="c1", text="First chunk.", score=0.9),
            RetrievedChunk(doc_id="d1", chunk_id="c2", text="Second chunk.", score=0.8),
        ]

        result = RetrievalResult(
            query="test",
            chunks=chunks,
            retriever_type="bm25",
            latency_ms=10.0,
        )

        context = result.get_context()
        assert "First chunk." in context
        assert "Second chunk." in context

    def test_doc_ids(self):
        chunks = [
            RetrievedChunk(doc_id="d1", chunk_id="c1", text="...", score=0.9),
            RetrievedChunk(doc_id="d2", chunk_id="c2", text="...", score=0.8),
            RetrievedChunk(doc_id="d1", chunk_id="c3", text="...", score=0.7),
        ]

        result = RetrievalResult(
            query="test",
            chunks=chunks,
            retriever_type="bm25",
            latency_ms=10.0,
        )

        # Should be unique and in order
        assert result.doc_ids == ["d1", "d2"]

    def test_top_chunk(self):
        chunks = [
            RetrievedChunk(doc_id="d1", chunk_id="c1", text="Top", score=0.9),
            RetrievedChunk(doc_id="d2", chunk_id="c2", text="Second", score=0.8),
        ]

        result = RetrievalResult(
            query="test",
            chunks=chunks,
            retriever_type="bm25",
            latency_ms=10.0,
        )

        assert result.top_chunk.text == "Top"

    def test_empty_result(self):
        result = RetrievalResult(
            query="test",
            chunks=[],
            retriever_type="bm25",
            latency_ms=10.0,
        )

        assert result.top_chunk is None
        assert result.get_context() == ""
        assert result.doc_ids == []
