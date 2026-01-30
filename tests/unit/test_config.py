"""Tests for configuration schemas."""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalab.config.schemas import (
    ChunkingConfig,
    DatasetConfig,
    GenerationConfig,
    RetrievalConfig,
    RetrieverType,
    RunConfig,
)


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_defaults(self):
        config = ChunkingConfig()
        assert config.size == 512
        assert config.overlap == 64

    def test_custom_values(self):
        config = ChunkingConfig(size=1024, overlap=128)
        assert config.size == 1024
        assert config.overlap == 128

    def test_invalid_size(self):
        with pytest.raises(ValidationError):
            ChunkingConfig(size=32)  # Below minimum

    def test_overlap_validation(self):
        # This should work - overlap < size
        config = ChunkingConfig(size=100, overlap=50)
        assert config.overlap == 50


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_defaults(self):
        config = GenerationConfig()
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.2
        assert config.max_tokens == 512

    def test_custom_model(self):
        config = GenerationConfig(model="gpt-4o", temperature=0.5)
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5

    def test_invalid_temperature(self):
        with pytest.raises(ValidationError):
            GenerationConfig(temperature=3.0)  # Max is 2.0


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_defaults(self):
        config = RetrievalConfig()
        assert config.enabled is False
        assert config.top_k == 5

    def test_hybrid_config(self):
        config = RetrievalConfig(
            enabled=True,
            retriever={"type": "hybrid", "bm25_weight": 0.3, "dense_weight": 0.7},
        )
        assert config.retriever.type == RetrieverType.HYBRID
        assert config.retriever.bm25_weight == 0.3


class TestDatasetConfig:
    """Tests for DatasetConfig."""

    def test_required_path(self):
        config = DatasetConfig(path="data/test.jsonl")
        assert config.path == "data/test.jsonl"

    def test_missing_path(self):
        with pytest.raises(ValidationError):
            DatasetConfig()  # path is required

    def test_max_examples(self):
        config = DatasetConfig(path="test.jsonl", max_examples=100)
        assert config.max_examples == 100


class TestRunConfig:
    """Tests for RunConfig."""

    def test_minimal_config(self):
        config = RunConfig(
            run_name="test_run",
            dataset={"path": "data/test.jsonl"},
            prompt={"template_path": "prompts/test.jinja"},
        )
        assert config.run_name == "test_run"
        assert config.dataset.path == "data/test.jsonl"

    def test_full_config(self):
        config = RunConfig(
            run_name="full_test",
            dataset={"path": "data/test.jsonl", "max_examples": 100},
            retrieval={"enabled": True, "corpus_path": "data/corpus.jsonl"},
            generation={"model": "gpt-4o", "temperature": 0.1},
            prompt={"template_path": "prompts/test.jinja"},
            evaluation={"suites": ["accuracy", "semantic"]},
            notes="Test run",
            tags=["test", "example"],
        )
        assert config.retrieval.enabled is True
        assert config.generation.model == "gpt-4o"
        assert "accuracy" in [s.value for s in config.evaluation.suites]

    def test_from_yaml(self):
        yaml_content = """
run_name: yaml_test
dataset:
  path: data/test.jsonl
prompt:
  template_path: prompts/test.jinja
generation:
  model: gpt-4o-mini
  temperature: 0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = RunConfig.from_yaml(f.name)

            assert config.run_name == "yaml_test"
            assert config.generation.model == "gpt-4o-mini"

    def test_to_yaml(self):
        config = RunConfig(
            run_name="test",
            dataset={"path": "test.jsonl"},
            prompt={"template_path": "test.jinja"},
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config.to_yaml(f.name)

            # Read back
            loaded = RunConfig.from_yaml(f.name)
            assert loaded.run_name == "test"
