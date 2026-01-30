"""Pytest configuration and fixtures."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_qa_examples():
    """Sample QA examples for testing."""
    return [
        {
            "id": "qa_001",
            "task": "qa",
            "input": {"question": "What is 2+2?", "context": "Basic math: 2+2=4"},
            "reference": {"answer": "4", "aliases": ["four"]},
            "metadata": {"domain": "math", "difficulty": "easy"},
        },
        {
            "id": "qa_002",
            "task": "qa",
            "input": {"question": "What color is the sky?", "context": "The sky appears blue."},
            "reference": {"answer": "blue"},
            "metadata": {"domain": "science", "difficulty": "easy"},
        },
    ]


@pytest.fixture
def sample_dataset_path(temp_dir, sample_qa_examples):
    """Create a sample dataset JSONL file."""
    dataset_path = temp_dir / "test_dataset.jsonl"
    with open(dataset_path, "w") as f:
        for example in sample_qa_examples:
            f.write(json.dumps(example) + "\n")
    return dataset_path


@pytest.fixture
def sample_corpus_docs():
    """Sample corpus documents for testing."""
    return [
        {
            "doc_id": "doc_001",
            "title": "Math Basics",
            "text": "Addition is a basic math operation. 2+2=4. 3+3=6.",
        },
        {
            "doc_id": "doc_002",
            "title": "Science Facts",
            "text": "The sky appears blue due to Rayleigh scattering. The sun is yellow.",
        },
    ]


@pytest.fixture
def sample_corpus_path(temp_dir, sample_corpus_docs):
    """Create a sample corpus JSONL file."""
    corpus_path = temp_dir / "test_corpus.jsonl"
    with open(corpus_path, "w") as f:
        for doc in sample_corpus_docs:
            f.write(json.dumps(doc) + "\n")
    return corpus_path


@pytest.fixture
def sample_run_config(temp_dir, sample_dataset_path):
    """Create a sample run configuration."""
    # Create a simple prompt template
    prompt_path = temp_dir / "prompt.jinja"
    prompt_path.write_text("Q: {{ question }}\nA:")

    return {
        "run_name": "test_run",
        "dataset": {"path": str(sample_dataset_path)},
        "generation": {"backend": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
        "prompt": {"template_path": str(prompt_path)},
        "evaluation": {"suites": ["accuracy"]},
        "attribution": {"enabled": False},
        "logging": {"save_traces": False, "output_dir": str(temp_dir / "runs")},
    }
