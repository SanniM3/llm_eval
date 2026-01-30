"""Data loading and validation for LLM-EvalLab."""

from evalab.data.schemas import (
    TaskType,
    Example,
    QAInput,
    QAReference,
    RAGQAInput,
    RAGQAReference,
    SummarizationInput,
    SummarizationReference,
    ClassificationInput,
    ClassificationReference,
    CorpusDocument,
)
from evalab.data.loader import load_dataset, load_corpus, validate_dataset
from evalab.data.versioning import compute_dataset_hash, get_dataset_id

__all__ = [
    "TaskType",
    "Example",
    "QAInput",
    "QAReference",
    "RAGQAInput",
    "RAGQAReference",
    "SummarizationInput",
    "SummarizationReference",
    "ClassificationInput",
    "ClassificationReference",
    "CorpusDocument",
    "load_dataset",
    "load_corpus",
    "validate_dataset",
    "compute_dataset_hash",
    "get_dataset_id",
]
