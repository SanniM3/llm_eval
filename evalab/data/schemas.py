"""Pydantic schemas for dataset and corpus data."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported task types."""

    QA = "qa"
    RAG_QA = "rag_qa"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"


# ============================================================================
# Input schemas (task-specific)
# ============================================================================


class QAInput(BaseModel):
    """Input for QA task."""

    question: str = Field(..., description="The question to answer")
    context: str | None = Field(default=None, description="Optional context for the question")


class RAGQAInput(BaseModel):
    """Input for RAG-QA task (retrieval will provide context)."""

    question: str = Field(..., description="The question to answer")


class SummarizationInput(BaseModel):
    """Input for summarization task."""

    document: str = Field(..., description="The document to summarize")
    max_length: int | None = Field(default=None, description="Optional max summary length")


class ClassificationInput(BaseModel):
    """Input for classification task."""

    text: str = Field(..., description="The text to classify")
    labels: list[str] = Field(..., description="Available labels/classes")


# ============================================================================
# Reference schemas (ground truth)
# ============================================================================


class QAReference(BaseModel):
    """Reference answer for QA task."""

    answer: str = Field(..., description="The correct answer")
    aliases: list[str] = Field(default_factory=list, description="Alternative acceptable answers")


class SupportingFact(BaseModel):
    """A supporting fact from the corpus."""

    doc_id: str = Field(..., description="Document ID")
    span: str = Field(..., description="Relevant text span")


class RAGQAReference(BaseModel):
    """Reference for RAG-QA task with citations."""

    answer: str = Field(..., description="The correct answer")
    supporting_facts: list[SupportingFact] = Field(
        default_factory=list, description="Supporting evidence from corpus"
    )
    aliases: list[str] = Field(default_factory=list, description="Alternative acceptable answers")


class SummarizationReference(BaseModel):
    """Reference summary."""

    summary: str = Field(..., description="The reference summary")
    key_points: list[str] = Field(
        default_factory=list, description="Key points that should be covered"
    )


class ClassificationReference(BaseModel):
    """Reference classification label(s)."""

    label: str = Field(..., description="The correct label")
    secondary_labels: list[str] = Field(
        default_factory=list, description="Additional acceptable labels"
    )


# ============================================================================
# Example schema (unified)
# ============================================================================


class ExampleMetadata(BaseModel):
    """Metadata for an example."""

    domain: str | None = Field(default=None, description="Domain category")
    difficulty: str | None = Field(default=None, description="Difficulty level")
    language: str | None = Field(default="en", description="Language code")
    source: str | None = Field(default=None, description="Data source")
    tags: list[str] = Field(default_factory=list, description="Additional tags")
    extra: dict[str, Any] = Field(default_factory=dict, description="Extra metadata fields")


class Example(BaseModel):
    """A single evaluation example."""

    id: str = Field(..., description="Unique example identifier")
    task: TaskType = Field(..., description="Task type")
    input: dict[str, Any] = Field(..., description="Task-specific input")
    reference: dict[str, Any] = Field(..., description="Ground truth reference")
    metadata: ExampleMetadata = Field(default_factory=ExampleMetadata)

    def get_typed_input(self) -> QAInput | RAGQAInput | SummarizationInput | ClassificationInput:
        """Get input as typed model based on task."""
        if self.task == TaskType.QA:
            return QAInput(**self.input)
        elif self.task == TaskType.RAG_QA:
            return RAGQAInput(**self.input)
        elif self.task == TaskType.SUMMARIZATION:
            return SummarizationInput(**self.input)
        elif self.task == TaskType.CLASSIFICATION:
            return ClassificationInput(**self.input)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def get_typed_reference(
        self,
    ) -> QAReference | RAGQAReference | SummarizationReference | ClassificationReference:
        """Get reference as typed model based on task."""
        if self.task == TaskType.QA:
            return QAReference(**self.reference)
        elif self.task == TaskType.RAG_QA:
            return RAGQAReference(**self.reference)
        elif self.task == TaskType.SUMMARIZATION:
            return SummarizationReference(**self.reference)
        elif self.task == TaskType.CLASSIFICATION:
            return ClassificationReference(**self.reference)
        else:
            raise ValueError(f"Unknown task type: {self.task}")


# ============================================================================
# Corpus document schema
# ============================================================================


class CorpusDocument(BaseModel):
    """A document in the corpus for RAG."""

    doc_id: str = Field(..., description="Unique document identifier")
    title: str | None = Field(default=None, description="Document title")
    text: str = Field(..., description="Document content")
    source: str | None = Field(default=None, description="Source of the document")
    timestamp: str | None = Field(default=None, description="Document timestamp (ISO format)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================================
# Dataset wrapper
# ============================================================================


class Dataset(BaseModel):
    """A complete evaluation dataset."""

    name: str = Field(..., description="Dataset name")
    version: str = Field(default="1.0", description="Dataset version")
    description: str | None = Field(default=None, description="Dataset description")
    examples: list[Example] = Field(default_factory=list, description="List of examples")
    hash: str | None = Field(default=None, description="Content hash for versioning")

    @property
    def task_distribution(self) -> dict[TaskType, int]:
        """Get distribution of task types."""
        dist: dict[TaskType, int] = {}
        for ex in self.examples:
            dist[ex.task] = dist.get(ex.task, 0) + 1
        return dist

    def filter_by_task(self, task: TaskType) -> "Dataset":
        """Filter examples by task type."""
        return Dataset(
            name=f"{self.name}_{task.value}",
            version=self.version,
            description=self.description,
            examples=[ex for ex in self.examples if ex.task == task],
        )

    def filter_by_metadata(self, key: str, value: Any) -> "Dataset":
        """Filter examples by metadata field."""
        filtered = []
        for ex in self.examples:
            meta_dict = ex.metadata.model_dump()
            if key in meta_dict and meta_dict[key] == value:
                filtered.append(ex)
            elif key in ex.metadata.extra and ex.metadata.extra[key] == value:
                filtered.append(ex)
        return Dataset(
            name=f"{self.name}_{key}_{value}",
            version=self.version,
            description=self.description,
            examples=filtered,
        )


class Corpus(BaseModel):
    """A document corpus for RAG."""

    name: str = Field(..., description="Corpus name")
    documents: list[CorpusDocument] = Field(default_factory=list)
    hash: str | None = Field(default=None, description="Content hash for versioning")

    def get_document(self, doc_id: str) -> CorpusDocument | None:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None
