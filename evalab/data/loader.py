"""Data loading utilities for datasets and corpus."""

import json
import logging
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from evalab.data.schemas import Corpus, CorpusDocument, Dataset, Example
from evalab.data.versioning import compute_dataset_hash

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Exception raised when data loading fails."""

    pass


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """
    Load a JSONL file and yield each line as a dictionary.

    Args:
        path: Path to the JSONL file

    Yields:
        Parsed JSON objects from each line

    Raises:
        DataLoadError: If file cannot be read or parsed
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise DataLoadError(f"Invalid JSON on line {line_num} in {path}: {e}")
    except IOError as e:
        raise DataLoadError(f"Failed to read file {path}: {e}")


def load_dataset(
    path: str | Path,
    name: str | None = None,
    max_examples: int | None = None,
    validate: bool = True,
) -> Dataset:
    """
    Load a dataset from a JSONL file.

    Each line in the file should be a JSON object matching the Example schema.

    Args:
        path: Path to the dataset JSONL file
        name: Optional name for the dataset (defaults to filename)
        max_examples: Maximum number of examples to load (None for all)
        validate: Whether to validate examples against schema

    Returns:
        Dataset object with loaded examples

    Raises:
        DataLoadError: If loading or validation fails
    """
    path = Path(path)
    dataset_name = name or path.stem

    examples: list[Example] = []
    errors: list[str] = []

    for idx, data in enumerate(load_jsonl(path)):
        if max_examples is not None and idx >= max_examples:
            break

        try:
            if validate:
                example = Example(**data)
            else:
                # Minimal validation
                example = Example.model_construct(**data)
            examples.append(example)
        except ValidationError as e:
            error_msg = f"Example {idx + 1}: {e}"
            errors.append(error_msg)
            if validate:
                logger.warning(f"Validation error: {error_msg}")

    if errors and validate:
        logger.warning(f"Dataset '{dataset_name}' had {len(errors)} validation errors")

    # Compute content hash
    content_hash = compute_dataset_hash(path)

    dataset = Dataset(
        name=dataset_name,
        examples=examples,
        hash=content_hash,
    )

    logger.info(
        f"Loaded dataset '{dataset_name}' with {len(examples)} examples "
        f"(hash: {content_hash[:12]}...)"
    )

    return dataset


def load_corpus(
    path: str | Path,
    name: str | None = None,
    validate: bool = True,
) -> Corpus:
    """
    Load a corpus from a JSONL file.

    Each line in the file should be a JSON object matching the CorpusDocument schema.

    Args:
        path: Path to the corpus JSONL file
        name: Optional name for the corpus (defaults to filename)
        validate: Whether to validate documents against schema

    Returns:
        Corpus object with loaded documents

    Raises:
        DataLoadError: If loading or validation fails
    """
    path = Path(path)
    corpus_name = name or path.stem

    documents: list[CorpusDocument] = []
    errors: list[str] = []

    for idx, data in enumerate(load_jsonl(path)):
        try:
            if validate:
                doc = CorpusDocument(**data)
            else:
                doc = CorpusDocument.model_construct(**data)
            documents.append(doc)
        except ValidationError as e:
            error_msg = f"Document {idx + 1}: {e}"
            errors.append(error_msg)
            if validate:
                logger.warning(f"Validation error: {error_msg}")

    if errors and validate:
        logger.warning(f"Corpus '{corpus_name}' had {len(errors)} validation errors")

    # Compute content hash
    content_hash = compute_dataset_hash(path)

    corpus = Corpus(
        name=corpus_name,
        documents=documents,
        hash=content_hash,
    )

    logger.info(
        f"Loaded corpus '{corpus_name}' with {len(documents)} documents "
        f"(hash: {content_hash[:12]}...)"
    )

    return corpus


def validate_dataset(dataset: Dataset) -> list[str]:
    """
    Validate a dataset for consistency and completeness.

    Args:
        dataset: Dataset to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors: list[str] = []

    # Check for duplicate IDs
    ids = [ex.id for ex in dataset.examples]
    if len(ids) != len(set(ids)):
        duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
        errors.append(f"Duplicate example IDs found: {set(duplicates)}")

    # Validate each example can parse its typed input/reference
    for ex in dataset.examples:
        try:
            ex.get_typed_input()
        except Exception as e:
            errors.append(f"Example {ex.id}: Invalid input for task {ex.task}: {e}")

        try:
            ex.get_typed_reference()
        except Exception as e:
            errors.append(f"Example {ex.id}: Invalid reference for task {ex.task}: {e}")

    return errors


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    """
    Save a dataset to a JSONL file.

    Args:
        dataset: Dataset to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for example in dataset.examples:
            f.write(example.model_dump_json() + "\n")

    logger.info(f"Saved dataset '{dataset.name}' to {path}")


def save_corpus(corpus: Corpus, path: str | Path) -> None:
    """
    Save a corpus to a JSONL file.

    Args:
        corpus: Corpus to save
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in corpus.documents:
            f.write(doc.model_dump_json() + "\n")

    logger.info(f"Saved corpus '{corpus.name}' to {path}")
