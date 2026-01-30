"""Content hashing and versioning for datasets and configs."""

import hashlib
import json
from pathlib import Path
from typing import Any


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file's contents.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)

    Returns:
        Hexadecimal hash string
    """
    path = Path(path)
    hasher = hashlib.new(algorithm)

    with open(path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_dataset_hash(path: str | Path) -> str:
    """
    Compute a content hash for a dataset file.

    This is used for versioning and ensuring reproducibility.

    Args:
        path: Path to the dataset JSONL file

    Returns:
        SHA256 hash of the file contents
    """
    return compute_file_hash(path, "sha256")


def get_dataset_id(name: str, path: str | Path) -> str:
    """
    Generate a unique dataset identifier from name and content hash.

    Args:
        name: Dataset name
        path: Path to the dataset file

    Returns:
        Dataset ID in format: {name}_{hash_prefix}
    """
    content_hash = compute_dataset_hash(path)
    # Use first 12 characters of hash for brevity
    return f"{name}_{content_hash[:12]}"


def compute_config_hash(config: dict[str, Any]) -> str:
    """
    Compute a hash of a configuration dictionary.

    This normalizes the config (sorted keys, consistent formatting)
    to ensure identical configs produce identical hashes.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 hash of the normalized config
    """
    # Normalize by sorting keys and using consistent JSON formatting
    normalized = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_string_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a string.

    Args:
        text: Input text
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def get_run_id(run_name: str, config_hash: str, timestamp: str | None = None) -> str:
    """
    Generate a unique run identifier.

    Args:
        run_name: Name of the run
        config_hash: Hash of the run configuration
        timestamp: Optional ISO timestamp

    Returns:
        Run ID in format: {run_name}_{hash_prefix}_{timestamp}
    """
    from datetime import datetime

    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Use first 8 chars of config hash
    return f"{run_name}_{config_hash[:8]}_{timestamp}"
