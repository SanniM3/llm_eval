"""
LLM-EvalLab: LLM Evaluation & Reliability Platform

A production-style evaluation platform for LLM applications including:
- QA, Summarization, Classification, and RAG tasks
- Multi-metric evaluation (accuracy, semantic, faithfulness, robustness, calibration)
- Component-wise error attribution
- Cost/latency accounting
- Dashboard and API for comparing runs
"""

__version__ = "0.1.0"
__author__ = "LLM-EvalLab Team"

from evalab.config.schemas import RunConfig
from evalab.data.loader import load_dataset, load_corpus
from evalab.storage.registry import RunRegistry

__all__ = [
    "__version__",
    "RunConfig",
    "load_dataset",
    "load_corpus",
    "RunRegistry",
]
