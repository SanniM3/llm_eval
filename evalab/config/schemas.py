"""Pydantic schemas for run configuration."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RetrieverType(str, Enum):
    """Supported retriever types."""

    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class BackendType(str, Enum):
    """Supported LLM backend types."""

    OPENAI = "openai"
    VLLM = "vllm"


class ChunkingConfig(BaseModel):
    """Configuration for text chunking."""

    size: int = Field(default=512, ge=64, le=8192, description="Chunk size in tokens/chars")
    overlap: int = Field(default=64, ge=0, description="Overlap between chunks")

    @field_validator("overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info: Any) -> int:
        """Ensure overlap is less than chunk size."""
        if "size" in info.data and v >= info.data["size"]:
            raise ValueError("overlap must be less than size")
        return v


class RetrieverConfig(BaseModel):
    """Configuration for the retriever."""

    type: RetrieverType = Field(default=RetrieverType.HYBRID, description="Retriever type")
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="BM25 weight for hybrid")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Dense weight for hybrid")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for dense retrieval"
    )


class RetrievalConfig(BaseModel):
    """Configuration for the retrieval pipeline."""

    enabled: bool = Field(default=False, description="Whether retrieval is enabled")
    corpus_path: str | None = Field(default=None, description="Path to corpus JSONL file")
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    top_k: int = Field(default=5, ge=1, le=100, description="Number of documents to retrieve")
    reranker: str | None = Field(default=None, description="Optional reranker model")


class GenerationConfig(BaseModel):
    """Configuration for LLM generation."""

    backend: BackendType = Field(default=BackendType.OPENAI, description="LLM backend to use")
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=512, ge=1, le=32768, description="Maximum output tokens")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    api_base: str | None = Field(default=None, description="Custom API base URL (for vLLM)")


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    template_path: str = Field(..., description="Path to Jinja2 prompt template")
    system_message: str | None = Field(default=None, description="Optional system message")


class EvaluationSuiteType(str, Enum):
    """Available evaluation suites."""

    ACCURACY = "accuracy"
    SEMANTIC = "semantic"
    FAITHFULNESS = "faithfulness"
    ROBUSTNESS = "robustness"
    CALIBRATION = "calibration"
    COST_LATENCY = "cost_latency"


class EvaluationConfig(BaseModel):
    """Configuration for evaluation suites."""

    suites: list[EvaluationSuiteType] = Field(
        default=[EvaluationSuiteType.ACCURACY, EvaluationSuiteType.SEMANTIC],
        description="Evaluation suites to run",
    )
    faithfulness_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for supported claims"
    )
    judge_model: str = Field(
        default="gpt-4o-mini", description="Model to use as LLM judge for faithfulness"
    )
    judge_samples: int = Field(
        default=3, ge=1, le=10, description="Number of judge samples for majority vote"
    )
    robustness_perturbations: list[str] = Field(
        default=["paraphrase", "distractor", "drop_top_k"],
        description="Perturbation types for robustness",
    )
    calibration_bins: int = Field(default=10, ge=5, le=50, description="Number of bins for ECE")
    calibration_samples: int = Field(
        default=5, ge=1, le=20, description="Number of samples for self-consistency"
    )


class AttributionProbeType(str, Enum):
    """Available attribution probes."""

    FIX_RETRIEVAL_VARY_PROMPT = "fix_retrieval_vary_prompt"
    FIX_PROMPT_VARY_RETRIEVAL = "fix_prompt_vary_retrieval"
    GOLD_CONTEXT = "gold_context"


class AttributionConfig(BaseModel):
    """Configuration for error attribution."""

    enabled: bool = Field(default=False, description="Whether attribution analysis is enabled")
    probes: list[AttributionProbeType] = Field(
        default=[AttributionProbeType.GOLD_CONTEXT], description="Counterfactual probes to run"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging and artifact storage."""

    save_traces: bool = Field(default=True, description="Whether to save detailed traces")
    output_dir: str = Field(default="runs/", description="Directory for run artifacts")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )


class DatasetConfig(BaseModel):
    """Configuration for the evaluation dataset."""

    path: str = Field(..., description="Path to dataset JSONL file")
    name: str | None = Field(default=None, description="Optional dataset name")
    subset: str | None = Field(default=None, description="Optional subset/split name")
    max_examples: int | None = Field(default=None, ge=1, description="Maximum examples to process")


class RunConfig(BaseModel):
    """Complete run configuration."""

    run_name: str = Field(..., description="Unique name for this run")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    prompt: PromptConfig = Field(..., description="Prompt configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    attribution: AttributionConfig = Field(default_factory=AttributionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    notes: str | None = Field(default=None, description="Optional notes about this run")
    tags: list[str] = Field(default_factory=list, description="Tags for organizing runs")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
