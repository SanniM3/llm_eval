"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================================
# Run schemas
# ============================================================================


class RunConfigRequest(BaseModel):
    """Request to create a new run."""

    run_name: str = Field(..., description="Name for the run")
    dataset: dict[str, Any] = Field(..., description="Dataset configuration")
    generation: dict[str, Any] = Field(default_factory=dict)
    prompt: dict[str, Any] = Field(...)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    attribution: dict[str, Any] = Field(default_factory=dict)
    logging: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    """Response with run details."""

    id: str
    name: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    config: dict[str, Any]
    notes: str | None = None
    tags: list[str] = Field(default_factory=list)
    error_message: str | None = None


class RunListResponse(BaseModel):
    """Response with list of runs."""

    runs: list[RunResponse]
    total: int
    offset: int
    limit: int


class RunCreateResponse(BaseModel):
    """Response after creating a run."""

    run_id: str
    status: str
    message: str


# ============================================================================
# Metrics schemas
# ============================================================================


class MetricValue(BaseModel):
    """A single metric value."""

    example_id: str
    metric_name: str
    value: float
    details: dict[str, Any] | None = None


class AggregateValue(BaseModel):
    """Aggregated metric value."""

    metric_name: str
    slice_key: str | None = None
    slice_value: str | None = None
    count: int
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


class MetricsResponse(BaseModel):
    """Response with metrics for a run."""

    run_id: str
    aggregates: list[AggregateValue]
    per_example: list[MetricValue] | None = None


# ============================================================================
# Example/Prediction schemas
# ============================================================================


class PredictionResponse(BaseModel):
    """Response with prediction details."""

    example_id: str
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    error: str | None = None


class ExampleDetailResponse(BaseModel):
    """Detailed response for a single example."""

    run_id: str
    example_id: str
    prediction: PredictionResponse
    metrics: list[MetricValue]
    retrieval_trace: dict[str, Any] | None = None


# ============================================================================
# Comparison schemas
# ============================================================================


class MetricDelta(BaseModel):
    """Delta between two runs for a metric."""

    metric_name: str
    run_a_mean: float | None
    run_b_mean: float | None
    delta: float
    improvement: bool


class CompareResponse(BaseModel):
    """Response comparing two runs."""

    run_a: str
    run_b: str
    metric_deltas: list[MetricDelta]
    summary: dict[str, Any]


# ============================================================================
# Health/Status schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
