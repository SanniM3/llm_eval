"""SQLAlchemy ORM models for the evaluation database."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Run(Base):
    """
    A single evaluation run.

    Stores configuration, metadata, and links to predictions/metrics.
    """

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), default="pending", nullable=False, index=True
    )  # pending, running, completed, failed
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    dataset_id: Mapped[str] = mapped_column(String(128), nullable=True, index=True)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="run", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", back_populates="run", cascade="all, delete-orphan"
    )
    aggregates: Mapped[list["Aggregate"]] = relationship(
        "Aggregate", back_populates="run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Run(id={self.id}, name={self.name}, status={self.status})>"


class Example(Base):
    """
    A cached example from a dataset.

    Used to avoid re-parsing JSONL files and enable efficient querying.
    """

    __tablename__ = "examples"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    task: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reference_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    # Indexes
    __table_args__ = (Index("ix_examples_dataset_task", "dataset_id", "task"),)

    def __repr__(self) -> str:
        return f"<Example(id={self.id}, task={self.task})>"


class Prediction(Base):
    """
    A single prediction from a run.

    Links a run to an example with the generated output.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    example_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    output_text: Mapped[str] = mapped_column(Text, nullable=False)
    trace_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    finish_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now(), nullable=False)

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="predictions")

    # Indexes
    __table_args__ = (Index("ix_predictions_run_example", "run_id", "example_id"),)

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, run_id={self.run_id}, example_id={self.example_id})>"


class Metric(Base):
    """
    A single metric value for an example in a run.

    Stores both scalar values and optional JSON details.
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    example_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="metrics")

    # Indexes
    __table_args__ = (
        Index("ix_metrics_run_example", "run_id", "example_id"),
        Index("ix_metrics_run_metric", "run_id", "metric_name"),
    )

    def __repr__(self) -> str:
        return f"<Metric(run_id={self.run_id}, example_id={self.example_id}, {self.metric_name}={self.metric_value})>"


class Aggregate(Base):
    """
    Aggregated metrics for a run, optionally sliced by metadata.

    Stores summary statistics like mean, std, percentiles.
    """

    __tablename__ = "aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    slice_key: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )  # e.g., "domain", "difficulty"
    slice_value: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )  # e.g., "finance", "easy"
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    agg_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False
    )  # mean, std, min, max, p50, p95, p99

    # Relationships
    run: Mapped["Run"] = relationship("Run", back_populates="aggregates")

    # Indexes
    __table_args__ = (
        Index("ix_aggregates_run_metric", "run_id", "metric_name"),
        Index("ix_aggregates_run_slice", "run_id", "slice_key", "slice_value"),
    )

    def __repr__(self) -> str:
        slice_info = f"{self.slice_key}={self.slice_value}" if self.slice_key else "global"
        return f"<Aggregate(run_id={self.run_id}, {self.metric_name}, {slice_info})>"
