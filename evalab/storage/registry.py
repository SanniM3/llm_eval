"""Run registry for managing evaluation runs."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from evalab.data.versioning import compute_config_hash, get_run_id
from evalab.storage.artifacts import ArtifactStore
from evalab.storage.database import session_scope
from evalab.storage.models import Aggregate, Example, Metric, Prediction, Run

logger = logging.getLogger(__name__)


class RunRegistry:
    """
    Registry for managing evaluation runs.

    Combines database storage with file artifact management.
    """

    def __init__(self, artifact_dir: str = "runs"):
        """
        Initialize run registry.

        Args:
            artifact_dir: Directory for artifact storage
        """
        self.artifacts = ArtifactStore(artifact_dir)

    def create_run(
        self,
        name: str,
        config: dict[str, Any],
        notes: str | None = None,
        tags: list[str] | None = None,
        git_commit: str | None = None,
    ) -> str:
        """
        Create a new evaluation run.

        Args:
            name: Run name
            config: Run configuration dictionary
            notes: Optional notes
            tags: Optional tags
            git_commit: Optional git commit hash

        Returns:
            Unique run ID
        """
        config_hash = compute_config_hash(config)
        run_id = get_run_id(name, config_hash)

        with session_scope() as session:
            run = Run(
                id=run_id,
                name=name,
                config_json=config,
                config_hash=config_hash,
                notes=notes,
                tags=tags or [],
                git_commit=git_commit,
                status="pending",
            )
            session.add(run)

        # Create artifact directory
        self.artifacts.create_run_dir(run_id)
        self.artifacts.save_config(run_id, config)

        logger.info(f"Created run: {run_id}")
        return run_id

    def start_run(self, run_id: str) -> None:
        """Mark a run as started."""
        with session_scope() as session:
            run = session.get(Run, run_id)
            if run:
                run.status = "running"

    def complete_run(self, run_id: str, error: str | None = None) -> None:
        """
        Mark a run as completed or failed.

        Args:
            run_id: Run identifier
            error: Optional error message (sets status to "failed")
        """
        with session_scope() as session:
            run = session.get(Run, run_id)
            if run:
                run.completed_at = datetime.utcnow()
                if error:
                    run.status = "failed"
                    run.error_message = error
                else:
                    run.status = "completed"

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID."""
        with session_scope() as session:
            return session.get(Run, run_id)

    def list_runs(
        self,
        status: str | None = None,
        name_contains: str | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Run]:
        """
        List runs with optional filters.

        Args:
            status: Filter by status
            name_contains: Filter by name substring
            tags: Filter by tags (any match)
            limit: Maximum number of runs
            offset: Offset for pagination

        Returns:
            List of Run objects
        """
        with session_scope() as session:
            query = select(Run).order_by(desc(Run.created_at))

            if status:
                query = query.where(Run.status == status)
            if name_contains:
                query = query.where(Run.name.contains(name_contains))
            # Note: tag filtering with JSON is database-specific

            query = query.limit(limit).offset(offset)
            result = session.execute(query)
            return list(result.scalars().all())

    def delete_run(self, run_id: str) -> bool:
        """
        Delete a run and all associated data.

        Args:
            run_id: Run identifier

        Returns:
            True if deleted, False if not found
        """
        with session_scope() as session:
            run = session.get(Run, run_id)
            if run:
                session.delete(run)
                self.artifacts.delete_run(run_id)
                logger.info(f"Deleted run: {run_id}")
                return True
        return False

    def add_prediction(
        self,
        run_id: str,
        example_id: str,
        output_text: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        trace_path: str | None = None,
        finish_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        """
        Add a prediction to a run.

        Args:
            run_id: Run identifier
            example_id: Example identifier
            output_text: Generated output
            latency_ms: Generation latency
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            trace_path: Optional path to trace file
            finish_reason: Completion reason
            error: Optional error message
        """
        with session_scope() as session:
            prediction = Prediction(
                run_id=run_id,
                example_id=example_id,
                output_text=output_text,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                trace_path=trace_path,
                finish_reason=finish_reason,
                error=error,
            )
            session.add(prediction)

        # Also save to artifacts
        pred_dict = {
            "example_id": example_id,
            "output_text": output_text,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "error": error,
        }
        self.artifacts.append_prediction(run_id, pred_dict)

    def get_predictions(self, run_id: str) -> list[Prediction]:
        """Get all predictions for a run."""
        with session_scope() as session:
            query = select(Prediction).where(Prediction.run_id == run_id)
            result = session.execute(query)
            return list(result.scalars().all())

    def add_metric(
        self,
        run_id: str,
        example_id: str,
        metric_name: str,
        metric_value: float,
        metric_json: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a metric for an example.

        Args:
            run_id: Run identifier
            example_id: Example identifier
            metric_name: Name of the metric
            metric_value: Scalar metric value
            metric_json: Optional detailed metric data
        """
        with session_scope() as session:
            metric = Metric(
                run_id=run_id,
                example_id=example_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_json=metric_json,
            )
            session.add(metric)

    def add_metrics_batch(
        self,
        run_id: str,
        metrics: list[dict[str, Any]],
    ) -> None:
        """
        Add multiple metrics in a batch.

        Args:
            run_id: Run identifier
            metrics: List of metric dictionaries with keys:
                - example_id, metric_name, metric_value, metric_json (optional)
        """
        with session_scope() as session:
            for m in metrics:
                metric = Metric(
                    run_id=run_id,
                    example_id=m["example_id"],
                    metric_name=m["metric_name"],
                    metric_value=m["metric_value"],
                    metric_json=m.get("metric_json"),
                )
                session.add(metric)

    def get_metrics(
        self,
        run_id: str,
        example_id: str | None = None,
        metric_name: str | None = None,
    ) -> list[Metric]:
        """
        Get metrics with optional filters.

        Args:
            run_id: Run identifier
            example_id: Optional example filter
            metric_name: Optional metric name filter

        Returns:
            List of Metric objects
        """
        with session_scope() as session:
            query = select(Metric).where(Metric.run_id == run_id)

            if example_id:
                query = query.where(Metric.example_id == example_id)
            if metric_name:
                query = query.where(Metric.metric_name == metric_name)

            result = session.execute(query)
            return list(result.scalars().all())

    def compute_aggregates(
        self,
        run_id: str,
        slice_keys: list[str] | None = None,
    ) -> None:
        """
        Compute and store aggregate metrics for a run.

        Args:
            run_id: Run identifier
            slice_keys: Optional metadata keys to slice by
        """
        with session_scope() as session:
            # Get all metrics for this run
            metrics = session.execute(
                select(Metric).where(Metric.run_id == run_id)
            ).scalars().all()

            if not metrics:
                return

            # Group by metric name
            by_metric: dict[str, list[float]] = {}
            for m in metrics:
                if m.metric_name not in by_metric:
                    by_metric[m.metric_name] = []
                by_metric[m.metric_name].append(m.metric_value)

            # Compute global aggregates
            for metric_name, values in by_metric.items():
                arr = np.array(values)
                agg = Aggregate(
                    run_id=run_id,
                    metric_name=metric_name,
                    slice_key=None,
                    slice_value=None,
                    count=len(values),
                    agg_json={
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "p50": float(np.percentile(arr, 50)),
                        "p95": float(np.percentile(arr, 95)),
                        "p99": float(np.percentile(arr, 99)),
                    },
                )
                session.add(agg)

    def get_aggregates(
        self,
        run_id: str,
        metric_name: str | None = None,
        slice_key: str | None = None,
    ) -> list[Aggregate]:
        """
        Get aggregates with optional filters.

        Args:
            run_id: Run identifier
            metric_name: Optional metric name filter
            slice_key: Optional slice key filter

        Returns:
            List of Aggregate objects
        """
        with session_scope() as session:
            query = select(Aggregate).where(Aggregate.run_id == run_id)

            if metric_name:
                query = query.where(Aggregate.metric_name == metric_name)
            if slice_key is not None:
                query = query.where(Aggregate.slice_key == slice_key)

            result = session.execute(query)
            return list(result.scalars().all())

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> dict[str, Any]:
        """
        Compare metrics between two runs.

        Args:
            run_id_a: First run ID
            run_id_b: Second run ID

        Returns:
            Comparison dictionary with deltas
        """
        aggs_a = {
            (a.metric_name, a.slice_key, a.slice_value): a.agg_json
            for a in self.get_aggregates(run_id_a)
        }
        aggs_b = {
            (a.metric_name, a.slice_key, a.slice_value): a.agg_json
            for a in self.get_aggregates(run_id_b)
        }

        comparison = {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "metrics": {},
        }

        all_keys = set(aggs_a.keys()) | set(aggs_b.keys())
        for key in all_keys:
            metric_name, slice_key, slice_value = key
            a_vals = aggs_a.get(key, {})
            b_vals = aggs_b.get(key, {})

            metric_key = metric_name
            if slice_key:
                metric_key = f"{metric_name}[{slice_key}={slice_value}]"

            comparison["metrics"][metric_key] = {
                "run_a": a_vals,
                "run_b": b_vals,
                "delta": {
                    stat: b_vals.get(stat, 0) - a_vals.get(stat, 0)
                    for stat in ["mean", "p50", "p95"]
                    if stat in a_vals or stat in b_vals
                },
            }

        return comparison

    def cache_examples(self, dataset_id: str, examples: list[dict[str, Any]]) -> None:
        """
        Cache dataset examples in the database.

        Args:
            dataset_id: Dataset identifier
            examples: List of example dictionaries
        """
        with session_scope() as session:
            for ex in examples:
                example = Example(
                    id=f"{dataset_id}_{ex['id']}",
                    dataset_id=dataset_id,
                    task=ex["task"],
                    input_json=ex["input"],
                    reference_json=ex["reference"],
                    metadata_json=ex.get("metadata", {}),
                )
                session.merge(example)  # Use merge to handle duplicates
