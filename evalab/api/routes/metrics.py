"""API routes for metrics."""

from fastapi import APIRouter, HTTPException, Query

from evalab.api.schemas import (
    AggregateValue,
    ExampleDetailResponse,
    MetricsResponse,
    MetricValue,
    PredictionResponse,
)
from evalab.storage.database import init_db
from evalab.storage.registry import RunRegistry

router = APIRouter(prefix="/runs/{run_id}", tags=["metrics"])

init_db()
_registry = RunRegistry()


def get_registry() -> RunRegistry:
    """Get the run registry."""
    return _registry


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    run_id: str,
    metric_name: str | None = Query(None, description="Filter by metric name"),
    include_examples: bool = Query(False, description="Include per-example metrics"),
) -> MetricsResponse:
    """Get metrics for a run."""
    registry = get_registry()

    # Check run exists
    run = registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Get aggregates
    aggregates = registry.get_aggregates(run_id, metric_name=metric_name)

    aggregate_values = [
        AggregateValue(
            metric_name=a.metric_name,
            slice_key=a.slice_key,
            slice_value=a.slice_value,
            count=a.count,
            mean=a.agg_json.get("mean", 0),
            std=a.agg_json.get("std", 0),
            min=a.agg_json.get("min", 0),
            max=a.agg_json.get("max", 0),
            p50=a.agg_json.get("p50", 0),
            p95=a.agg_json.get("p95", 0),
            p99=a.agg_json.get("p99", 0),
        )
        for a in aggregates
    ]

    # Get per-example metrics if requested
    per_example = None
    if include_examples:
        metrics = registry.get_metrics(run_id, metric_name=metric_name)
        per_example = [
            MetricValue(
                example_id=m.example_id,
                metric_name=m.metric_name,
                value=m.metric_value,
                details=m.metric_json,
            )
            for m in metrics
        ]

    return MetricsResponse(
        run_id=run_id,
        aggregates=aggregate_values,
        per_example=per_example,
    )


@router.get("/examples/{example_id}", response_model=ExampleDetailResponse)
async def get_example_detail(
    run_id: str,
    example_id: str,
) -> ExampleDetailResponse:
    """Get detailed results for a specific example."""
    registry = get_registry()

    # Check run exists
    run = registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Get prediction
    predictions = registry.get_predictions(run_id)
    prediction = next((p for p in predictions if p.example_id == example_id), None)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Example not found: {example_id}",
        )

    # Get metrics for this example
    metrics = registry.get_metrics(run_id, example_id=example_id)

    metric_values = [
        MetricValue(
            example_id=m.example_id,
            metric_name=m.metric_name,
            value=m.metric_value,
            details=m.metric_json,
        )
        for m in metrics
    ]

    # Get retrieval trace if available
    retrieval_trace = None
    traces = registry.artifacts.load_retrieval_traces(run_id)
    for trace in traces:
        if trace.get("query") and example_id in str(trace):
            retrieval_trace = trace
            break

    return ExampleDetailResponse(
        run_id=run_id,
        example_id=example_id,
        prediction=PredictionResponse(
            example_id=prediction.example_id,
            output_text=prediction.output_text,
            latency_ms=prediction.latency_ms,
            input_tokens=prediction.input_tokens,
            output_tokens=prediction.output_tokens,
            finish_reason=prediction.finish_reason,
            error=prediction.error,
        ),
        metrics=metric_values,
        retrieval_trace=retrieval_trace,
    )
