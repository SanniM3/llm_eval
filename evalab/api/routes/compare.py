"""API routes for run comparison."""

from fastapi import APIRouter, HTTPException, Query

from evalab.api.schemas import CompareResponse, MetricDelta
from evalab.storage.database import init_db
from evalab.storage.registry import RunRegistry

router = APIRouter(prefix="/compare", tags=["compare"])

init_db()
_registry = RunRegistry()


def get_registry() -> RunRegistry:
    """Get the run registry."""
    return _registry


@router.get("", response_model=CompareResponse)
async def compare_runs(
    a: str = Query(..., description="First run ID"),
    b: str = Query(..., description="Second run ID"),
) -> CompareResponse:
    """
    Compare metrics between two runs.

    Returns deltas for each metric, with positive values indicating
    improvement from run A to run B.
    """
    registry = get_registry()

    # Verify both runs exist
    run_a = registry.get_run(a)
    run_b = registry.get_run(b)

    if not run_a:
        raise HTTPException(status_code=404, detail=f"Run A not found: {a}")
    if not run_b:
        raise HTTPException(status_code=404, detail=f"Run B not found: {b}")

    # Get comparison
    comparison = registry.compare_runs(a, b)

    # Build metric deltas
    metric_deltas = []
    improvements = 0
    regressions = 0

    for metric_name, data in comparison["metrics"].items():
        a_mean = data.get("run_a", {}).get("mean")
        b_mean = data.get("run_b", {}).get("mean")
        delta_mean = data.get("delta", {}).get("mean", 0)

        # Determine if improvement (depends on metric type)
        # For most metrics, higher is better
        # For latency/cost, lower is better
        lower_is_better = any(x in metric_name.lower() for x in ["latency", "cost", "error", "loss"])

        if lower_is_better:
            is_improvement = delta_mean < 0
        else:
            is_improvement = delta_mean > 0

        if delta_mean != 0:
            if is_improvement:
                improvements += 1
            else:
                regressions += 1

        metric_deltas.append(
            MetricDelta(
                metric_name=metric_name,
                run_a_mean=a_mean,
                run_b_mean=b_mean,
                delta=delta_mean,
                improvement=is_improvement,
            )
        )

    # Sort by absolute delta (most significant changes first)
    metric_deltas.sort(key=lambda x: abs(x.delta), reverse=True)

    summary = {
        "total_metrics": len(metric_deltas),
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": len(metric_deltas) - improvements - regressions,
        "run_a_name": run_a.name,
        "run_b_name": run_b.name,
        "run_a_status": run_a.status,
        "run_b_status": run_b.status,
    }

    return CompareResponse(
        run_a=a,
        run_b=b,
        metric_deltas=metric_deltas,
        summary=summary,
    )
