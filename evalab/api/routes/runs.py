"""API routes for run management."""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from evalab.api.schemas import (
    RunConfigRequest,
    RunCreateResponse,
    RunListResponse,
    RunResponse,
)
from evalab.config.schemas import RunConfig
from evalab.pipeline.runner import PipelineRunner
from evalab.storage.database import init_db
from evalab.storage.registry import RunRegistry

router = APIRouter(prefix="/runs", tags=["runs"])

# Initialize registry
init_db()
_registry = RunRegistry()


def get_registry() -> RunRegistry:
    """Get the run registry."""
    return _registry


@router.get("", response_model=RunListResponse)
async def list_runs(
    status: str | None = Query(None, description="Filter by status"),
    name: str | None = Query(None, description="Filter by name substring"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> RunListResponse:
    """List all evaluation runs."""
    registry = get_registry()

    runs = registry.list_runs(
        status=status,
        name_contains=name,
        limit=limit,
        offset=offset,
    )

    run_responses = [
        RunResponse(
            id=r.id,
            name=r.name,
            status=r.status,
            created_at=r.created_at,
            completed_at=r.completed_at,
            config=r.config_json,
            notes=r.notes,
            tags=r.tags,
            error_message=r.error_message,
        )
        for r in runs
    ]

    return RunListResponse(
        runs=run_responses,
        total=len(runs),  # In production, would query total count separately
        offset=offset,
        limit=limit,
    )


@router.post("", response_model=RunCreateResponse)
async def create_run(
    config: RunConfigRequest,
    background_tasks: BackgroundTasks,
    run_async: bool = Query(True, description="Run evaluation asynchronously"),
) -> RunCreateResponse:
    """
    Create and optionally start a new evaluation run.

    If run_async=True (default), the run executes in the background.
    """
    registry = get_registry()

    try:
        # Convert to RunConfig
        run_config = RunConfig(
            run_name=config.run_name,
            dataset=config.dataset,
            generation=config.generation,
            prompt=config.prompt,
            retrieval=config.retrieval,
            evaluation=config.evaluation,
            attribution=config.attribution,
            logging=config.logging,
            notes=config.notes,
            tags=config.tags,
        )

        if run_async:
            # Create run entry
            run_id = registry.create_run(
                name=run_config.run_name,
                config=run_config.model_dump(),
                notes=run_config.notes,
                tags=run_config.tags,
            )

            # Run in background
            def run_pipeline():
                runner = PipelineRunner(run_config, registry=registry)
                runner.run()

            background_tasks.add_task(run_pipeline)

            return RunCreateResponse(
                run_id=run_id,
                status="pending",
                message="Run started in background",
            )
        else:
            # Run synchronously
            runner = PipelineRunner(run_config, registry=registry)
            result = runner.run()

            return RunCreateResponse(
                run_id=result.run_id,
                status=result.status,
                message=f"Run completed in {result.duration_sec:.2f}s",
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Get details for a specific run."""
    registry = get_registry()

    run = registry.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return RunResponse(
        id=run.id,
        name=run.name,
        status=run.status,
        created_at=run.created_at,
        completed_at=run.completed_at,
        config=run.config_json,
        notes=run.notes,
        tags=run.tags,
        error_message=run.error_message,
    )


@router.delete("/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Delete a run and its artifacts."""
    registry = get_registry()

    if registry.delete_run(run_id):
        return {"status": "deleted", "run_id": run_id}
    else:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
