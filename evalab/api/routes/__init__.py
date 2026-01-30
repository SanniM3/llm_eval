"""API route modules."""

from evalab.api.routes.runs import router as runs_router
from evalab.api.routes.metrics import router as metrics_router
from evalab.api.routes.compare import router as compare_router

__all__ = ["runs_router", "metrics_router", "compare_router"]
