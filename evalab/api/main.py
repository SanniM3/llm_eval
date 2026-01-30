"""FastAPI application for LLM-EvalLab."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from evalab import __version__
from evalab.api.routes import compare_router, metrics_router, runs_router
from evalab.api.schemas import HealthResponse
from evalab.storage.database import init_db


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="LLM-EvalLab API",
        description="API for LLM Evaluation & Reliability Platform",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize database on startup
    @app.on_event("startup")
    async def startup_event():
        init_db()

    # Include routers
    app.include_router(runs_router)
    app.include_router(metrics_router)
    app.include_router(compare_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Check API health status."""
        return HealthResponse(
            status="healthy",
            version=__version__,
            database="connected",
        )

    # Root endpoint
    @app.get("/", tags=["health"])
    async def root() -> dict:
        """Root endpoint with API info."""
        return {
            "name": "LLM-EvalLab API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create app instance for uvicorn
app = create_app()
