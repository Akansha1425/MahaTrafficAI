"""MahaTraffic AI — FastAPI Application Entry Point.

Production-grade API initializing routing, middleware, CORS,
centralized settings, and health check endpoints.
"""

from contextlib import asynccontextmanager
import logging
import sys
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from backend.app.config import get_settings
from backend.app.models.responses import HealthResponse
from backend.app.routes import api_router

# Configure structured logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mahatraffic.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    logger.info("Starting up %s in [%s] environment...", settings.APP_NAME, settings.ENVIRONMENT)
    logger.info("API prefix mounted at: %s", settings.API_V1_PREFIX)
    yield
    logger.info("Shutting down %s cleanly.", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "A Reliable Multi-Agent Road-Risk Intelligence System using "
        "Big Data, Social Media Analytics, MCP and Guardrails."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restricted in production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["System"],
    summary="Service Health Check",
)
async def health_check():
    """Verify service operational availability."""
    return HealthResponse(
        status="ok",
        project=settings.APP_NAME,
        version="0.1.0",
        environment=settings.ENVIRONMENT,
    )


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
