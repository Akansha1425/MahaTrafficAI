"""API Route aggregation module."""

from fastapi import APIRouter
from backend.app.routes.dashboard import router as dashboard_router
from backend.app.routes.analytics import router as analytics_router
from backend.app.routes.risk import router as risk_router
from backend.app.routes.social import router as social_router
from backend.app.routes.agent import router as agent_router

api_router = APIRouter()
api_router.include_router(dashboard_router)
api_router.include_router(analytics_router)
api_router.include_router(risk_router)
api_router.include_router(social_router)
api_router.include_router(agent_router)

__all__ = [
    "api_router",
    "dashboard_router",
    "analytics_router",
    "risk_router",
    "social_router",
    "agent_router",
]
