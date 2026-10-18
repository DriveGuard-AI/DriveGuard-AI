"""REST API 路由（`/api/v1`）。"""

from fastapi import APIRouter

from .events import router as events_router
from .health import router as health_router
from .models import router as models_router
from .monitoring import router as monitoring_router
from .settings import router as settings_router
from .statistics import router as statistics_router
from .system import router as system_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(system_router)
api_v1_router.include_router(events_router)
api_v1_router.include_router(statistics_router)
api_v1_router.include_router(settings_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(monitoring_router)

__all__ = ["api_v1_router"]
