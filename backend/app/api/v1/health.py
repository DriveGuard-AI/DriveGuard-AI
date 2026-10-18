"""API-001 健康检查（匿名可用）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import AppContext, get_context
from ...schemas import HealthResponse
from ...timeutil import utcnow

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(context: AppContext = Depends(get_context)) -> HealthResponse:
    """只表示 API 进程存活，不代表摄像头或模型健康。"""

    return HealthResponse(
        status="ok",
        service_version=context.config.service_version,
        time=utcnow(),
    )


__all__ = ["router"]
