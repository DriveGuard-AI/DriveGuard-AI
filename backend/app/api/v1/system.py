"""API-002 系统状态。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import AppContext, get_context, require_permission
from ...schemas import SystemStatus
from ...security import PERMISSION_MONITORING_READ

router = APIRouter(tags=["system"])


@router.get(
    "/system/status",
    response_model=SystemStatus,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def system_status(context: AppContext = Depends(get_context)) -> SystemStatus:
    return context.system.status()


__all__ = ["router"]
