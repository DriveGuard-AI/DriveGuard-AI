"""API-009 当前风险快照。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import AppContext, get_context, require_permission
from ...errors import MonitoringNotActiveError
from ...schemas import RiskSnapshot
from ...security import PERMISSION_MONITORING_READ

router = APIRouter(tags=["monitoring"])


@router.get(
    "/monitoring/snapshot",
    response_model=RiskSnapshot,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def current_snapshot(context: AppContext = Depends(get_context)) -> RiskSnapshot:
    """页面首次加载或 WebSocket 重连后获取完整最新状态。"""

    latest = context.engine.latest()
    if latest is None or context.engine.session_id is None:
        raise MonitoringNotActiveError()
    return latest


__all__ = ["router"]
