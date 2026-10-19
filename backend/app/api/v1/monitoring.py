"""API-009 当前风险快照。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import AppContext, get_context, require_permission
from ...errors import MonitoringNotActiveError
from ...schemas import PipelineControlResponse, RiskSnapshot
from ...security import PERMISSION_MONITORING_READ, PERMISSION_MONITORING_WRITE

router = APIRouter(tags=["monitoring"])


@router.post(
    "/monitoring/resume",
    response_model=PipelineControlResponse,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_WRITE))],
)
async def resume_monitoring(context: AppContext = Depends(get_context)) -> PipelineControlResponse:
    """从离线回放切回实时采集。幂等；有分析任务在跑时返回 409。"""

    return await context.analysis.resume_realtime()


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
