"""API-009 快照、API-010 实时预览、管线切回实时。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ...deps import AppContext, get_context, require_permission
from ...errors import MonitoringNotActiveError
from ...schemas import PipelineControlResponse, PreviewState, RiskSnapshot
from ...security import PERMISSION_MONITORING_READ, PERMISSION_MONITORING_WRITE
from ...services.preview import BOUNDARY

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


@router.get(
    "/monitoring/preview/state",
    response_model=PreviewState,
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
def preview_state(context: AppContext = Depends(get_context)) -> PreviewState:
    """预览开关、连接数与最近一帧的序号/摘要（用于 seq/hash 对齐展示）。"""

    return context.preview.state()


@router.get(
    "/monitoring/preview",
    dependencies=[Depends(require_permission(PERMISSION_MONITORING_READ))],
)
async def preview_stream(request: Request, context: AppContext = Depends(get_context)):
    """实时视频预览（MJPEG）。

    每个 part 的 `X-Seq` / `X-Timestamp-Ms` / `X-Sha256` 描述的是**送进推理的
    那一帧**，前端用 sha256 即可确认"看到的就是模型输入"。错误按
    开关 → 会话 → 设备 → 连接数 的顺序在开流前返回 JSON；一旦返回 200，
    后续故障只能关闭连接。
    """

    context.preview.ensure_ready()
    return StreamingResponse(
        context.preview.stream(is_disconnected=request.is_disconnected),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            # 反向代理（nginx）默认会缓冲流，实时预览必须显式关闭。
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
