"""WebSocket 端点：`GET /ws/v1/monitoring`。

注意：本模块不导入 `app.deps`，以保持包依赖为单向（避免循环导入）。
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..security import (
    PERMISSION_MONITORING_READ,
    ensure_permission,
    principal_from_websocket,
)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/v1/monitoring")
async def monitoring_socket(websocket: WebSocket) -> None:
    context = websocket.app.state.context
    principal = principal_from_websocket(context.config, websocket)

    try:
        ensure_permission(principal, PERMISSION_MONITORING_READ)
    except Exception:
        # 未授权：拒绝握手（1008 = policy violation），不进入消息循环
        await websocket.close(code=1008)
        return

    hub = context.hub
    engine = context.engine
    await hub.connect(websocket, session_id=engine.session_id)

    latest = engine.latest()
    if latest is not None:
        await hub.broadcast_snapshot(latest)

    try:
        while True:
            # 客户端消息当前无业务语义，仅用于维持连接并检测断开
            await websocket.receive_text()
    except WebSocketDisconnect:
        await hub.disconnect(websocket)
    except Exception:
        await hub.disconnect(websocket)


__all__ = ["router"]
