"""WebSocket Hub。

`docs/api/API_DESIGN.md` 第 5 节的实现要点：
- 消息 envelope 含 `type / schema_version / sequence / sent_at / data`
- 默认最高推送 10 Hz，且不得超过推理快照产生率
- 客户端过慢时丢弃旧 `risk.snapshot`，保留最新状态，不阻塞推理
- `event.changed` 不得静默丢弃
- 连接就绪时发送 `connection.ready`
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import WebSocket

from ..config import WS_SCHEMA_VERSION, AppConfig
from ..schemas import RiskSnapshot
from ..timeutil import to_iso_z, utcnow
from ..services.events import EventChange

QUEUE_SIZE = 8
SEND_TIMEOUT_SECONDS = 2.0


@dataclass
class _Client:
    ws: WebSocket
    queue: asyncio.Queue[dict] = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_SIZE))
    sender: asyncio.Task | None = None
    session_id: str | None = None


class ConnectionManager:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._clients: dict[int, _Client] = {}
        self._sequence = 0
        self._last_snapshot_sent = 0.0
        self.dropped_total = 0

    # ------------------------------------------------------------ 连接管理

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def sequence(self) -> int:
        return self._sequence

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    async def connect(self, ws: WebSocket, *, session_id: str | None) -> None:
        await ws.accept()
        client = _Client(ws=ws, session_id=session_id)
        self._clients[id(client)] = client
        client.sender = asyncio.create_task(self._pump(client), name="driveguard-ws-sender")

        await self._send_now(
            client,
            self._envelope(
                "connection.ready",
                {
                    "heartbeat_seconds": self._config.heartbeat_seconds,
                    "max_hz": self._config.ws_max_hz,
                    "session_id": session_id,
                    "inference_mode": "fake" if session_id else "unknown",
                },
            ),
        )

    async def disconnect(self, ws: WebSocket) -> None:
        for key, client in list(self._clients.items()):
            if client.ws is ws:
                self._clients.pop(key, None)
                if client.sender is not None:
                    client.sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await client.sender
                break

    # ------------------------------------------------------------ 推送

    async def broadcast_snapshot(self, snapshot: RiskSnapshot) -> None:
        # 节流：默认最高 10 Hz
        throttle = 1.0 / self._config.ws_max_hz
        loop_time = asyncio.get_running_loop().time()
        if loop_time - self._last_snapshot_sent < throttle:
            return
        self._last_snapshot_sent = loop_time

        # envelope 的 `sequence` 使用 Hub 全局单调计数器，保证同一连接上所有消息类型
        # （ready / snapshot / heartbeat / event.changed / system.status）严格递增；
        # 推理快照自身的序号保留在 payload 的 data.sequence 中（用于展示与丢帧判断）。
        message = self._envelope("risk.snapshot", snapshot.model_dump(mode="json"))
        for client in list(self._clients.values()):
            self._enqueue(client, message)

    async def broadcast_event_changed(self, change: EventChange) -> None:
        message = self._envelope("event.changed", change.to_payload())
        for client in list(self._clients.values()):
            # 事件变化消息不得静默丢弃
            self._enqueue(client, message, force=True)

    async def broadcast_heartbeat(self, moment: datetime | None = None) -> None:
        message = self._envelope("heartbeat", {"time": to_iso_z(moment or utcnow())})
        for client in list(self._clients.values()):
            self._enqueue(client, message)

    async def broadcast_system_status(self, payload: dict) -> None:
        message = self._envelope("system.status", payload)
        for client in list(self._clients.values()):
            self._enqueue(client, message)

    async def send_error(self, ws: WebSocket, code: str, message: str) -> None:
        for client in list(self._clients.values()):
            if client.ws is ws:
                self._enqueue(client, self._envelope("error", {"code": code, "message": message}), force=True)
                return

    # ------------------------------------------------------------ 内部

    def _envelope(self, message_type: str, data, *, sequence: int | None = None) -> dict:
        return {
            "type": message_type,
            "schema_version": WS_SCHEMA_VERSION,
            "sequence": self._next_sequence() if sequence is None else sequence,
            "sent_at": to_iso_z(utcnow()),
            "data": data,
        }

    def _enqueue(self, client: _Client, message: dict, *, force: bool = False) -> None:
        try:
            client.queue.put_nowait(message)
            return
        except asyncio.QueueFull:
            pass

        # 背压：丢弃最旧消息（通常是过期快照），保留最新状态
        with contextlib.suppress(asyncio.QueueEmpty):
            client.queue.get_nowait()
            self.dropped_total += 1

        try:
            client.queue.put_nowait(message)
        except asyncio.QueueFull:  # pragma: no cover - 极端竞争
            if not force:
                return

    async def _send_now(self, client: _Client, message: dict) -> None:
        try:
            await asyncio.wait_for(client.ws.send_json(message), timeout=SEND_TIMEOUT_SECONDS)
        except Exception:
            await self.disconnect(client.ws)

    async def _pump(self, client: _Client) -> None:
        try:
            while True:
                message = await client.queue.get()
                await asyncio.wait_for(client.ws.send_json(message), timeout=SEND_TIMEOUT_SECONDS)
        except asyncio.CancelledError:  # pragma: no cover - 正常关闭
            raise
        except Exception:
            await self.disconnect(client.ws)


__all__ = ["ConnectionManager"]
