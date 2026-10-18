"""监控服务与推理引擎。

`InferenceEngine` 负责：
- 启动时恢复或创建活动监测会话，并播种配置与模型元数据（NFR-002）
- 以固定频率从 `InferencePort` 取快照，驱动事件生命周期并推送到 WebSocket Hub
- 汇总 FPS / 延迟 / 丢帧率等可观测指标（NFR-006）

注意：推理实现为脚本化占位，`inference_mode` 与 `degraded` 会如实上报。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any

from ..config import AppConfig
from ..database import Database
from ..inference.port import InferencePort
from ..repositories.models_registry import ModelRegistryRepository
from ..repositories.sessions import SessionRepository
from ..repositories.settings import SettingsRepository
from ..schemas import RiskSnapshot
from ..timeutil import to_iso_z, utcnow
from .events import EventTracker

MAX_LATENCY_SAMPLES = 240
MAX_TICK_SAMPLES = 120


class InferenceEngine:
    def __init__(
        self,
        db: Database,
        config: AppConfig,
        hub,
        inference: InferencePort,
    ) -> None:
        self._db = db
        self._config = config
        self._hub = hub
        self._inference = inference
        self._tracker = EventTracker(db)

        self._session_id: str | None = None
        self._sequence = 0
        self._latest: RiskSnapshot | None = None
        self._task: asyncio.Task | None = None
        self._state = "stopped"

        self._latencies: deque[float] = deque(maxlen=MAX_LATENCY_SAMPLES)
        self._tick_times: deque[float] = deque(maxlen=MAX_TICK_SAMPLES)
        self._last_heartbeat = 0.0

    # ------------------------------------------------------------ 生命周期

    async def start(self) -> None:
        if self._task is not None:
            return

        now = utcnow()
        now_iso = to_iso_z(now)
        with self._db.session() as session:
            SettingsRepository(session).ensure_defaults(now_iso=now_iso)
            ModelRegistryRepository(session).ensure_stub_models(now_iso=now_iso)
            sessions = SessionRepository(session)
            existing = sessions.get_active()
            if existing is not None:
                # 服务重启后恢复既有活动会话，避免重复写入
                self._session_id = existing.session_id
            else:
                bundle_version = SettingsRepository(session).get_value("model.bundle_version")
                created = sessions.create(
                    session_id=str(uuid.uuid4()),
                    started_at=now_iso,
                    camera_id="cam-local",
                    device_id="edge-local",
                    model_bundle_version=str(bundle_version or "unknown"),
                )
                self._session_id = created.session_id

        self._state = "running"
        self._task = asyncio.create_task(self._run(), name="driveguard-inference")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._state = "stopped"

        if self._session_id:
            with self._db.session() as session:
                SessionRepository(session).end(self._session_id, to_iso_z(utcnow()))

    # ------------------------------------------------------------ 主循环

    async def _run(self) -> None:
        interval = self._config.snapshot_interval_seconds
        try:
            while True:
                tick_started = time.perf_counter()
                now = utcnow()
                snapshot = self._inference.next_snapshot(
                    session_id=self._session_id or "unknown",
                    sequence=self._sequence,
                    now=now,
                )
                self._sequence += 1
                self._latest = snapshot
                self._tick_times.append(time.monotonic())

                change = self._tracker.process(snapshot)
                await self._hub.broadcast_snapshot(snapshot)
                if change is not None:
                    await self._hub.broadcast_event_changed(change)

                self._latencies.append((time.perf_counter() - tick_started) * 1000.0)
                await self._maybe_heartbeat(now)

                elapsed = time.perf_counter() - tick_started
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:  # pragma: no cover - 正常关闭路径
            raise

    async def _maybe_heartbeat(self, now: datetime) -> None:
        monotonic_now = time.monotonic()
        if monotonic_now - self._last_heartbeat < self._config.heartbeat_seconds:
            return
        self._last_heartbeat = monotonic_now
        await self._hub.broadcast_heartbeat(now)

    # ------------------------------------------------------------ 只读访问

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def client_count(self) -> int:
        return self._hub.client_count

    def latest(self) -> RiskSnapshot | None:
        return self._latest

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "sequence": self._sequence,
            "processed_fps": self._processed_fps(),
            "latency_p50_ms": self._percentile(50),
            "latency_p95_ms": self._percentile(95),
            "dropped_frame_rate": self._dropped_frame_rate(),
            "inference_mode": self._inference.mode,
            "degraded": self._inference.degraded,
        }

    @property
    def inference_mode(self) -> str:
        return self._inference.mode

    @property
    def degraded(self) -> bool:
        return self._inference.degraded

    # ------------------------------------------------------------ 指标计算

    def _processed_fps(self) -> float:
        if len(self._tick_times) < 2:
            return 0.0
        span = self._tick_times[-1] - self._tick_times[0]
        if span <= 0:
            return 0.0
        return round((len(self._tick_times) - 1) / span, 3)

    def _percentile(self, percentile: float) -> float:
        if not self._latencies:
            return 0.0
        ordered = sorted(self._latencies)
        index = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
        return round(ordered[index], 3)

    def _dropped_frame_rate(self) -> float:
        dropped = getattr(self._hub, "dropped_total", 0)
        if self._sequence <= 0:
            return 0.0
        return round(dropped / self._sequence, 6)


__all__ = ["InferenceEngine"]
