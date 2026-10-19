"""监控服务与推理引擎。

`InferenceEngine` 负责：
- 启动时恢复或创建活动监测会话，并播种配置与模型元数据（NFR-002）
- 以固定频率从 `InferencePort` 取快照，驱动事件生命周期并推送到 WebSocket Hub
- 汇总 FPS / 延迟 / 丢帧率等可观测指标（NFR-006）

注意：推理实现为脚本化占位，`inference_mode` 与 `degraded` 会如实上报。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Callable

from ..config import AppConfig
from ..database import Database
from ..inference.port import AnalysisSource, InferencePort
from ..repositories.models_registry import ModelRegistryRepository
from ..repositories.sessions import SessionRepository
from ..repositories.settings import SettingsRepository
from ..schemas import RiskSnapshot
from ..timeutil import parse_iso, to_iso_z, utcnow
from .events import EventTracker

MAX_LATENCY_SAMPLES = 240
MAX_TICK_SAMPLES = 120

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        db: Database,
        config: AppConfig,
        hub,
        inference: InferencePort,
        builder: Callable[[AnalysisSource], InferencePort] | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._hub = hub
        self._inference = inference
        # 离线分析需要「换一个素材就换一个推理源」。构建器由外层注入，
        # 避免服务层反向依赖具体实现（fake/live）。
        self._builder = builder
        self._tracker = EventTracker(db)

        self._session_id: str | None = None
        self._session_started_at: datetime | None = None
        self._last_error: str | None = None
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
                self._session_started_at = parse_iso(existing.started_at)
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
                self._session_started_at = now

        self._state = "running"
        starter = getattr(self._inference, "start", None)
        if callable(starter):
            starter()
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
        closer = getattr(self._inference, "close", None)
        if callable(closer):
            closer()

        if self._session_id:
            with self._db.session() as session:
                SessionRepository(session).end(self._session_id, to_iso_z(utcnow()))

    async def run_analysis(self, source: AnalysisSource) -> str:
        """按素材来源重新开始分析，返回新会话 ID。

        离线分析要能反复跑不同素材，而推理源在启动时是固定的。这里停掉旧循环、
        关闭旧推理，用注入的构建器造一个新的，并**新开一个会话**：事件按会话隔离，
        时间轴锚点也必须跟着重置，否则新素材的第一帧会锚到上一个素材的起点上。

        入参是素材来源描述而不是路径：后续要加摄像头拉流或浏览器推流时，
        只需新增 `kind` 与对应实现，本方法不需要改。
        """

        if self._builder is None:
            raise RuntimeError("未注入推理构建器，无法在运行期切换素材")
        await self.stop()
        self._inference = self._builder(source)
        self._open_new_session()
        self._state = "running"
        starter = getattr(self._inference, "start", None)
        if callable(starter):
            starter()
        self._task = asyncio.create_task(self._run(), name="driveguard-inference")
        return self._session_id or ""

    def _open_new_session(self) -> None:
        now = utcnow()
        now_iso = to_iso_z(now)
        with self._db.session() as session:
            settings = SettingsRepository(session)
            settings.ensure_defaults(now_iso=now_iso)
            ModelRegistryRepository(session).ensure_stub_models(now_iso=now_iso)
            created = SessionRepository(session).create(
                session_id=str(uuid.uuid4()),
                started_at=now_iso,
                camera_id="cam-local",
                device_id="edge-local",
                model_bundle_version=str(settings.get_value("model.bundle_version") or "unknown"),
            )
        self._session_id = created.session_id
        self._session_started_at = now
        self._sequence = 0
        self._latest = None
        self._last_error = None
        self._latencies.clear()
        self._tick_times.clear()
        self._last_heartbeat = 0.0

    # ------------------------------------------------------------ 主循环

    async def _run(self) -> None:
        interval = self._config.snapshot_interval_seconds
        try:
            while True:
                tick_started = time.perf_counter()
                try:
                    await self._tick(tick_started)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 - 单轮失败不得终止采样
                    # 之前这里没有兜底：任何一轮异常都会让任务静默退出，而 status
                    # 仍报 running，表现为"服务在跑但快照不再更新"。
                    message = f"{type(error).__name__}: {error}"[:400]
                    self._last_error = message
                    logger.exception("推理循环本轮失败，继续下一轮: %s", message)
                elapsed = time.perf_counter() - tick_started
                await asyncio.sleep(max(0.0, interval - elapsed))
        except asyncio.CancelledError:  # pragma: no cover - 正常关闭路径
            raise

    async def _tick(self, tick_started: float) -> None:
        now = utcnow()
        snapshot = self._inference.next_snapshot(
            session_id=self._session_id or "unknown",
            sequence=self._sequence,
            now=now,
        )
        self._sequence += 1
        self._latest = snapshot
        self._tick_times.append(time.monotonic())

        # 离线素材跑完后时间轴停止推进，合并间隔不会再判超时。此时只收尾、
        # 不再根据快照建事件：否则每轮都会「建一个 → 立刻关掉」，产生大量 0ms 碎片。
        if self._source_completed():
            change = self._tracker.finalize(self._session_id or "unknown")
        else:
            change = self._tracker.process(snapshot, timeline_anchor=self._session_started_at)
        await self._hub.broadcast_snapshot(snapshot)
        if change is not None:
            await self._hub.broadcast_event_changed(change)

        self._latencies.append((time.perf_counter() - tick_started) * 1000.0)
        await self._maybe_heartbeat(now)

    def _source_completed(self) -> bool:
        """推理源是否已经跑完（离线分析专用；实时模式永远为 False）。"""

        reporter = getattr(self._inference, "status", None)
        if not callable(reporter):
            return False
        try:
            return (reporter() or {}).get("state") == "completed"
        except Exception:  # noqa: BLE001 - 状态查询失败按未完成处理
            return False

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
            # 最近一次主循环异常。非空表示快照可能在停滞，供运维判定。
            "last_error": self._last_error,
        }

    @property
    def inference_mode(self) -> str:
        return self._inference.mode

    def inference_status(self) -> dict[str, Any]:
        """推理实现的详细状态（可选）；失败时返回空字典，不影响主循环。"""

        reporter = getattr(self._inference, "status", None)
        if not callable(reporter):
            return {}
        try:
            value = reporter()
        except Exception:  # noqa: BLE001 - 状态查询不能拖垮监控
            return {}
        return value if isinstance(value, dict) else {}

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
