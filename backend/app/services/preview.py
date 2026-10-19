"""实时预览服务（API-010）。

预览通道的定位很窄：**回答"模型此刻到底看到了什么"**。因此：

- 画面来自 `PreviewTap` 记录的、**已经送进推理的同一份 JPEG 字节**，
  不是重新抓一帧摄像头（那会与模型输入错位）；
- 每个 part 带 `X-Seq`、`X-Timestamp-Ms`、`X-Sha256`，前端可以据此确认
  "显示的这帧 = 序号 N 的推理输入"；
- 只读内存最新帧，不落盘、不参与推理，慢客户端丢旧保新，绝不反压推理线程。

开关与上限来自设置表（`preview.enabled` / `preview.max_fps` / `preview.max_clients`），
运行时生效，不需要重启；关闭开关会终止已有连接。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

from ..config import AppConfig
from ..errors import (
    MonitoringNotActiveError,
    PipelineUnavailableError,
    PreviewDisabledError,
)
from ..schemas import PreviewState
from .monitoring import InferenceEngine
from .settings import SettingsService

logger = logging.getLogger(__name__)

BOUNDARY = "frame"
# 没有新帧时的收流时限：超过即关闭连接（不是 200 后的 JSON 错误，避免污染 JPEG 流）。
DEFAULT_IDLE_TIMEOUT_SECONDS = 5.0
# 检查开关/会话状态的间隔：关闭预览后最多等这么久就断开。
DEFAULT_RECHECK_SECONDS = 1.0
# 没有新帧时的轮询间隔。
DEFAULT_IDLE_POLL_SECONDS = 0.05


class PreviewService:
    def __init__(
        self,
        config: AppConfig,
        settings: SettingsService,
        engine: InferenceEngine,
        *,
        idle_timeout_seconds: float = DEFAULT_IDLE_TIMEOUT_SECONDS,
        recheck_seconds: float = DEFAULT_RECHECK_SECONDS,
        idle_poll_seconds: float = DEFAULT_IDLE_POLL_SECONDS,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._settings = settings
        self._engine = engine
        self._idle_timeout = float(idle_timeout_seconds)
        self._recheck = float(recheck_seconds)
        self._idle_poll = float(idle_poll_seconds)
        self._sleep = sleep
        self._clients = 0

    # ------------------------------------------------------------ 配置读取

    def enabled(self) -> bool:
        value = self._settings.read_runtime_value("preview.enabled")
        return bool(value)

    def max_fps(self) -> float:
        value = self._settings.read_runtime_value("preview.max_fps")
        try:
            fps = float(value)
        except (TypeError, ValueError):
            fps = self._config.preview_max_fps
        return min(30.0, max(0.5, fps))

    def max_clients(self) -> int:
        value = self._settings.read_runtime_value("preview.max_clients")
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = self._config.preview_max_clients
        return max(1, count)

    # ------------------------------------------------------------ 状态

    @property
    def clients(self) -> int:
        return self._clients

    def state(self) -> PreviewState:
        tap = self._engine.preview_tap()
        stats = _tap_stats(tap)
        pipeline = _pipeline_state(self._engine)
        enabled = self.enabled()
        max_clients = self.max_clients()
        has_frame = stats.get("sequence") is not None
        return PreviewState(
            enabled=enabled,
            available=bool(enabled and pipeline == "realtime" and has_frame and self._clients < max_clients),
            clients=self._clients,
            max_clients=max_clients,
            max_fps=self.max_fps(),
            pipeline=pipeline,
            source="camera" if (self._config.live_source or "").lower() == "camera" else "frames",
            device_index=self._config.camera_device_index,
            sequence=stats.get("sequence"),
            source_timestamp_ms=stats.get("source_timestamp_ms"),
            sha256=stats.get("sha256"),
            bytes=stats.get("bytes"),
            age_ms=stats.get("age_ms"),
            frames_out=int(stats.get("frames_out") or 0),
            last_error=stats.get("last_read_error"),
        )

    # ------------------------------------------------------------ 连接准入

    def ensure_ready(self) -> None:
        """按文档约定顺序校验：开关 → 会话 → 来源/设备 → 连接数。

        必须在返回 `StreamingResponse` **之前**调用，这样错误还是 JSON；
        一旦开始推流，任何故障只能关闭连接（不能把 JSON 混进 JPEG 流）。
        """

        if not self.enabled():
            raise PreviewDisabledError()
        if self._engine.session_id is None:
            raise MonitoringNotActiveError()

        pipeline = _pipeline_state(self._engine)
        if pipeline != "realtime":
            raise PipelineUnavailableError(
                "离线回放期间没有实时画面；请使用分析回放页查看素材"
                if pipeline == "playback"
                else "推理管线未运行，无法提供实时预览"
            )

        tap = self._engine.preview_tap()
        if tap is None:
            raise PipelineUnavailableError("当前推理实现未提供帧源，无法预览（例如 inference_mode=fake）")
        if tap.latest() is None:
            raise PipelineUnavailableError("摄像头画面尚未就绪，请稍后重试")
        if self._clients >= self.max_clients():
            raise PipelineUnavailableError(f"预览连接数已达上限（{self.max_clients()}）")

    # ------------------------------------------------------------ 推流

    async def stream(
        self,
        *,
        max_parts: int | None = None,
        is_disconnected: Callable[[], Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """产出 `multipart/x-mixed-replace` 分片。

        `max_parts` 仅用于测试与一次性抓取：正常调用方不传，连接由客户端
        断开或空闲超时结束。
        """

        self._clients += 1
        sent_sequence: int | None = None
        parts = 0
        interval = 1.0 / self.max_fps()
        idle_elapsed = 0.0
        since_check = 0.0
        try:
            while True:
                if is_disconnected is not None:
                    try:
                        if await _maybe_await(is_disconnected()):
                            break
                    except Exception:  # noqa: BLE001 - 断开检测失败按继续处理
                        pass

                tap = self._engine.preview_tap()
                frame = tap.latest() if tap is not None else None
                if frame is not None and frame.sequence != sent_sequence:
                    yield _part(frame)
                    sent_sequence = frame.sequence
                    parts += 1
                    idle_elapsed = 0.0
                    if max_parts is not None and parts >= max_parts:
                        break
                    await _maybe_await(self._sleep(interval))
                    continue

                idle_elapsed += self._idle_poll
                if idle_elapsed >= self._idle_timeout:
                    logger.info("预览空闲超过 %.1fs，关闭连接", self._idle_timeout)
                    break

                # 开关被关掉或切到回放时主动断开，不继续占用摄像头带宽。
                since_check += self._idle_poll
                if since_check >= self._recheck:
                    since_check = 0.0
                    if not self.enabled() or self._engine.preview_tap() is None:
                        break

                await _maybe_await(self._sleep(self._idle_poll))
            if parts:
                yield f"--{BOUNDARY}--\r\n".encode("ascii")
        finally:
            self._clients = max(0, self._clients - 1)


def _part(frame) -> bytes:
    header = (
        f"--{BOUNDARY}\r\n"
        "Content-Type: image/jpeg\r\n"
        f"Content-Length: {frame.size_bytes}\r\n"
        f"X-Seq: {frame.sequence}\r\n"
        f"X-Timestamp-Ms: {frame.timestamp_ms}\r\n"
        f"X-Sha256: {frame.sha256}\r\n"
        "\r\n"
    ).encode("ascii")
    return header + frame.jpeg + b"\r\n"


def _tap_stats(tap) -> dict[str, Any]:
    if tap is None:
        return {}
    reporter = getattr(tap, "stats", None)
    if not callable(reporter):
        return {}
    try:
        value = reporter()
    except Exception:  # noqa: BLE001 - 状态查询不能阻断推流
        return {}
    return value if isinstance(value, dict) else {}


def _pipeline_state(engine: InferenceEngine) -> str:
    if engine.in_realtime_mode:
        return "realtime"
    return "playback" if getattr(engine, "status", lambda: {})().get("playback") else "stopped"


async def _maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


__all__ = ["BOUNDARY", "PreviewService"]
