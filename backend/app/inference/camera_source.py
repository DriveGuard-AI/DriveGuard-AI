"""摄像头帧源：按 25 Hz 网格产帧，供 live 推理与预览共用。

三条约束决定了这里的实现：

1. **时间戳必须落在固定网格上**（`index × step_ms`），不能用墙钟。live 服务的
   多数分支按 40 ms 窗口对齐，墙钟抖动会让证据过期或出现空洞。
2. **不追帧、只丢时间片**。相机比网格快时按网格等；相机比网格慢时直接进入下一
   个时间片（时间戳仍连续），而不是把旧帧积压后一次性灌进去——积压会让"模型看到
   的画面"与"实时画面"越来越远。
3. **不能把故障伪装成结束**。`next_frame()` 返回 None 在协议里表示"素材读完"，
   会导致会话正常收尾。摄像头读不到帧属于故障，必须抛 `FrameSourceError`，
   交由推理泵重试并上报降级。

opencv 采用**惰性导入**：离线分析/单元测试不需要 opencv 也能导入本模块。
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Callable

from .frame_source import FrameSourceError

logger = logging.getLogger(__name__)

# Windows 上 CAP_DSHOW 的打开延迟明显低于 MSMF，且对"设备被占用"报错更明确。
DEFAULT_BACKEND = "dshow" if sys.platform == "win32" else "any"
SUPPORTED_BACKENDS = ("any", "dshow", "msmf", "v4l2", "gstreamer")


def _resolve_backend(cv2, name: str) -> int:
    """把配置里的后端名映射成 opencv 常量；未知名称回退到自动选择。"""

    mapping = {
        "dshow": getattr(cv2, "CAP_DSHOW", None),
        "msmf": getattr(cv2, "CAP_MSMF", None),
        "v4l2": getattr(cv2, "CAP_V4L2", None),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
    }
    return mapping.get(name) or getattr(cv2, "CAP_ANY", 0)


class CameraSource:
    """本地/网络摄像头帧源。

    `capture_factory`、`clock`、`sleep` 都可注入，测试因此不需要真实摄像头。
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        step_ms: int = 40,
        jpeg_quality: int = 85,
        backend: str = DEFAULT_BACKEND,
        capture_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        read_retries: int = 2,
        reopen_attempts: int = 2,
        reopen_delay_seconds: float = 0.5,
        max_stale_drops: int = 5,
    ) -> None:
        if step_ms <= 0:
            raise ValueError("step_ms 必须为正数")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError("jpeg_quality 需在 1..100 之间")
        self._device_index = int(device_index)
        self._width = int(width)
        self._height = int(height)
        self._step_ms = int(step_ms)
        self._quality = int(jpeg_quality)
        self._backend = (backend or "any").lower()
        if self._backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"不支持的摄像头后端: {backend}（可选 {SUPPORTED_BACKENDS}）")
        self._capture_factory = capture_factory
        self._clock = clock
        self._sleep = sleep
        self._read_retries = max(0, int(read_retries))
        self._reopen_attempts = max(0, int(reopen_attempts))
        self._reopen_delay = max(0.0, float(reopen_delay_seconds))
        self._max_stale_drops = max(0, int(max_stale_drops))

        self._capture: Any | None = None
        self._started = False
        self._origin = 0.0
        self._index = 0
        self._frames = 0
        self._reconnects = 0
        self._last_error: str | None = None
        self._cv2: Any | None = None

    # ------------------------------------------------------------ 生命周期

    def start(self) -> None:
        if self._started:
            return
        self._open()
        self._started = True
        self._origin = self._clock()
        self._index = 0
        self._frames = 0

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._started = False
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001 - 释放失败不影响关闭
                logger.debug("释放摄像头失败（已忽略）", exc_info=True)

    # ------------------------------------------------------------ 产帧

    def next_frame(self) -> tuple[int, bytes] | None:
        if not self._started:
            self.start()

        # 对齐网格：早于目标时刻就等待；已经落后则不再补眠。
        target = self._origin + self._index * (self._step_ms / 1000.0)
        now = self._clock()
        if now < target:
            self._sleep(target - now)
        else:
            self._drop_stale(now - target)

        jpeg = self._read_jpeg()
        timestamp_ms = self._index * self._step_ms
        self._index += 1
        self._frames += 1
        return timestamp_ms, jpeg

    def _drop_stale(self, behind_seconds: float) -> None:
        """相机驱动会缓存旧帧，落后多个时间片时先丢弃，避免越跑越旧。"""

        capture = self._capture
        if capture is None or self._max_stale_drops <= 0:
            return
        stale = int(behind_seconds / (self._step_ms / 1000.0))
        if stale < 1:
            return
        grab = getattr(capture, "grab", None)
        if not callable(grab):
            return
        for _ in range(min(stale, self._max_stale_drops)):
            if not grab():
                break

    def _read_jpeg(self) -> bytes:
        last_error = "unknown"
        for attempt in range(self._read_retries + 1):
            capture = self._capture
            if capture is not None:
                try:
                    ok, frame = capture.read()
                except Exception as error:  # noqa: BLE001 - 驱动层异常一律视为读失败
                    ok, frame = False, None
                    last_error = f"{type(error).__name__}: {error}"
                if ok and frame is not None:
                    self._last_error = None
                    return self._encode(frame)
                last_error = "read() 未返回帧"
            if attempt < self._read_retries:
                if self._reopen_delay:
                    self._sleep(self._reopen_delay)
                try:
                    self._reopen()
                except FrameSourceError as error:
                    last_error = str(error)
        self._last_error = last_error[:400]
        raise FrameSourceError(f"摄像头读帧失败（device={self._device_index}）: {last_error}")

    def _encode(self, frame: Any) -> bytes:
        cv2 = self._require_cv2()
        try:
            ok, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
            )
        except Exception as error:  # noqa: BLE001 - 编码异常统一收敛
            raise FrameSourceError(f"JPEG 编码失败: {type(error).__name__}: {error}") from None
        if not ok:
            raise FrameSourceError("JPEG 编码失败")
        return bytes(buffer.tobytes())

    # ------------------------------------------------------------ 打开与重连

    def _open(self) -> None:
        self.close()
        cv2 = self._require_cv2()
        try:
            if self._capture_factory is not None:
                capture = self._capture_factory()
            else:
                capture = cv2.VideoCapture(
                    self._device_index, _resolve_backend(cv2, self._backend)
                )
        except Exception as error:  # noqa: BLE001 - 打开失败统一报 FrameSourceError
            raise FrameSourceError(
                f"摄像头打开失败（device={self._device_index}）: {type(error).__name__}: {error}"
            ) from None

        if capture is None or not self._is_opened(capture):
            try:
                if capture is not None:
                    capture.release()
            finally:
                raise FrameSourceError(
                    f"摄像头不可用（device={self._device_index}, backend={self._backend}）"
                )

        for prop, value in (
            (getattr(cv2, "CAP_PROP_FRAME_WIDTH", None), self._width),
            (getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None), self._height),
        ):
            if prop is not None:
                try:
                    capture.set(prop, value)
                except Exception:  # noqa: BLE001 - 属性设置失败不致命
                    logger.debug("设置摄像头属性失败（已忽略）", exc_info=True)

        self._capture = capture
        # 预热一帧：有些设备 open() 成功但首帧总是失败。
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            self._capture = None
            raise FrameSourceError(
                f"摄像头已打开但读不到画面（device={self._device_index}）"
            )

    def _reopen(self) -> None:
        last_error: str | None = None
        for _ in range(max(1, self._reopen_attempts)):
            try:
                self._open()
            except FrameSourceError as error:
                last_error = str(error)
                if self._reopen_delay:
                    self._sleep(self._reopen_delay)
                continue
            self._reconnects += 1
            logger.warning("摄像头已重连（device=%s，第 %s 次）", self._device_index, self._reconnects)
            return
        raise FrameSourceError(last_error or "摄像头重连失败")

    @staticmethod
    def _is_opened(capture: Any) -> bool:
        checker = getattr(capture, "isOpened", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:  # noqa: BLE001 - 视为未打开
                return False
        return True

    def _require_cv2(self):
        if self._cv2 is None:
            try:
                import cv2  # noqa: PLC0415 - 惰性导入，避免离线场景强依赖
            except ImportError:
                raise FrameSourceError(
                    "未安装 opencv，无法使用摄像头帧源（pip install opencv-python-headless）"
                ) from None
            self._cv2 = cv2
        return self._cv2

    # ------------------------------------------------------------ 状态

    def status(self) -> dict[str, Any]:
        return {
            "source": "camera",
            "device_index": self._device_index,
            "backend": self._backend,
            "width": self._width,
            "height": self._height,
            "step_ms": self._step_ms,
            "frames": self._frames,
            "reconnects": self._reconnects,
            "opened": self._capture is not None,
            "last_error": self._last_error,
        }


__all__ = ["CameraSource", "DEFAULT_BACKEND", "SUPPORTED_BACKENDS"]
