"""帧源：向推理端口提供 `(timestamp_ms, jpeg_bytes)`。

已实现两种源：

- `JpegSequenceSource`：读取预先按固定步长抽取好的 JPEG 序列（离线分析）；
- `CameraSource`（`camera_source.py`）：本地/网络摄像头，按 25 Hz 网格产帧（实时）。

两者都只产出 JPEG 字节，推理端口不关心图像从哪来。视频解码依赖（opencv / av）
**刻意不引入应用依赖**：离线素材由外部工具解帧，实时摄像头才需要 opencv。

`PreviewTap` 是 `FrameSource` 装饰器：把**送进推理的同一份 JPEG 字节**留一份给
预览通道，并带上序号与 sha256。预览与推理因此可以逐帧对齐——看到的就是模型
看到的，不需要"再抓一帧摄像头"导致画面与结论错位。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


class FrameSourceError(RuntimeError):
    """帧源不可用：摄像头打不开、读帧失败、重连耗尽。

    与"素材读完了"（`next_frame()` 返回 None）必须区分开：前者要重试并上报
    降级，后者是正常结束。
    """


class FrameSource(Protocol):
    """按顺序产帧的源；`next_frame` 返回 None 表示已结束。"""

    def start(self) -> None: ...

    def next_frame(self) -> tuple[int, bytes] | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PreviewFrame:
    """最近一帧送给推理的画面快照。

    `sequence` 是**送进推理的帧序号**（从 0 开始、单调递增），不是摄像头序号；
    `sha256` 是 `jpeg` 的摘要，用来证明预览与推理吃的是同一份字节。
    """

    sequence: int
    timestamp_ms: int
    jpeg: bytes
    sha256: str
    captured_monotonic: float

    @property
    def size_bytes(self) -> int:
        return len(self.jpeg)


class PreviewTap:
    """`FrameSource` 装饰器：记录最近一帧供预览读取。

    只保存最新一帧（旧帧直接丢弃），预览端消费慢也不会拖住推理线程。
    `latest()` 返回的字节对象与推理拿到的**是同一个 `bytes`**，不是重新编码的副本。
    """

    def __init__(self, inner: FrameSource, *, clock=time.monotonic) -> None:
        self._inner = inner
        self._clock = clock
        self._lock = threading.Lock()
        self._latest: PreviewFrame | None = None
        self._sequence = 0
        self._frames_out = 0
        self._last_read_error: str | None = None

    # ------------------------------------------------------------ 委托

    def start(self) -> None:
        self._inner.start()

    def close(self) -> None:
        self._inner.close()

    def next_frame(self) -> tuple[int, bytes] | None:
        try:
            frame = self._inner.next_frame()
        except FrameSourceError as error:
            with self._lock:
                self._last_read_error = str(error)[:400]
            raise
        if frame is None:
            return None
        timestamp_ms, jpeg = frame
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
            self._frames_out += 1
            self._last_read_error = None
            self._latest = PreviewFrame(
                sequence=sequence,
                timestamp_ms=int(timestamp_ms),
                jpeg=jpeg,
                sha256=hashlib.sha256(jpeg).hexdigest(),
                captured_monotonic=self._clock(),
            )
        return frame

    # ------------------------------------------------------------ 预览访问

    def latest(self) -> PreviewFrame | None:
        with self._lock:
            return self._latest

    def stats(self) -> dict[str, object]:
        with self._lock:
            latest = self._latest
            return {
                "frames_out": self._frames_out,
                "sequence": None if latest is None else latest.sequence,
                "source_timestamp_ms": None if latest is None else latest.timestamp_ms,
                "sha256": None if latest is None else latest.sha256,
                "bytes": None if latest is None else latest.size_bytes,
                "age_ms": None if latest is None else round((self._clock() - latest.captured_monotonic) * 1000.0, 1),
                "last_read_error": self._last_read_error,
            }

    def reset(self) -> None:
        """换源/重开会话时清掉旧画面，避免预览停留在上一段素材。"""

        with self._lock:
            self._latest = None
            self._sequence = 0
            self._frames_out = 0
            self._last_read_error = None

    def status(self) -> dict[str, object]:
        """帧源自身状态 + 预览对齐信息（供 system.status / preview.state 使用）。"""

        inner = None
        reporter = getattr(self._inner, "status", None)
        if callable(reporter):
            try:
                inner = reporter()
            except Exception:  # noqa: BLE001 - 状态查询不能影响推流
                inner = None
        return {"preview": self.stats(), "source": inner}


class JpegSequenceSource:
    """按文件名排序读取目录中的 JPEG，时间戳 = 序号 × step_ms。

    时间戳从 0 开始、严格递增，满足 live 服务的会话约束；不要用墙钟，
    否则 25 Hz 网格会出现空洞，多数分支会因为证据过期而不可用。
    """

    def __init__(self, directory: str | Path, *, step_ms: int = 40) -> None:
        if step_ms <= 0:
            raise ValueError("step_ms 必须为正数")
        self._directory = Path(directory)
        self._step_ms = int(step_ms)
        self._files: list[Path] = []
        self._index = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if not self._directory.is_dir():
            raise ValueError(f"JPEG 序列目录不存在: {self._directory}")
        files = sorted(
            path for path in self._directory.iterdir()
            if path.is_file() and path.suffix.lower() in JPEG_SUFFIXES
        )
        if not files:
            raise ValueError(f"JPEG 序列目录为空: {self._directory}")
        self._files = files
        self._index = 0
        self._started = True

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def total(self) -> int:
        return len(self._files)

    @property
    def remaining(self) -> int:
        return max(0, len(self._files) - self._index)

    def next_frame(self) -> tuple[int, bytes] | None:
        if not self._started:
            self.start()
        if self._index >= len(self._files):
            return None
        path = self._files[self._index]
        timestamp_ms = self._index * self._step_ms
        self._index += 1
        return timestamp_ms, path.read_bytes()

    def close(self) -> None:
        self._started = False


__all__ = [
    "FrameSource",
    "FrameSourceError",
    "JpegSequenceSource",
    "JPEG_SUFFIXES",
    "PreviewFrame",
    "PreviewTap",
]

