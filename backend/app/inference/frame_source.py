"""帧源：向推理端口提供 `(timestamp_ms, jpeg_bytes)`。

当前只实现 `JpegSequenceSource`：读取预先按固定步长抽取好的 JPEG 序列。
视频解码依赖（opencv / av）**刻意不引入应用依赖**，解帧由外部工具完成，
应用只负责按 25 Hz 网格送帧。将来接真实摄像头时新增 `CameraSource`，
`FrameSource` 协议不变。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


class FrameSource(Protocol):
    """按顺序产帧的源；`next_frame` 返回 None 表示已结束。"""

    def start(self) -> None: ...

    def next_frame(self) -> tuple[int, bytes] | None: ...

    def close(self) -> None: ...


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


__all__ = ["FrameSource", "JpegSequenceSource", "JPEG_SUFFIXES"]
