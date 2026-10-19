"""视频 → 固定栅格 JPEG 序列。

抽帧放在应用侧而不是让用户先跑一遍外部脚本：否则"上传一段视频就能分析"这件事
并不成立。规则与模型仓库的 `extract_frames.py` 一致（`index * step_ms` 时间戳、
补零命名、按 25 Hz 栅格采样），因为权重是按这个栅格训练和校准的。

解码是纯 CPU 的阻塞工作，调用方必须把它放进线程（`asyncio.to_thread`），
不要在事件循环里直接调用。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 模型权重的时序契约是 25 Hz（40 ms）；抽帧步长必须是它的整数倍，
# 否则送进推理的时间戳会落在 25 Hz 栅格之外。见 KL-11。
GRID_STEP_MS = 40
STEP_MS_CHOICES = (40, 80, 120, 160, 200)
DEFAULT_SIZE = "1280x720"
DEFAULT_QUALITY = 85
FALLBACK_FPS = 25.0
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})


class ExtractionError(Exception):
    """素材不可解码或参数不合法，属于调用方输入问题。"""


class DecoderUnavailableError(Exception):
    """服务端缺少解码依赖，属于部署问题。"""


def _import_cv2():
    try:
        import cv2  # noqa: PLC0415 - 延迟导入，缺依赖时其它接口仍可用
    except ImportError:  # pragma: no cover - 部署错误
        raise DecoderUnavailableError(
            "服务端缺少 opencv-python-headless，无法解码视频"
        ) from None
    return cv2


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = str(value).lower().split("x")
        width, height = int(width_text), int(height_text)
    except (AttributeError, ValueError):
        raise ExtractionError(f"size 必须形如 1280x720，收到 {value!r}") from None
    if width <= 0 or height <= 0:
        raise ExtractionError(f"size 必须为正数，收到 {value!r}")
    if width % 2 or height % 2:
        # 奇数边长在部分编码器上会被静默裁剪，导致下游坐标偏移。
        raise ExtractionError(f"size 的宽高必须是偶数，收到 {value!r}")
    return width, height


def validate_options(*, step_ms: int, size: str, quality: int) -> tuple[int, int]:
    if step_ms not in STEP_MS_CHOICES:
        raise ExtractionError(f"step_ms 必须是 {list(STEP_MS_CHOICES)} 之一（25 Hz 栅格的整数倍）")
    if not 1 <= quality <= 100:
        raise ExtractionError(f"quality 必须在 1..100，收到 {quality}")
    return parse_size(size)


def extract_frames(
    video: Path,
    output: Path,
    *,
    step_ms: int = GRID_STEP_MS,
    size: str = DEFAULT_SIZE,
    quality: int = DEFAULT_QUALITY,
    max_frames: int | None = None,
) -> int:
    """把 `video` 抽成 `output/000000.jpg...`，返回写入帧数。

    `max_frames` 既是资源上限也是"视频过长"的判定条件：先按时长估算挡一次，
    再在循环里硬停，避免为一个 1 小时视频写满磁盘之后才报错。
    """

    cv2 = _import_cv2()
    width, height = validate_options(step_ms=step_ms, size=size, quality=quality)
    step_seconds = step_ms / 1000.0

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ExtractionError(f"无法打开视频，可能不是视频文件或编码不受支持: {video.name}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or FALLBACK_FPS
        if max_frames:
            total = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if total > 0:
                estimated = int(total / fps / step_seconds)
                if estimated > max_frames * 1.2:
                    raise ExtractionError(
                        f"视频约 {estimated} 帧，超过上限 {max_frames} 帧"
                        f"（约 {max_frames * step_ms / 1000:.0f} 秒）"
                    )

        output.mkdir(parents=True, exist_ok=True)
        written = 0
        frame_index = 0
        next_emit = 0.0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            frame_index += 1
            if timestamp + 1e-9 < next_emit:
                continue
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            target = output / f"{written:06d}.jpg"
            if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]):
                raise ExtractionError(f"写入帧失败: {target.name}")
            written += 1
            if max_frames and written > max_frames:
                raise ExtractionError(
                    f"视频超过上限 {max_frames} 帧（约 {max_frames * step_ms / 1000:.0f} 秒）"
                )
            next_emit = written * step_seconds
    finally:
        capture.release()

    if written == 0:
        raise ExtractionError("视频没有可解码的帧")
    return written


__all__ = [
    "DEFAULT_QUALITY",
    "DEFAULT_SIZE",
    "DecoderUnavailableError",
    "ExtractionError",
    "GRID_STEP_MS",
    "STEP_MS_CHOICES",
    "VIDEO_SUFFIXES",
    "extract_frames",
    "parse_size",
    "validate_options",
]
