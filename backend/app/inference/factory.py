"""推理实现工厂。

`DGB_INFERENCE_MODE=fake`（默认）使用脚本化回放；`=live` 使用真实模型服务。
`DGB_LIVE_SOURCE` 决定实时帧从哪来：`frames`（预抽好的 JPEG 序列）或 `camera`
（本地摄像头）。两种都包一层 `PreviewTap`，让预览看到的就是送进模型的字节。

live 模式下配置缺失时**立即报错**，而不是悄悄退回 fake——否则会把占位数据
误当成模型结论（业务规则 1）。
"""

from __future__ import annotations

import logging

from ..config import AppConfig
from .camera_source import DEFAULT_BACKEND, CameraSource
from .fake import FakeInference
from .frame_source import FrameSource, JpegSequenceSource, PreviewTap
from .live_client import LiveCamera, LiveClient
from .live_port import LiveInference
from .port import InferencePort
from .scenarios import get_scenario

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("fake", "live")
SUPPORTED_SOURCES = ("frames", "camera")



def _parse_size(raw: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in raw.lower().split("x"))
    except ValueError:
        raise ValueError(f"DGB_LIVE_IMAGE_SIZE 需形如 1280x720，收到: {raw}") from None
    if not (16 <= width <= 1920 and 16 <= height <= 1920 and width * height <= 1920 * 1080):
        raise ValueError(f"DGB_LIVE_IMAGE_SIZE 超出 live 服务允许范围: {raw}")
    return width, height


def _parse_roi(raw: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(part) for part in raw.split(","))
    except ValueError:
        raise ValueError(f"DGB_LIVE_ROI 需形如 0.15,0,0.75,1，收到: {raw}") from None
    if len(values) != 4 or not (0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1):
        raise ValueError(f"DGB_LIVE_ROI 非法: {raw}")
    return values  # type: ignore[return-value]


def _build_source(config: AppConfig, source_kind: str | None = None) -> tuple[FrameSource, str]:
    """按来源类型构造帧源；返回值第二项是实际生效的来源名。"""

    kind = (source_kind or config.live_source or "frames").lower()
    if kind not in SUPPORTED_SOURCES:
        raise ValueError(f"DGB_LIVE_SOURCE 只支持 {SUPPORTED_SOURCES}，收到: {kind}")

    if kind == "camera":
        source = CameraSource(
            config.camera_device_index,
            width=config.camera_width,
            height=config.camera_height,
            step_ms=config.live_step_ms,
            jpeg_quality=config.camera_jpeg_quality,
            backend=config.camera_backend or DEFAULT_BACKEND,
        )
        return source, kind

    if not config.frame_sequence_dir:
        raise ValueError("DGB_LIVE_SOURCE=frames 时必须配置 DGB_FRAME_SEQUENCE（JPEG 序列目录）")
    return JpegSequenceSource(config.frame_sequence_dir, step_ms=config.live_step_ms), kind


def build_inference(config: AppConfig, *, source_kind: str | None = None) -> InferencePort:
    mode = (config.inference_mode or "fake").lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"DGB_INFERENCE_MODE 只支持 {SUPPORTED_MODES}，收到: {mode}")

    if mode == "fake":
        return FakeInference(get_scenario(config.inference_scenario))

    if not config.live_token_file:
        raise ValueError("DGB_INFERENCE_MODE=live 时必须配置 DGB_LIVE_TOKEN_FILE")

    client = LiveClient.from_token_file(config.live_url, config.live_token_file)
    camera = LiveCamera(
        view=config.live_view,
        image_size=_parse_size(config.live_image_size),
        driver_roi=_parse_roi(config.live_roi),
    )
    # 预览与推理共用同一个帧源：预览里显示的每一帧，就是模型刚收到的那一帧。
    source, resolved = _build_source(config, source_kind)
    tap = PreviewTap(source)
    logger.info(
        "live 推理已启用: url=%s source=%s size=%s view=%s step_ms=%s",
        config.live_url,
        config.frame_sequence_dir if resolved == "frames" else f"camera:{config.camera_device_index}",
        config.live_image_size,
        config.live_view,
        config.live_step_ms,
    )
    return LiveInference(client, camera, tap, preview_tap=tap)


__all__ = ["build_inference", "SUPPORTED_MODES", "SUPPORTED_SOURCES"]
