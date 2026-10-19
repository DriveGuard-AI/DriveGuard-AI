"""推理实现工厂。

`DGB_INFERENCE_MODE=fake`（默认）使用脚本化回放；`=live` 使用真实模型服务。
live 模式下配置缺失时**立即报错**，而不是悄悄退回 fake——否则会把占位数据
误当成模型结论（业务规则 1）。
"""

from __future__ import annotations

import logging

from ..config import AppConfig
from .fake import FakeInference
from .frame_source import JpegSequenceSource
from .live_client import LiveCamera, LiveClient
from .live_port import LiveInference
from .port import InferencePort
from .scenarios import get_scenario

logger = logging.getLogger(__name__)

SUPPORTED_MODES = ("fake", "live")


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


def build_inference(config: AppConfig) -> InferencePort:
    mode = (config.inference_mode or "fake").lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"DGB_INFERENCE_MODE 只支持 {SUPPORTED_MODES}，收到: {mode}")

    if mode == "fake":
        return FakeInference(get_scenario(config.inference_scenario))

    if not config.live_token_file:
        raise ValueError("DGB_INFERENCE_MODE=live 时必须配置 DGB_LIVE_TOKEN_FILE")
    if not config.frame_sequence_dir:
        raise ValueError("DGB_INFERENCE_MODE=live 时必须配置 DGB_FRAME_SEQUENCE（JPEG 序列目录）")

    client = LiveClient.from_token_file(config.live_url, config.live_token_file)
    camera = LiveCamera(
        view=config.live_view,
        image_size=_parse_size(config.live_image_size),
        driver_roi=_parse_roi(config.live_roi),
    )
    source = JpegSequenceSource(config.frame_sequence_dir, step_ms=config.live_step_ms)
    logger.info(
        "live 推理已启用: url=%s source=%s size=%s view=%s step_ms=%s",
        config.live_url, config.frame_sequence_dir, config.live_image_size,
        config.live_view, config.live_step_ms,
    )
    return LiveInference(client, camera, source)


__all__ = ["build_inference", "SUPPORTED_MODES"]
