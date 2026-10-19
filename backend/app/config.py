"""应用配置。

全部配置来自环境变量，便于本地开发、测试与边缘部署使用不同取值。
参见 `docs/requirements/REQUIREMENTS.md` NFR-004 与 `docs/api/API_DESIGN.md` 第 2 节。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = BACKEND_DIR.parent

API_MAJOR_VERSION = "v1"
SCHEMA_VERSION = "1.0"
WS_SCHEMA_VERSION = "1.0"

DEFAULT_SERVICE_VERSION = "0.1.0"


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    """不可变运行时配置。"""

    service_version: str
    database_url: str
    admin_token: str | None
    inference_scenario: str
    inference_hz: float
    ws_max_hz: float
    heartbeat_seconds: float
    auto_create_schema: bool
    static_dir: Path
    media_dir: Path
    inference_mode: str = "fake"
    live_url: str = "http://127.0.0.1:8765"
    live_token_file: str | None = None
    frame_sequence_dir: str | None = None
    live_view: str = "body"
    live_image_size: str = "1280x720"
    live_roi: str = "0.15,0.0,0.75,1.0"
    live_step_ms: int = 40
    # 实时帧来源：frames=预抽好的 JPEG 序列（离线素材/联调），camera=本地摄像头。
    live_source: str = "frames"
    camera_device_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_jpeg_quality: int = 85
    camera_backend: str = ""  # 空 = 按平台默认（Windows 用 dshow）
    # 实时预览（API-010）的部署上限：不承诺等于推理 FPS，也不允许无限客户端。
    preview_max_fps: float = 10.0
    preview_max_clients: int = 2
    # 离线分析产物的约束（NFR-003：上传必须配套保留期、磁盘上限与访问控制）。
    analysis_max_frames: int = 15000  # 约 10 分钟 @40ms，用于挡住长视频
    analysis_max_disk_mb: int = 2048
    analysis_max_upload_mb: int = 512
    analysis_retain_source: bool = False
    analysis_retain_frames_hours: int = 24

    @property
    def auth_enabled(self) -> bool:
        """是否配置了管理 Token。未配置时为本地只读演示模式。"""
        return bool(self.admin_token)

    @property
    def snapshot_interval_seconds(self) -> float:
        hz = self.inference_hz if self.inference_hz > 0 else 5.0
        return 1.0 / hz


def load_config() -> AppConfig:
    """从环境变量加载配置。"""

    default_db = BACKEND_DIR / "var" / "driveguard.db"
    database_url = _env_str("DGB_DATABASE_URL", f"sqlite:///{default_db.as_posix()}")
    static_dir = Path(_env_str("DGB_STATIC_DIR", (REPO_DIR / "frontend" / "dist").as_posix()))
    media_dir = Path(_env_str("DGB_MEDIA_DIR", (BACKEND_DIR / "var" / "media").as_posix()))

    return AppConfig(
        service_version=_env_str("DGB_SERVICE_VERSION", DEFAULT_SERVICE_VERSION) or DEFAULT_SERVICE_VERSION,
        database_url=database_url or f"sqlite:///{default_db.as_posix()}",
        admin_token=_env_str("DGB_ADMIN_TOKEN"),
        inference_scenario=_env_str("DGB_INFERENCE_SCENARIO", "default") or "default",
        inference_hz=max(0.1, _env_float("DGB_INFERENCE_HZ", 5.0)),
        ws_max_hz=max(0.1, _env_float("DGB_WS_MAX_HZ", 10.0)),
        heartbeat_seconds=max(1.0, _env_float("DGB_HEARTBEAT_SECONDS", 15.0)),
        auto_create_schema=_env_bool("DGB_AUTO_CREATE_SCHEMA", True),
        static_dir=static_dir,
        media_dir=media_dir,
        inference_mode=(_env_str("DGB_INFERENCE_MODE", "fake") or "fake").lower(),
        live_url=_env_str("DGB_LIVE_URL", "http://127.0.0.1:8765") or "http://127.0.0.1:8765",
        live_token_file=_env_str("DGB_LIVE_TOKEN_FILE"),
        frame_sequence_dir=_env_str("DGB_FRAME_SEQUENCE"),
        live_view=(_env_str("DGB_LIVE_VIEW", "body") or "body").lower(),
        live_image_size=_env_str("DGB_LIVE_IMAGE_SIZE", "1280x720") or "1280x720",
        live_roi=_env_str("DGB_LIVE_ROI", "0.15,0.0,0.75,1.0") or "0.15,0.0,0.75,1.0",
        live_step_ms=max(1, int(_env_float("DGB_LIVE_STEP_MS", 40.0))),
        live_source=(_env_str("DGB_LIVE_SOURCE", "frames") or "frames").lower(),
        camera_device_index=max(0, int(_env_float("DGB_CAMERA_DEVICE_INDEX", 0.0))),
        camera_width=max(16, int(_env_float("DGB_CAMERA_WIDTH", 1280.0))),
        camera_height=max(16, int(_env_float("DGB_CAMERA_HEIGHT", 720.0))),
        camera_jpeg_quality=min(100, max(1, int(_env_float("DGB_CAMERA_JPEG_QUALITY", 85.0)))),
        camera_backend=(_env_str("DGB_CAMERA_BACKEND", "") or "").lower(),
        preview_max_fps=min(30.0, max(0.5, _env_float("DGB_PREVIEW_MAX_FPS", 10.0))),
        preview_max_clients=max(1, int(_env_float("DGB_PREVIEW_MAX_CLIENTS", 2.0))),
        analysis_max_frames=max(1, int(_env_float("DGB_ANALYSIS_MAX_FRAMES", 15000.0))),
        analysis_max_disk_mb=max(1, int(_env_float("DGB_ANALYSIS_MAX_DISK_MB", 2048.0))),
        analysis_max_upload_mb=max(1, int(_env_float("DGB_ANALYSIS_MAX_UPLOAD_MB", 512.0))),
        analysis_retain_source=_env_bool("DGB_ANALYSIS_RETAIN_SOURCE", False),
        analysis_retain_frames_hours=max(1, int(_env_float("DGB_ANALYSIS_RETAIN_HOURS", 24.0))),
    )
