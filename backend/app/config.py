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
    )
