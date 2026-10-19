"""配置项注册表：键名、类型、范围与默认值。

`docs/requirements/REQUIREMENTS.md` FR-009 要求校验类型与范围，且无效配置不得
覆盖当前有效值；`docs/api/API_DESIGN.md` API-007 要求乐观锁与 `422` 明细。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

GLOBAL_REVISION_KEY = "system.settings_revision"


@dataclass(frozen=True)
class SettingSpec:
    key: str
    value_type: str
    default: Any
    requires_restart: bool = False
    is_secret: bool = False
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, value: Any) -> str | None:
        """返回错误原因；`None` 表示合法。"""

        if self.is_secret:
            if not isinstance(value, str) or not value.strip():
                return "must be a non-empty string"
            return None

        if self.value_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return "must be an integer"
        elif self.value_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return "must be a number"
        elif self.value_type == "boolean":
            if not isinstance(value, bool):
                return "must be a boolean"
        elif self.value_type == "string":
            if not isinstance(value, str):
                return "must be a string"
        elif self.value_type == "object":
            if not isinstance(value, dict):
                return "must be an object"
        elif self.value_type == "array":
            if not isinstance(value, list):
                return "must be an array"

        if self.minimum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            if value < self.minimum:
                return f"must be >= {self.minimum}"
        if self.maximum is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            if value > self.maximum:
                return f"must be <= {self.maximum}"
        return None


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec("camera.device_index", "integer", 0, requires_restart=True, minimum=0, maximum=15,
                description="本地摄像头序号"),
    SettingSpec("camera.frame_width", "integer", 1280, requires_restart=True, minimum=320, maximum=4096),
    SettingSpec("camera.frame_height", "integer", 720, requires_restart=True, minimum=240, maximum=2160),
    SettingSpec("preview.enabled", "boolean", False,
                description="实时视频预览开关（运行时生效，默认关闭）"),
    SettingSpec("preview.max_fps", "number", 10.0, minimum=1.0, maximum=30.0,
                description="预览通道最大输出帧率，不承诺等于推理 FPS"),
    SettingSpec("preview.max_clients", "integer", 2, minimum=1, maximum=8,
                description="同时允许的预览连接数"),
    SettingSpec("alarm.delay_ms", "integer", 1500, minimum=0, maximum=60000, description="行为出现到告警的延迟"),
    SettingSpec("alarm.cooldown_ms", "integer", 5000, minimum=0, maximum=600000, description="重复告警冷却时间"),
    SettingSpec("event.min_duration_ms", "integer", 1500, minimum=0, maximum=60000,
                description="低于该时长的事件会被丢弃"),
    SettingSpec("event.merge_gap_ms", "integer", 2000, minimum=0, maximum=120000,
                description="同类风险聚合的最大间隔"),
    SettingSpec("risk.warning_threshold", "number", 0.5, minimum=0.0, maximum=1.0),
    SettingSpec("risk.danger_threshold", "number", 0.8, minimum=0.0, maximum=1.0),
    SettingSpec("temporal.window_seconds", "number", 4.0, minimum=1.0, maximum=10.0),
    SettingSpec("storage.event_retention_days", "integer", 90, minimum=1, maximum=3650),
    SettingSpec("storage.clip_enabled", "boolean", False, description="默认关闭事件视频片段保存"),
    SettingSpec("model.bundle_version", "string", "dgb-mvp-stub", requires_restart=True,
                description="当前模型组合版本（占位）"),
    SettingSpec("model.detector_path", "string", "models/stub-detector.onnx", requires_restart=True),
    SettingSpec("model.temporal_path", "string", "models/stub-tcn.onnx", requires_restart=True),
    SettingSpec("security.admin_token", "string", "", requires_restart=False, is_secret=True,
                description="管理 Token；API 只返回是否已配置"),
)

SETTING_SPEC_BY_KEY: dict[str, SettingSpec] = {spec.key: spec for spec in SETTING_SPECS}

# 待确认参数（文档第 10 节"风险与待确认项"）：实现中作为可配置项提供
PENDING_CONFIRMATION_KEYS = frozenset(
    {
        "alarm.delay_ms",
        "alarm.cooldown_ms",
        "risk.warning_threshold",
        "risk.danger_threshold",
        "temporal.window_seconds",
        "storage.event_retention_days",
    }
)


def get_spec(key: str) -> SettingSpec | None:
    return SETTING_SPEC_BY_KEY.get(key)


__all__ = [
    "GLOBAL_REVISION_KEY",
    "PENDING_CONFIRMATION_KEYS",
    "SETTING_SPECS",
    "SETTING_SPEC_BY_KEY",
    "SettingSpec",
    "get_spec",
]
