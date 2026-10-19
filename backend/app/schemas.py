"""对外数据契约（Pydantic v2）。

字段与 `docs/api/API_DESIGN.md` 第 3、4、5 节保持一致：
- 时间统一为 UTC，序列化为 `...Z`
- 枚举固定大写风险等级 `SAFE/WARNING/DANGER`
- REST 字段使用 `snake_case`
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

RiskLevel = Literal["SAFE", "WARNING", "DANGER"]
EventStatus = Literal["active", "ended", "discarded"]
Quality = Literal["good", "degraded", "unknown"]
EventType = Literal["fatigue", "phone_use", "drinking", "distraction", "eyes_off_road", "other"]
InferenceMode = Literal["fake", "real"]


def _iso_z(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    return f"{value.strftime('%Y-%m-%dT%H:%M:%S')}.{value.microsecond // 1000:03d}Z"


UtcDateTime = Annotated[datetime, PlainSerializer(_iso_z, return_type=str)]


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service_version: str
    time: UtcDateTime


class HeadPose(BaseModel):
    yaw: float
    pitch: float
    roll: float


class FeatureSet(BaseModel):
    ear: float | None = None
    mar: float | None = None
    perclos: float | None = None
    head_pose: HeadPose | None = None


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float] = Field(default_factory=list)


class RiskSnapshot(BaseModel):
    """风险快照：风险融合层输出的版本化实时状态（见 `.ai-memory/project_memory.md`）。

    `inference_mode` 为契约新增的可选字段，用于显式声明当前离线实现为脚本化回放，
    避免把占位数据误读为真实模型结论。
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: str = "1.0"
    session_id: str
    sequence: int
    captured_at: UtcDateTime
    risk_level: RiskLevel
    primary_risk: str | None = None
    confidence: float
    quality: Quality
    risk_scores: dict[str, float] = Field(default_factory=dict)
    features: FeatureSet = Field(default_factory=FeatureSet)
    detections: list[Detection] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    inference_mode: InferenceMode = "fake"
    # 视频时间轴位置（毫秒）。离线分析时由 live 服务的 window_end_ms 提供，
    # 用于把事件锚定到素材时间轴；实时模式与无证据时为 None。
    source_timestamp_ms: int | None = None


class EventOut(BaseModel):
    event_id: str
    session_id: str
    event_type: EventType
    started_at: UtcDateTime
    ended_at: UtcDateTime | None = None
    duration_ms: int
    max_confidence: float
    max_risk_level: RiskLevel
    status: EventStatus
    quality: Quality
    risk_scores: dict[str, float] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    model_version_id: str | None = None
    clip_available: bool = False


class EventListResponse(BaseModel):
    items: list[EventOut]
    page: int
    page_size: int
    total: int
    has_next: bool


class AnalysisJobCreate(BaseModel):
    """创建一次离线分析。

    只接受 `frames`（已抽好的 JPEG 目录，位于媒体根目录之下）。`camera`/`push`
    为实时采集预留，当前返回 400。
    """

    source_kind: str = "frames"
    source_path: str


class AnalysisJobOut(BaseModel):
    job_id: str
    session_id: str
    source_kind: str
    source_path: str
    source_sha256: str | None = None
    status: str
    frames_total: int
    frames_done: int
    step_ms: int = 40
    timeline_ms: int
    error: str | None = None
    # 原始视频是否还在：在则前端用 <video> 回放，否则逐帧取图。
    source_available: bool = False
    created_at: UtcDateTime
    started_at: UtcDateTime | None = None
    ended_at: UtcDateTime | None = None
    updated_at: UtcDateTime


class AnalysisJobListResponse(BaseModel):
    items: list[AnalysisJobOut]
    page: int
    page_size: int
    total: int
    has_next: bool


class RiskPoint(BaseModel):
    """回放用的风险采样点。服务端按需降采样，前端按 `ms` 画曲线。"""

    ms: int
    frame: int
    risk_level: RiskLevel
    primary_risk: str | None = None
    confidence: float = 0.0


class EvidenceEvent(BaseModel):
    """事件在素材时间轴/帧号上的位置，供回放定位。"""

    event_id: str
    event_type: str
    risk_level: str
    started_ms: int
    ended_ms: int
    start_frame: int
    end_frame: int
    duration_ms: int


class AnalysisEvidence(BaseModel):
    job_id: str
    session_id: str
    step_ms: int
    frames_total: int
    duration_ms: int
    source_available: bool
    source_kind: str
    timeline: list[RiskPoint] = Field(default_factory=list)
    events: list[EvidenceEvent] = Field(default_factory=list)


class CountByType(BaseModel):
    event_type: str
    count: int
    total_duration_ms: int


class CountByRisk(BaseModel):
    risk_level: str
    count: int


class TrendBucket(BaseModel):
    bucket_start: UtcDateTime
    count: int


class StatisticsSummary(BaseModel):
    started_from: UtcDateTime
    started_to: UtcDateTime
    # 回显过滤条件：前端据此确认“空结果”确实是这个会话的，而不是请求漏带了参数。
    session_id: str | None = None
    total_events: int
    by_type: list[CountByType] = Field(default_factory=list)
    by_risk_level: list[CountByRisk] = Field(default_factory=list)
    trend: list[TrendBucket] = Field(default_factory=list)


class SettingItem(BaseModel):
    key: str
    value_type: str
    value: Any | None = None
    is_secret: bool = False
    is_configured: bool = False
    requires_restart: bool = False
    updated_at: UtcDateTime | None = None
    applied: bool = True


class SettingsResponse(BaseModel):
    version: int
    items: list[SettingItem]


class SettingsUpdateRequest(BaseModel):
    version: int = Field(ge=1)
    changes: dict[str, Any]


class SettingsUpdateResponse(BaseModel):
    version: int
    applied: dict[str, Any] = Field(default_factory=dict)
    restart_required: list[str] = Field(default_factory=list)
    rejected: dict[str, str] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    model_version_id: str
    model_name: str
    model_type: str
    semantic_version: str
    runtime: str
    feature_schema_version: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    status: str
    activated_at: UtcDateTime | None = None


class CameraStatus(BaseModel):
    available: bool
    device_id: str
    source: str


class PipelineStatus(BaseModel):
    state: str
    processed_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    dropped_frame_rate: float
    inference_mode: InferenceMode
    degraded: bool
    # 当前推理源是离线素材（由分析任务切换）而非实时采集。
    # 页面据此提示“当前为离线回放”，不把回放当实时。
    playback: bool = False


class DatabaseStatus(BaseModel):
    writable: bool
    driver: str


class WebSocketStatus(BaseModel):
    clients: int
    max_hz: float


class ModelStatus(BaseModel):
    loaded: bool
    bundle_version: str
    active_models: list[str] = Field(default_factory=list)


class SystemStatus(BaseModel):
    service_version: str
    camera: CameraStatus
    pipeline: PipelineStatus
    database: DatabaseStatus
    websocket: WebSocketStatus
    model: ModelStatus


class WSMessage(BaseModel):
    type: str
    schema_version: str = "1.0"
    sequence: int
    sent_at: UtcDateTime
    data: Any = None


__all__ = [
    "CountByRisk",
    "CountByType",
    "Detection",
    "ErrorResponse",
    "EventListResponse",
    "EventOut",
    "EventStatus",
    "EventType",
    "FeatureSet",
    "HeadPose",
    "HealthResponse",
    "InferenceMode",
    "ModelInfo",
    "Quality",
    "RiskLevel",
    "RiskSnapshot",
    "SettingItem",
    "SettingsResponse",
    "SettingsUpdateRequest",
    "SettingsUpdateResponse",
    "StatisticsSummary",
    "SystemStatus",
    "TrendBucket",
    "UtcDateTime",
    "WSMessage",
]
