"""ORM 模型，字段与约束对齐 `docs/database/DATABASE_DESIGN.md` 第 4、5 节。"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

RISK_LEVELS = ("SAFE", "WARNING", "DANGER")
EVENT_TYPES = ("fatigue", "phone_use", "drinking", "distraction", "eyes_off_road", "other")
EVENT_STATUSES = ("active", "ended", "discarded")
QUALITIES = ("good", "degraded", "unknown")
SESSION_STATUSES = ("active", "ended", "failed")
MODEL_TYPES = ("detector", "landmarker", "temporal", "fusion")
MODEL_RUNTIMES = ("pytorch", "onnxruntime", "tensorrt")
MODEL_STATUSES = ("registered", "active", "retired", "failed")
VALUE_TYPES = ("string", "integer", "number", "boolean", "object", "array")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    joined = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({joined})"


class MonitoringSession(Base):
    __tablename__ = "monitoring_sessions"

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    started_at: Mapped[str] = mapped_column(Text, nullable=False)
    ended_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    camera_id: Mapped[str] = mapped_column(Text, nullable=False)
    device_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    model_bundle_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)

    events: Mapped[list["DrivingEvent"]] = relationship(back_populates="session")

    __table_args__ = (
        CheckConstraint(_in_list("status", SESSION_STATUSES), name="ck_sessions_status"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_sessions_time"),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_version_id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_type: Mapped[str] = mapped_column(Text, nullable=False)
    semantic_version: Mapped[str] = mapped_column(Text, nullable=False)
    runtime: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    feature_schema_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    activated_at: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("model_type", MODEL_TYPES), name="ck_models_type"),
        CheckConstraint(_in_list("runtime", MODEL_RUNTIMES), name="ck_models_runtime"),
        CheckConstraint(_in_list("status", MODEL_STATUSES), name="ck_models_status"),
        CheckConstraint("length(sha256) = 64", name="ck_models_sha256"),
        UniqueConstraint("model_name", "semantic_version", "runtime", name="uq_models_identity"),
    )


class DrivingEvent(Base):
    __tablename__ = "driving_events"

    event_id: Mapped[str] = mapped_column(Text, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("monitoring_sessions.session_id"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[str] = mapped_column(Text, nullable=False)
    ended_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    max_risk_level: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    model_version_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("model_versions.model_version_id"), nullable=True
    )
    risk_scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    quality: Mapped[str] = mapped_column(Text, nullable=False)
    clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped[MonitoringSession] = relationship(back_populates="events")

    __table_args__ = (
        CheckConstraint(_in_list("event_type", EVENT_TYPES), name="ck_events_type"),
        CheckConstraint(_in_list("max_risk_level", RISK_LEVELS), name="ck_events_risk"),
        CheckConstraint(_in_list("status", EVENT_STATUSES), name="ck_events_status"),
        CheckConstraint(_in_list("quality", QUALITIES), name="ck_events_quality"),
        CheckConstraint("duration_ms >= 0", name="ck_events_duration"),
        CheckConstraint("max_confidence BETWEEN 0.0 AND 1.0", name="ck_events_confidence"),
        CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_events_time"),
        CheckConstraint(
            "(status = 'active' AND ended_at IS NULL) OR status <> 'active'",
            name="ck_events_active_boundary",
        ),
    )


class SystemSetting(Base):
    __tablename__ = "system_settings"

    setting_key: Mapped[str] = mapped_column(Text, primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    requires_restart: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_secret: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(_in_list("value_type", VALUE_TYPES), name="ck_settings_value_type"),
        CheckConstraint("version >= 1", name="ck_settings_version"),
        CheckConstraint("requires_restart IN (0, 1)", name="ck_settings_restart"),
        CheckConstraint("is_secret IN (0, 1)", name="ck_settings_secret"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id: Mapped[str] = mapped_column(Text, primary_key=True)
    occurred_at: Mapped[str] = mapped_column(Text, nullable=False)
    actor_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    resource_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str] = mapped_column(Text, nullable=False)
    request_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")

    __table_args__ = (
        CheckConstraint(_in_list("result", ("success", "failure")), name="ck_audit_result"),
    )


__all__ = [
    "AuditLog",
    "DrivingEvent",
    "ModelVersion",
    "MonitoringSession",
    "SystemSetting",
    "EVENT_STATUSES",
    "EVENT_TYPES",
    "MODEL_RUNTIMES",
    "MODEL_STATUSES",
    "MODEL_TYPES",
    "QUALITIES",
    "RISK_LEVELS",
    "SESSION_STATUSES",
    "VALUE_TYPES",
]
