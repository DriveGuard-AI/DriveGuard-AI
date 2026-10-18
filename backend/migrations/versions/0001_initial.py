"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-11

与 `docs/database/DATABASE_DESIGN.md` 第 5 节的 DDL 基线一致：
5 张表、CHECK 约束、外键与索引。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "monitoring_sessions",
        sa.Column("session_id", sa.Text(), primary_key=True),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("camera_id", sa.Text(), nullable=False),
        sa.Column("device_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("model_bundle_version", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "status IN ('active', 'ended', 'failed')", name="ck_sessions_status"
        ),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_sessions_time"),
    )

    op.create_table(
        "model_versions",
        sa.Column("model_version_id", sa.Text(), primary_key=True),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("model_type", sa.Text(), nullable=False),
        sa.Column("semantic_version", sa.Text(), nullable=False),
        sa.Column("runtime", sa.Text(), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("feature_schema_version", sa.Text(), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("activated_at", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "model_type IN ('detector', 'landmarker', 'temporal', 'fusion')",
            name="ck_models_type",
        ),
        sa.CheckConstraint(
            "runtime IN ('pytorch', 'onnxruntime', 'tensorrt')", name="ck_models_runtime"
        ),
        sa.CheckConstraint(
            "status IN ('registered', 'active', 'retired', 'failed')", name="ck_models_status"
        ),
        sa.CheckConstraint("length(sha256) = 64", name="ck_models_sha256"),
        sa.UniqueConstraint(
            "model_name", "semantic_version", "runtime", name="uq_models_identity"
        ),
    )

    op.create_table(
        "driving_events",
        sa.Column("event_id", sa.Text(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("monitoring_sessions.session_id"),
            nullable=False,
        ),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=False),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_confidence", sa.Float(), nullable=False),
        sa.Column("max_risk_level", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "model_version_id",
            sa.Text(),
            sa.ForeignKey("model_versions.model_version_id"),
            nullable=True,
        ),
        sa.Column("risk_scores_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("reason_codes_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("quality", sa.Text(), nullable=False),
        sa.Column("clip_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('fatigue', 'phone_use', 'drinking', 'distraction', "
            "'eyes_off_road', 'other')",
            name="ck_events_type",
        ),
        sa.CheckConstraint(
            "max_risk_level IN ('SAFE', 'WARNING', 'DANGER')", name="ck_events_risk"
        ),
        sa.CheckConstraint(
            "status IN ('active', 'ended', 'discarded')", name="ck_events_status"
        ),
        sa.CheckConstraint(
            "quality IN ('good', 'degraded', 'unknown')", name="ck_events_quality"
        ),
        sa.CheckConstraint("duration_ms >= 0", name="ck_events_duration"),
        sa.CheckConstraint(
            "max_confidence BETWEEN 0.0 AND 1.0", name="ck_events_confidence"
        ),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="ck_events_time"),
        sa.CheckConstraint(
            "(status = 'active' AND ended_at IS NULL) OR status <> 'active'",
            name="ck_events_active_boundary",
        ),
    )

    op.create_table(
        "system_settings",
        sa.Column("setting_key", sa.Text(), primary_key=True),
        sa.Column("value_json", sa.Text(), nullable=False),
        sa.Column("value_type", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("requires_restart", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_secret", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.Column("updated_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "value_type IN ('string', 'integer', 'number', 'boolean', 'object', 'array')",
            name="ck_settings_value_type",
        ),
        sa.CheckConstraint("version >= 1", name="ck_settings_version"),
        sa.CheckConstraint("requires_restart IN (0, 1)", name="ck_settings_restart"),
        sa.CheckConstraint("is_secret IN (0, 1)", name="ck_settings_secret"),
    )

    op.create_table(
        "audit_logs",
        sa.Column("audit_id", sa.Text(), primary_key=True),
        sa.Column("occurred_at", sa.Text(), nullable=False),
        sa.Column("actor_id", sa.Text(), nullable=True),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column("resource_id", sa.Text(), nullable=True),
        sa.Column("result", sa.Text(), nullable=False),
        sa.Column("request_id", sa.Text(), nullable=True),
        sa.Column("details_json", sa.Text(), nullable=False, server_default="{}"),
        sa.CheckConstraint("result IN ('success', 'failure')", name="ck_audit_result"),
    )

    op.create_index("idx_sessions_started_at", "monitoring_sessions", ["started_at"])
    op.create_index("idx_events_started_at", "driving_events", ["started_at"])
    op.create_index("idx_events_type_started", "driving_events", ["event_type", "started_at"])
    op.create_index("idx_events_risk_started", "driving_events", ["max_risk_level", "started_at"])
    op.create_index("idx_events_session_started", "driving_events", ["session_id", "started_at"])
    op.create_index("idx_events_status", "driving_events", ["status"])
    op.create_index("idx_models_name_status", "model_versions", ["model_name", "status"])
    op.create_index("idx_audit_occurred_at", "audit_logs", ["occurred_at"])


def downgrade() -> None:
    op.drop_index("idx_audit_occurred_at", table_name="audit_logs")
    op.drop_index("idx_models_name_status", table_name="model_versions")
    op.drop_index("idx_events_status", table_name="driving_events")
    op.drop_index("idx_events_session_started", table_name="driving_events")
    op.drop_index("idx_events_risk_started", table_name="driving_events")
    op.drop_index("idx_events_type_started", table_name="driving_events")
    op.drop_index("idx_events_started_at", table_name="driving_events")
    op.drop_index("idx_sessions_started_at", table_name="monitoring_sessions")
    op.drop_table("audit_logs")
    op.drop_table("system_settings")
    op.drop_table("driving_events")
    op.drop_table("model_versions")
    op.drop_table("monitoring_sessions")
