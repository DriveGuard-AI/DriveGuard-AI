"""analysis jobs

Revision ID: 0002_analysis_jobs
Revises: 0001_initial
Create Date: 2026-10-19

离线分析任务表。只存指针与进度，素材与抽帧产物落在媒体目录
（`media/analysis/<job_id>/`），不进数据库。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_analysis_jobs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "analysis_jobs",
        sa.Column("job_id", sa.Text(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Text(),
            sa.ForeignKey("monitoring_sessions.session_id"),
            nullable=False,
        ),
        sa.Column("source_kind", sa.Text(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("source_sha256", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("frames_total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("frames_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("timeline_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.Column("started_at", sa.Text(), nullable=True),
        sa.Column("ended_at", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(
            "source_kind IN ('frames', 'camera', 'push')", name="ck_jobs_kind"
        ),
        sa.CheckConstraint("frames_total >= 0", name="ck_jobs_total"),
        sa.CheckConstraint(
            "frames_done >= 0 AND frames_done <= frames_total", name="ck_jobs_done"
        ),
        sa.CheckConstraint("timeline_ms >= 0", name="ck_jobs_timeline"),
        sa.CheckConstraint(
            "(status IN ('running','completed','failed') AND started_at IS NOT NULL)"
            " OR status IN ('queued','cancelled')",
            name="ck_jobs_started",
        ),
        sa.CheckConstraint(
            "(status IN ('completed','failed','cancelled') AND ended_at IS NOT NULL)"
            " OR status IN ('queued','running')",
            name="ck_jobs_ended",
        ),
        sa.CheckConstraint(
            "ended_at IS NULL OR started_at IS NULL OR ended_at >= started_at",
            name="ck_jobs_time",
        ),
    )
    op.create_index("ix_jobs_created", "analysis_jobs", ["created_at"])
    op.create_index("ix_jobs_status", "analysis_jobs", ["status"])
    op.create_index("ix_jobs_session", "analysis_jobs", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_session", table_name="analysis_jobs")
    op.drop_index("ix_jobs_status", table_name="analysis_jobs")
    op.drop_index("ix_jobs_created", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
