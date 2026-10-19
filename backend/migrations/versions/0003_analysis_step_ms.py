"""analysis job step_ms

Revision ID: 0003_analysis_step_ms
Revises: 0002_analysis_jobs
Create Date: 2026-10-19

回放定位要把「事件时间轴」换算成「帧号」，所以任务表必须记住抽帧步长。
存量任务按 40 ms（25 Hz，默认值）回填。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_analysis_step_ms"
down_revision = "0002_analysis_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column("step_ms", sa.Integer(), nullable=False, server_default="40"),
    )


def downgrade() -> None:
    with op.batch_alter_table("analysis_jobs") as batch:
        batch.drop_column("step_ms")
