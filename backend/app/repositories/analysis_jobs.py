"""分析任务仓储（`analysis_jobs`）。"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select

from ..models import AnalysisJob
from ..timeutil import to_iso_z


class AnalysisJobRepository:
    def __init__(self, session) -> None:
        self._session = session

    def create(
        self,
        *,
        job_id: str,
        session_id: str,
        source_kind: str,
        source_path: str,
        frames_total: int,
        source_sha256: str | None = None,
        now: datetime,
    ) -> AnalysisJob:
        timestamp = to_iso_z(now)
        record = AnalysisJob(
            job_id=job_id,
            session_id=session_id,
            source_kind=source_kind,
            source_path=source_path,
            source_sha256=source_sha256,
            status="running",
            frames_total=frames_total,
            frames_done=0,
            timeline_ms=0,
            created_at=timestamp,
            started_at=timestamp,
            updated_at=timestamp,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def get(self, job_id: str) -> AnalysisJob | None:
        return self._session.get(AnalysisJob, job_id)

    def list(
        self, *, status: str | None = None, page: int = 1, page_size: int = 20
    ) -> tuple[list[AnalysisJob], int]:
        base = select(AnalysisJob)
        counter = select(func.count()).select_from(AnalysisJob)
        if status:
            base = base.where(AnalysisJob.status == status)
            counter = counter.where(AnalysisJob.status == status)
        total = self._session.execute(counter).scalar_one()
        rows = (
            self._session.execute(
                base.order_by(AnalysisJob.created_at.desc(), AnalysisJob.job_id.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            .scalars()
            .all()
        )
        return list(rows), int(total)

    def progress(
        self, job: AnalysisJob, *, frames_done: int, timeline_ms: int, now: datetime
    ) -> AnalysisJob:
        job.frames_done = max(0, min(frames_done, job.frames_total))
        job.timeline_ms = max(job.timeline_ms, timeline_ms)
        job.updated_at = to_iso_z(now)
        self._session.flush()
        return job

    def finish(
        self,
        job: AnalysisJob,
        *,
        status: str,
        now: datetime,
        error: str | None = None,
        frames_done: int | None = None,
        timeline_ms: int | None = None,
    ) -> AnalysisJob:
        job.status = status
        job.error = (error or None) and str(error)[:400]
        if frames_done is not None:
            job.frames_done = max(0, min(frames_done, job.frames_total))
        if timeline_ms is not None:
            job.timeline_ms = max(job.timeline_ms, timeline_ms)
        timestamp = to_iso_z(now)
        job.ended_at = timestamp
        job.updated_at = timestamp
        self._session.flush()
        return job
