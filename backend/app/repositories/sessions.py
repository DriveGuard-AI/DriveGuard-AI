"""监测会话仓储。"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import MonitoringSession


class SessionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_active(self) -> MonitoringSession | None:
        stmt = (
            select(MonitoringSession)
            .where(MonitoringSession.status == "active")
            .order_by(MonitoringSession.started_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def get(self, session_id: str) -> MonitoringSession | None:
        return self._session.get(MonitoringSession, session_id)

    def create(
        self,
        *,
        session_id: str,
        started_at: str,
        camera_id: str,
        device_id: str,
        model_bundle_version: str,
    ) -> MonitoringSession:
        record = MonitoringSession(
            session_id=session_id,
            started_at=started_at,
            ended_at=None,
            camera_id=camera_id,
            device_id=device_id,
            status="active",
            model_bundle_version=model_bundle_version,
            created_at=started_at,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def end(self, session_id: str, ended_at: str, *, status: str = "ended") -> None:
        record = self._session.get(MonitoringSession, session_id)
        if record is None:
            return
        record.ended_at = ended_at
        record.status = status
        self._session.flush()
