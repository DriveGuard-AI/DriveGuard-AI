"""危险事件仓储：创建、聚合更新、结束与查询。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..models import DrivingEvent
from ..schemas import RiskSnapshot
from ..timeutil import duration_ms, parse_iso, to_iso_z

MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class EventFilters:
    event_type: str | None = None
    risk_level: str | None = None
    status: str | None = None
    started_from: datetime | None = None
    started_to: datetime | None = None
    session_id: str | None = None


class EventRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    # ---------------------------------------------------------------- 写入

    def get(self, event_id: str) -> DrivingEvent | None:
        return self._session.get(DrivingEvent, event_id)

    def get_active(self, session_id: str) -> DrivingEvent | None:
        stmt = (
            select(DrivingEvent)
            .where(DrivingEvent.session_id == session_id, DrivingEvent.status == "active")
            .order_by(DrivingEvent.started_at.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def create(
        self,
        *,
        event_id: str,
        session_id: str,
        event_type: str,
        snapshot: RiskSnapshot,
        model_version_id: str | None,
        now: datetime,
    ) -> DrivingEvent:
        timestamp = to_iso_z(now)
        record = DrivingEvent(
            event_id=event_id,
            session_id=session_id,
            event_type=event_type,
            started_at=to_iso_z(snapshot.captured_at),
            ended_at=None,
            duration_ms=duration_ms(to_iso_z(snapshot.captured_at), timestamp),
            max_confidence=snapshot.confidence,
            max_risk_level=snapshot.risk_level,
            status="active",
            model_version_id=model_version_id,
            risk_scores_json=json.dumps(snapshot.risk_scores, sort_keys=True),
            reason_codes_json=json.dumps(snapshot.reason_codes),
            quality=snapshot.quality,
            clip_path=None,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._session.add(record)
        self._session.flush()
        return record

    def accumulate(self, event: DrivingEvent, snapshot: RiskSnapshot, *, now: datetime) -> DrivingEvent:
        """聚合同类连续风险：更新时长、最高置信度与最高风险等级。"""

        timestamp = to_iso_z(now)
        event.duration_ms = duration_ms(event.started_at, timestamp)
        event.max_confidence = max(event.max_confidence, snapshot.confidence)
        event.max_risk_level = _max_risk_level(event.max_risk_level, snapshot.risk_level)
        event.risk_scores_json = json.dumps(snapshot.risk_scores, sort_keys=True)
        event.reason_codes_json = json.dumps(snapshot.reason_codes)
        event.quality = snapshot.quality
        event.updated_at = timestamp
        self._session.flush()
        return event

    def close(
        self,
        event: DrivingEvent,
        *,
        ended_at: datetime,
        min_duration_ms: int,
    ) -> DrivingEvent:
        """结束事件；未达到最短持续时间则标记为 discarded。"""

        timestamp = to_iso_z(ended_at)
        event.ended_at = timestamp
        event.duration_ms = duration_ms(event.started_at, timestamp)
        event.status = "ended" if event.duration_ms >= min_duration_ms else "discarded"
        event.updated_at = timestamp
        self._session.flush()
        return event

    # ---------------------------------------------------------------- 查询

    def list(self, filters: EventFilters, *, page: int, page_size: int) -> tuple[list[DrivingEvent], int]:
        conditions = self._conditions(filters)
        base = select(DrivingEvent)
        if conditions:
            base = base.where(*conditions)

        total = self._session.execute(
            select(func.count()).select_from(base.subquery())
        ).scalar_one()

        # 稳定排序：started_at DESC, event_id DESC
        stmt = (
            base.order_by(DrivingEvent.started_at.desc(), DrivingEvent.event_id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        items = list(self._session.execute(stmt).scalars())
        return items, int(total)

    def summary(self, *, started_from: datetime, started_to: datetime) -> dict[str, Any]:
        """按时间范围聚合：总数、按类型、按风险等级、按天趋势。"""

        window = (
            DrivingEvent.started_at >= to_iso_z(started_from),
            DrivingEvent.started_at <= to_iso_z(started_to),
        )

        total = self._session.execute(
            select(func.count()).select_from(DrivingEvent).where(*window)
        ).scalar_one()

        by_type_rows = self._session.execute(
            select(
                DrivingEvent.event_type,
                func.count(),
                func.coalesce(func.sum(DrivingEvent.duration_ms), 0),
            )
            .where(*window)
            .group_by(DrivingEvent.event_type)
            .order_by(DrivingEvent.event_type)
        ).all()

        by_risk_rows = self._session.execute(
            select(DrivingEvent.max_risk_level, func.count())
            .where(*window)
            .group_by(DrivingEvent.max_risk_level)
            .order_by(DrivingEvent.max_risk_level)
        ).all()

        trend_rows = self._session.execute(
            select(
                func.substr(DrivingEvent.started_at, 1, 10).label("day"),
                func.count(),
            )
            .where(*window)
            .group_by("day")
            .order_by("day")
        ).all()

        return {
            "total_events": int(total),
            "by_type": [
                {"event_type": row[0], "count": int(row[1]), "total_duration_ms": int(row[2])}
                for row in by_type_rows
            ],
            "by_risk_level": [{"risk_level": row[0], "count": int(row[1])} for row in by_risk_rows],
            "trend": [
                {"bucket_start": datetime.fromisoformat(f"{row[0]}T00:00:00+00:00"), "count": int(row[1])}
                for row in trend_rows
            ],
        }

    def count_events(self) -> int:
        return int(self._session.execute(select(func.count()).select_from(DrivingEvent)).scalar_one())

    def latest_updated_at(self) -> datetime | None:
        value = self._session.execute(select(func.max(DrivingEvent.updated_at))).scalar_one()
        return parse_iso(value) if value else None

    # ---------------------------------------------------------------- 内部

    def _conditions(self, filters: EventFilters) -> list[Any]:
        conditions: list[Any] = []
        if filters.event_type:
            conditions.append(DrivingEvent.event_type == filters.event_type)
        if filters.risk_level:
            conditions.append(DrivingEvent.max_risk_level == filters.risk_level)
        if filters.status:
            conditions.append(DrivingEvent.status == filters.status)
        if filters.started_from:
            conditions.append(DrivingEvent.started_at >= to_iso_z(filters.started_from))
        if filters.started_to:
            conditions.append(DrivingEvent.started_at <= to_iso_z(filters.started_to))
        if filters.session_id:
            conditions.append(DrivingEvent.session_id == filters.session_id)
        return conditions


def _max_risk_level(current: str, candidate: str) -> str:
    order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


def iso_minus_ms(value: str, milliseconds: int) -> datetime:
    return parse_iso(value) - timedelta(milliseconds=milliseconds)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["EventFilters", "EventRepository", "MAX_PAGE_SIZE", "ensure_utc", "iso_minus_ms"]
