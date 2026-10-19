"""事件服务：查询侧 + 生命周期跟踪（写入侧）。

对应 `docs/requirements/REQUIREMENTS.md` FR-006 与
`docs/architecture/ARCHITECTURE.md` §4.2 的事件生命周期
`candidate → active → ended / discarded`。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..database import Database
from ..errors import EventNotFoundError
from ..models import DrivingEvent
from ..repositories.audit import AuditRepository
from ..repositories.events import EventFilters, EventRepository
from ..repositories.models_registry import ModelRegistryRepository
from ..repositories.settings import SettingsRepository
from ..schemas import EventListResponse, EventOut, RiskSnapshot
from ..timeutil import parse_iso, to_iso_z, utcnow

VALID_EVENT_TYPES = frozenset(
    {"fatigue", "phone_use", "drinking", "distraction", "eyes_off_road", "other"}
)


def normalize_event_type(value: str | None) -> str:
    """将主风险名称映射到数据库事件类型枚举。"""

    if value and value in VALID_EVENT_TYPES:
        return value
    return "other"


def event_to_schema(record: DrivingEvent) -> EventOut:
    return EventOut(
        event_id=record.event_id,
        session_id=record.session_id,
        event_type=record.event_type,  # type: ignore[arg-type]
        started_at=parse_iso(record.started_at),
        ended_at=parse_iso(record.ended_at) if record.ended_at else None,
        duration_ms=record.duration_ms,
        max_confidence=record.max_confidence,
        max_risk_level=record.max_risk_level,  # type: ignore[arg-type]
        status=record.status,  # type: ignore[arg-type]
        quality=record.quality,  # type: ignore[arg-type]
        risk_scores=_load_json(record.risk_scores_json, {}),
        reason_codes=_load_json(record.reason_codes_json, []),
        model_version_id=record.model_version_id,
        clip_available=bool(record.clip_path),
    )


def _load_json(raw: str, fallback: object) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return fallback


class EventService:
    """事件查询服务（只读）。"""

    def __init__(self, db: Database) -> None:
        self._db = db

    def list_events(self, filters: EventFilters, *, page: int, page_size: int) -> EventListResponse:
        with self._db.session() as session:
            records, total = EventRepository(session).list(filters, page=page, page_size=page_size)
        return EventListResponse(
            items=[event_to_schema(record) for record in records],
            page=page,
            page_size=page_size,
            total=total,
            has_next=page * page_size < total,
        )

    def get_event(self, event_id: str) -> EventOut:
        with self._db.session() as session:
            record = EventRepository(session).get(event_id)
        if record is None:
            raise EventNotFoundError(event_id)
        return event_to_schema(record)


@dataclass(frozen=True)
class EventChange:
    """一次事件状态变化，用于 WebSocket `event.changed` 通知。"""

    action: str  # created | updated | ended | discarded
    event: EventOut

    def to_payload(self) -> dict[str, object]:
        return {"action": self.action, "event": json.loads(self.event.model_dump_json())}


class EventTracker:
    """按状态边界创建/聚合/结束事件。

    规则：
    - 仅当快照给出了行为结论（`primary_risk` 非空且等级为 WARNING/DANGER）才记录；
    - 输入降级（quality != good）不产生行为结论，也不得被当作安全；
    - 同类、同会话且在合并间隔内的连续风险聚合为一个事件；
    - 结束时未达到最短持续时间的候选事件标记为 `discarded`。
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        # 会话 → 最近一次时间轴时刻。离线运行结束后快照会退化为 RESULT_STALE 且
        # 不再带 source_timestamp_ms，若此时回落墙钟，同一会话里就会混用两种时钟：
        # 事件时长按墙钟被放大，且时钟可能倒退，撞上 ck_events_time 约束。
        self._timeline: dict[str, datetime] = {}

    def _clock(self, snapshot: RiskSnapshot, timeline_anchor: datetime | None) -> datetime:
        """事件计时基准。

        离线分析的真实运行时间比素材慢数倍（实测 38 秒素材跑了 324 秒），用墙钟
        会把「视频里 20 秒的喝水」写成 161 秒。只要调用方给出会话起点锚点、快照
        又带有素材时间轴位置，事件时刻就取「锚点 + 素材偏移」；时间轴一旦建立，
        后续没有位置的快照沿用最后位置，不再回退墙钟，也不允许倒退。
        实时模式没有锚点，仍用墙钟。
        """

        session = snapshot.session_id
        position = snapshot.source_timestamp_ms
        previous = self._timeline.get(session)
        if timeline_anchor is not None and position is not None:
            value = timeline_anchor + timedelta(milliseconds=max(0, int(position)))
            if previous is not None and value < previous:
                value = previous
            self._timeline[session] = value
            return value
        if timeline_anchor is not None and previous is not None:
            return previous
        return utcnow()

    def finalize(self, session_id: str) -> EventChange | None:
        """离线素材跑完后收尾：按最后的时间轴位置关闭仍在活动的事件。

        时间轴时钟在停滞期间保持不动，合并间隔永远不会被判超时，所以不显式收尾
        的话事件会一直挂在 active。
        """

        now = self._timeline.get(session_id)
        if now is None:
            return None
        with self._db.session() as session:
            repo = EventRepository(session)
            active = repo.get_active(session_id)
            if active is None:
                return None
            min_duration_ms = SettingsRepository(session).get_int("event.min_duration_ms", 1500)
            closed = repo.close(active, ended_at=now, min_duration_ms=min_duration_ms)
            AuditRepository(session).add(
                action="event.closed",
                resource_type="driving_event",
                resource_id=closed.event_id,
                result="success",
                occurred_at=to_iso_z(now),
                details={"status": closed.status, "duration_ms": closed.duration_ms, "reason": "source_completed"},
            )
            return EventChange(closed.status, event_to_schema(closed))

    def process(
        self, snapshot: RiskSnapshot, *, timeline_anchor: datetime | None = None
    ) -> EventChange | None:
        now = self._clock(snapshot, timeline_anchor)
        recordable = snapshot.primary_risk is not None and snapshot.risk_level in ("WARNING", "DANGER")

        with self._db.session() as session:
            settings = SettingsRepository(session)
            min_duration_ms = settings.get_int("event.min_duration_ms", 1500)
            merge_gap_ms = settings.get_int("event.merge_gap_ms", 2000)
            repo = EventRepository(session)
            active = repo.get_active(snapshot.session_id)

            if recordable:
                event_type = normalize_event_type(snapshot.primary_risk)
                if active is not None and active.event_type == event_type and _within_gap(
                    active.updated_at, now, merge_gap_ms
                ):
                    repo.accumulate(active, snapshot, now=now)
                    return EventChange("updated", event_to_schema(active))

                if active is not None:
                    closed = repo.close(active, ended_at=now, min_duration_ms=min_duration_ms)
                    AuditRepository(session).add(
                        action="event.closed",
                        resource_type="driving_event",
                        resource_id=closed.event_id,
                        result="success",
                        occurred_at=to_iso_z(now),
                        details={"status": closed.status, "duration_ms": closed.duration_ms},
                    )
                    action = closed.status

                created = repo.create(
                    event_id=_new_event_id(),
                    session_id=snapshot.session_id,
                    event_type=event_type,
                    snapshot=snapshot,
                    model_version_id=ModelRegistryRepository(session).temporal_model_id(),
                    now=now,
                )
                AuditRepository(session).add(
                    action="event.created",
                    resource_type="driving_event",
                    resource_id=created.event_id,
                    result="success",
                    occurred_at=to_iso_z(now),
                    details={"event_type": created.event_type, "risk_level": created.max_risk_level},
                )
                return EventChange("created", event_to_schema(created))

            if active is not None and _gap_exceeded(active.updated_at, now, merge_gap_ms):
                closed = repo.close(active, ended_at=now, min_duration_ms=min_duration_ms)
                AuditRepository(session).add(
                    action="event.closed",
                    resource_type="driving_event",
                    resource_id=closed.event_id,
                    result="success",
                    occurred_at=to_iso_z(now),
                    details={"status": closed.status, "duration_ms": closed.duration_ms},
                )
                return EventChange(closed.status, event_to_schema(closed))
        return None


def _new_event_id() -> str:
    import uuid

    return str(uuid.uuid4())


def _within_gap(updated_at: str, now, merge_gap_ms: int) -> bool:
    return (now - parse_iso(updated_at)).total_seconds() * 1000 <= merge_gap_ms


def _gap_exceeded(updated_at: str, now, merge_gap_ms: int) -> bool:
    return (now - parse_iso(updated_at)).total_seconds() * 1000 > merge_gap_ms


__all__ = [
    "EventChange",
    "EventService",
    "EventTracker",
    "event_to_schema",
    "normalize_event_type",
]
