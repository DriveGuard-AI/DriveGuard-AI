"""事件生命周期单元测试（`ARCHITECTURE.md` §4.2）。

使用受控时钟，验证 created → updated → ended/discarded 的边界行为。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.repositories.events import EventFilters, EventRepository
from app.repositories.settings import SettingsRepository
from app.schemas import RiskSnapshot
from app.services.events import EventTracker
from app.timeutil import to_iso_z, utcnow

SESSION_ID = "11111111-1111-1111-1111-111111111111"


class Clock:
    def __init__(self) -> None:
        self.current = datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, milliseconds: int) -> None:
        self.current = self.current + timedelta(milliseconds=milliseconds)


@pytest.fixture()
def clock(monkeypatch):
    controlled = Clock()
    monkeypatch.setattr("app.services.events.utcnow", controlled)
    return controlled


def _snapshot(clock: Clock, *, risk_level: str, primary_risk: str | None, quality: str = "good") -> RiskSnapshot:
    return RiskSnapshot(
        session_id=SESSION_ID,
        sequence=1,
        captured_at=clock.current,
        risk_level=risk_level,  # type: ignore[arg-type]
        primary_risk=primary_risk,
        confidence=0.9 if risk_level != "SAFE" else 0.05,
        quality=quality,  # type: ignore[arg-type]
        risk_scores={"fatigue": 0.9} if primary_risk else {},
        reason_codes=["EYES_CLOSED_SUSTAINED"] if primary_risk else ["NO_RISK"],
    )


def _configure(database, *, merge_gap_ms: int, min_duration_ms: int) -> None:
    with database.session() as session:
        repo = SettingsRepository(session)
        now_iso = to_iso_z(utcnow())
        repo.set_value("event.merge_gap_ms", merge_gap_ms, value_type="integer", actor="test", now_iso=now_iso)
        repo.set_value(
            "event.min_duration_ms", min_duration_ms, value_type="integer", actor="test", now_iso=now_iso
        )


def _events(database):
    with database.session() as session:
        records, total = EventRepository(session).list(EventFilters(), page=1, page_size=50)
        return records, total


def test_safe_snapshots_do_not_create_events(database, clock):
    tracker = EventTracker(database)
    for _ in range(5):
        assert tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None)) is None
        clock.advance(200)
    assert _events(database)[1] == 0


def test_degraded_input_is_not_treated_as_safe_and_creates_no_event(database, clock):
    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)

    change = tracker.process(
        _snapshot(clock, risk_level="WARNING", primary_risk=None, quality="degraded")
    )
    # 输入降级不构成行为结论：不创建事件，但风险等级也不是 SAFE
    assert change is None
    assert _events(database)[1] == 0


def test_event_created_accumulated_and_ended(database, clock):
    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)

    created = tracker.process(_snapshot(clock, risk_level="WARNING", primary_risk="fatigue"))
    assert created is not None and created.action == "created"

    clock.advance(2000)
    updated = tracker.process(_snapshot(clock, risk_level="DANGER", primary_risk="fatigue"))
    assert updated is not None and updated.action == "updated"
    assert updated.event.max_risk_level == "DANGER"

    # 仍在合并窗口内，事件保持活动
    clock.advance(500)
    assert tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None)) is None

    clock.advance(3000)
    ended = tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None))
    assert ended is not None and ended.action == "ended"

    records, total = _events(database)
    assert total == 1
    record = records[0]
    assert record.status == "ended"
    assert record.event_type == "fatigue"
    assert record.max_risk_level == "DANGER"
    assert record.duration_ms >= 1500
    assert record.ended_at is not None


def test_short_candidate_is_discarded(database, clock):
    _configure(database, merge_gap_ms=500, min_duration_ms=5000)
    tracker = EventTracker(database)

    created = tracker.process(_snapshot(clock, risk_level="WARNING", primary_risk="phone_use"))
    assert created is not None and created.action == "created"

    clock.advance(600)
    result = tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None))
    assert result is not None and result.action == "discarded"

    records, total = _events(database)
    assert total == 1
    assert records[0].status == "discarded"


def test_risk_type_change_closes_previous_event(database, clock):
    _configure(database, merge_gap_ms=2000, min_duration_ms=0)
    tracker = EventTracker(database)

    tracker.process(_snapshot(clock, risk_level="WARNING", primary_risk="fatigue"))
    clock.advance(300)
    change = tracker.process(_snapshot(clock, risk_level="WARNING", primary_risk="phone_use"))

    assert change is not None and change.action == "created"
    records, total = _events(database)
    assert total == 2
    statuses = {record.event_type: record.status for record in records}
    assert statuses["fatigue"] == "ended"
    assert statuses["phone_use"] == "active"
