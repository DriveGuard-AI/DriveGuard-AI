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
from app.timeutil import parse_iso, to_iso_z, utcnow

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


def test_offline_timeline_anchor_keeps_video_duration(database, clock):
    """离线分析的事件时长必须按素材时间轴算，不能按墙钟。

    实测过 38.4 秒素材跑了 324 秒墙钟，若用墙钟，一条 20 秒的喝水会写成 161 秒。
    """

    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)
    anchor = clock.current

    def offline(timeline_ms: int, *, risk_level: str = "WARNING", primary_risk: str | None = "drinking"):
        return _snapshot(clock, risk_level=risk_level, primary_risk=primary_risk).model_copy(
            update={"source_timestamp_ms": timeline_ms}
        )

    created = tracker.process(offline(0), timeline_anchor=anchor)
    assert created is not None and created.action == "created"

    # 墙钟每步推进 45 秒（模拟离线处理远慢于实时），素材只前进几十毫秒。
    clock.advance(45_000)
    assert tracker.process(offline(40), timeline_anchor=anchor).action == "updated"
    clock.advance(45_000)
    assert tracker.process(offline(200), timeline_anchor=anchor).action == "updated"

    # 素材到了 3 秒且风险消失，超过合并间隔即结束。
    clock.advance(45_000)
    ended = tracker.process(offline(3000, risk_level="SAFE", primary_risk=None), timeline_anchor=anchor)
    assert ended is not None and ended.action == "ended"

    record = _events(database)[0][0]
    assert record.event_type == "drinking"
    # 墙钟已走过 135 秒，但事件时长必须是素材的 3 秒。
    assert record.duration_ms == 3000
    assert parse_iso(record.started_at) == anchor
    assert record.ended_at is not None
    assert record.ended_at is not None


def test_stale_snapshot_does_not_fall_back_to_wall_clock(database, clock):
    """素材跑完后快照退化为 RESULT_STALE，不得把事件时钟切回墙钟。

    真实故障：10 秒素材被写成 21 秒事件，且同一会话内两种时钟来回跳，
    最终撞上 ck_events_time（ended_at < started_at）。
    """

    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)
    anchor = clock.current

    def at(timeline_ms: int, **kwargs):
        return _snapshot(clock, **kwargs).model_copy(update={"source_timestamp_ms": timeline_ms})

    tracker.process(at(3960, risk_level="WARNING", primary_risk="drinking"), timeline_anchor=anchor)
    clock.advance(20_000)
    tracker.process(at(5200, risk_level="WARNING", primary_risk="drinking"), timeline_anchor=anchor)
    clock.advance(20_000)
    tracker.process(at(6400, risk_level="WARNING", primary_risk="drinking"), timeline_anchor=anchor)

    # 墙钟再走两分钟，但快照已没有时间轴位置：应当沿用最后位置而不是墙钟。
    clock.advance(120_000)
    assert tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None), timeline_anchor=anchor) is None

    closed = tracker.finalize(SESSION_ID)
    assert closed is not None and closed.action == "ended"
    record = _events(database)[0][0]
    assert record.duration_ms == 2440  # 3.96s → 6.40s，与墙钟的 160 秒无关


def test_timeline_never_rewinds_within_a_session(database, clock):
    """会话内时间轴只会前进；倒退会被钳住，避免 ended_at < started_at。"""

    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)
    anchor = clock.current

    def at(timeline_ms: int) -> RiskSnapshot:
        return _snapshot(clock, risk_level="WARNING", primary_risk="fatigue").model_copy(
            update={"source_timestamp_ms": timeline_ms}
        )

    tracker.process(at(9000), timeline_anchor=anchor)
    # 模拟会话重开后时间轴回到 0：不得让事件时钟倒退。
    change = tracker.process(at(0), timeline_anchor=anchor)
    assert change is not None
    record = _events(database)[0][0]
    assert record.started_at is not None


def test_without_timeline_the_tracker_still_uses_wall_clock(database, clock):
    """实时模式没有锚点，行为必须与改动前一致。"""

    _configure(database, merge_gap_ms=2000, min_duration_ms=1500)
    tracker = EventTracker(database)

    tracker.process(_snapshot(clock, risk_level="WARNING", primary_risk="fatigue"))
    clock.advance(3000)
    ended = tracker.process(_snapshot(clock, risk_level="SAFE", primary_risk=None))
    assert ended is not None and ended.action == "ended"

    record = _events(database)[0][0]
    assert record.duration_ms == 3000


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
