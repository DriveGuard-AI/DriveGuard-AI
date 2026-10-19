"""API-005 统计摘要。"""

from __future__ import annotations


def test_empty_statistics_returns_zero_not_error(client, auth_headers):
    response = client.get(
        "/api/v1/statistics/summary"
        "?started_from=2026-09-01T00:00:00Z"
        "&started_to=2026-09-10T23:59:59Z",
        headers=auth_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    # FR-008：无数据时返回空统计结果，不得以服务异常代替
    assert payload["total_events"] == 0
    assert payload["by_type"] == []
    assert payload["by_risk_level"] == []
    assert payload["trend"] == []


def test_statistics_requires_time_range(client, auth_headers):
    response = client.get("/api/v1/statistics/summary", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_statistics_range_too_large(client, auth_headers):
    response = client.get(
        "/api/v1/statistics/summary"
        "?started_from=2026-01-01T00:00:00Z"
        "&started_to=2026-09-10T00:00:00Z",
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["code"] == "TIME_RANGE_TOO_LARGE"


def test_reversed_range_is_invalid(client, auth_headers):
    response = client.get(
        "/api/v1/statistics/summary"
        "?started_from=2026-09-10T00:00:00Z"
        "&started_to=2026-09-01T00:00:00Z",
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_TIME_RANGE"


def test_session_filter_is_echoed_and_scoped(client, auth_headers, app_context):
    """离线重跑会多出会话；统计必须能把结果限在单个会话上。"""

    from app.repositories.events import EventRepository
    from app.repositories.sessions import SessionRepository
    from app.schemas import RiskSnapshot
    from app.timeutil import to_iso_z, utcnow

    now = utcnow()
    event_clock = now.replace(microsecond=0)
    snapshot = RiskSnapshot(
        session_id="ignored",
        sequence=1,
        captured_at=now,
        risk_level="DANGER",
        confidence=0.9,
        quality="good",
        inference_mode="fake",
    )
    only_session = "22222222-2222-2222-2222-222222222222"
    other_session = "33333333-3333-3333-3333-333333333333"
    with app_context.db.session() as session:
        sessions = SessionRepository(session)
        for session_id in (only_session, other_session):
            sessions.create(
                session_id=session_id,
                started_at=to_iso_z(event_clock),
                camera_id="cam-test",
                device_id="edge-test",
                model_bundle_version="test",
            )
        repository = EventRepository(session)
        for session_id in (only_session, other_session):
            repository.create(
                event_id=f"event-{session_id[:4]}",
                session_id=session_id,
                event_type="fatigue",
                snapshot=snapshot,
                now=event_clock,
                model_version_id=None,
            )

    window = (
        f"?started_from={to_iso_z(event_clock)}"
        f"&started_to={to_iso_z(event_clock)}"
    )
    scoped = client.get(
        f"/api/v1/statistics/summary{window}&session_id={only_session}",
        headers=auth_headers,
    )
    assert scoped.status_code == 200
    assert scoped.json()["session_id"] == only_session
    assert scoped.json()["total_events"] == 1

    every = client.get(f"/api/v1/statistics/summary{window}", headers=auth_headers)
    assert every.json()["total_events"] == 2
    assert every.json()["session_id"] is None


def test_session_filter_rejects_non_uuid(client, auth_headers):
    response = client.get(
        "/api/v1/statistics/summary"
        "?started_from=2026-09-01T00:00:00Z"
        "&started_to=2026-09-10T00:00:00Z"
        "&session_id=not-a-uuid",
        headers=auth_headers,
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
