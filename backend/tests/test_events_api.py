"""API-003 / API-004 事件查询的校验与分页行为。"""

from __future__ import annotations

from app.repositories.events import MAX_PAGE_SIZE


def test_empty_events_list(client, auth_headers):
    response = client.get("/api/v1/events", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload == {"items": [], "page": 1, "page_size": 20, "total": 0, "has_next": False}


def test_page_size_bounds(client, auth_headers):
    assert client.get("/api/v1/events?page_size=0", headers=auth_headers).status_code == 422
    assert client.get("/api/v1/events?page_size=1", headers=auth_headers).status_code == 200
    assert (
        client.get(f"/api/v1/events?page_size={MAX_PAGE_SIZE}", headers=auth_headers).status_code == 200
    )
    assert client.get(f"/api/v1/events?page_size={MAX_PAGE_SIZE + 1}", headers=auth_headers).status_code == 422


def test_reversed_time_range_returns_invalid_time_range(client, auth_headers):
    response = client.get(
        "/api/v1/events"
        "?started_from=2026-09-10T00:00:00Z"
        "&started_to=2026-09-01T00:00:00Z",
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_TIME_RANGE"


def test_unknown_enum_returns_validation_error(client, auth_headers):
    response = client.get("/api/v1/events?risk_level=UNKNOWN", headers=auth_headers)
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["details"][0]["field"] == "risk_level"


def test_illegal_uuid_is_rejected(client, auth_headers):
    response = client.get("/api/v1/events/not-a-uuid", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_missing_event_returns_not_found(client, auth_headers):
    response = client.get("/api/v1/events/22222222-2222-2222-2222-222222222222", headers=auth_headers)
    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "EVENT_NOT_FOUND"
    # NFR-004：错误响应不得泄露绝对路径
    assert "C:\\" not in payload["message"]
    assert "\\" not in payload["message"]


def test_events_ordered_by_started_at_desc(client, app_context, auth_headers):
    from app.repositories.events import EventFilters

    events = app_context.events.list_events(EventFilters(), page=1, page_size=10)
    assert events.total >= 0
    starts = [item.started_at for item in events.items]
    assert starts == sorted(starts, reverse=True)
