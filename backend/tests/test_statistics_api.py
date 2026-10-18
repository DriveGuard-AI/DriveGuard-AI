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
