"""契约一致性检查（REQ-010 / API_DESIGN.md 第 10 节）。

实现侧生成的 OpenAPI 必须覆盖 `docs/api/openapi.yaml` 声明的全部端点。
"""

from __future__ import annotations

EXPECTED_OPERATIONS = {
    ("get", "/api/v1/health"),
    ("get", "/api/v1/system/status"),
    ("get", "/api/v1/events"),
    ("get", "/api/v1/events/{event_id}"),
    ("get", "/api/v1/statistics/summary"),
    ("get", "/api/v1/settings"),
    ("put", "/api/v1/settings"),
    ("get", "/api/v1/models"),
    ("get", "/api/v1/monitoring/snapshot"),
    ("get", "/api/v1/monitoring/preview"),
    ("get", "/api/v1/monitoring/preview/state"),
}

EXPECTED_SCHEMAS = {
    "RiskSnapshot",
    "EventOut",
    "EventListResponse",
    "StatisticsSummary",
    "SettingsResponse",
    "SettingsUpdateRequest",
    "SettingsUpdateResponse",
    "ModelInfo",
    "SystemStatus",
    "PreviewState",
    "TaskDetail",
}


def test_openapi_covers_documented_endpoints(client):
    document = client.get("/openapi.json").json()
    paths = document["paths"]
    actual = {
        (method, path)
        for path, operations in paths.items()
        for method in operations
        if method in {"get", "put", "post", "delete"}
    }
    missing = EXPECTED_OPERATIONS - actual
    assert not missing, f"缺少文档声明的端点：{sorted(missing)}"


def test_openapi_exposes_contract_schemas(client):
    schemas = set(client.get("/openapi.json").json()["components"]["schemas"])
    missing = EXPECTED_SCHEMAS - schemas
    assert not missing, f"缺少契约 schema：{sorted(missing)}"


def test_documented_error_codes_are_reachable(client, auth_headers):
    # AUTH_REQUIRED
    assert client.get("/api/v1/settings").json()["code"] == "AUTH_REQUIRED"
    # VALIDATION_ERROR
    assert client.get("/api/v1/events?page_size=0", headers=auth_headers).json()["code"] == "VALIDATION_ERROR"
    # EVENT_NOT_FOUND
    assert (
        client.get("/api/v1/events/44444444-4444-4444-4444-444444444444", headers=auth_headers).json()["code"]
        == "EVENT_NOT_FOUND"
    )
    # TIME_RANGE_TOO_LARGE
    assert (
        client.get(
            "/api/v1/statistics/summary"
            "?started_from=2026-01-01T00:00:00Z&started_to=2026-09-01T00:00:00Z",
            headers=auth_headers,
        ).json()["code"]
        == "TIME_RANGE_TOO_LARGE"
    )
