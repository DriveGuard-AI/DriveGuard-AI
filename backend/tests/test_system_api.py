"""API-001 / API-002 / API-008 / API-009 的基本契约测试。"""

from __future__ import annotations


def test_health_is_anonymous(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["time"].endswith("Z")
    assert "service_version" in payload


def test_request_id_header_is_returned(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_system_status_exposes_fake_inference(client, auth_headers):
    response = client.get("/api/v1/system/status", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    # 无模型阶段必须如实声明降级，避免被误读为真实模型能力
    assert payload["pipeline"]["inference_mode"] == "fake"
    assert payload["pipeline"]["degraded"] is True
    assert payload["model"]["loaded"] is False
    assert payload["database"]["writable"] is True
    assert payload["websocket"]["max_hz"] == 10.0


def test_snapshot_matches_contract(client, auth_headers):
    response = client.get("/api/v1/monitoring/snapshot", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    for key in (
        "schema_version",
        "session_id",
        "sequence",
        "captured_at",
        "risk_level",
        "confidence",
        "quality",
        "risk_scores",
        "features",
        "detections",
        "reason_codes",
    ):
        assert key in payload
    assert payload["risk_level"] in {"SAFE", "WARNING", "DANGER"}
    assert payload["schema_version"] == "1.0"
    assert payload["captured_at"].endswith("Z")


def test_models_hide_artifact_path(client, auth_headers):
    response = client.get("/api/v1/models", headers=auth_headers)
    assert response.status_code == 200
    models = response.json()
    assert models, "占位模型应已注册"
    for model in models:
        assert "artifact_path" not in model
        assert "sha256" not in model
        assert model["model_type"] in {"detector", "landmarker", "temporal", "fusion"}


def test_models_rejects_unknown_type(client, auth_headers):
    response = client.get("/api/v1/models?model_type=unknown", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
