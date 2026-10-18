"""NFR-003 / NFR-004 隐私与安全边界。"""

from __future__ import annotations

from sqlalchemy import select

from app.models import AuditLog


def test_no_video_or_frames_are_persisted(client, app_context):
    # 默认配置：不保存原始视频，也不创建媒体目录内容
    client.get("/api/v1/monitoring/snapshot", headers={"Authorization": "Bearer test-admin-token"})
    media_dir = app_context.config.media_dir
    if media_dir.exists():
        assert list(media_dir.iterdir()) == []


def test_events_never_expose_clip(client, app_context, auth_headers):
    response = client.get("/api/v1/events?page_size=50", headers=auth_headers)
    assert response.status_code == 200
    for item in response.json()["items"]:
        assert item["clip_available"] is False
        assert "clip_path" not in item


def test_audit_log_redacts_secret_values(client, app_context, auth_headers):
    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": version, "changes": {"security.admin_token": "super-secret-value"}},
    )
    assert response.status_code == 200

    with app_context.db.session() as session:
        rows = list(session.execute(select(AuditLog).where(AuditLog.action == "settings.update")).scalars())

    assert rows, "配置更新应写入审计日志"
    for row in rows:
        assert "super-secret-value" not in row.details_json
        assert "***" in row.details_json or "changed_keys" in row.details_json


def test_error_responses_are_structured_and_path_safe(client, auth_headers):
    response = client.get("/api/v1/events/33333333-3333-3333-3333-333333333333", headers=auth_headers)
    payload = response.json()
    assert set(payload) >= {"code", "message", "details", "request_id"}
    assert payload["request_id"]
    assert "\\" not in payload["message"]
    assert "/" not in payload["message"]


def test_unknown_config_key_is_rejected(client, auth_headers):
    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": version, "changes": {"../../etc/passwd": "x"}},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
