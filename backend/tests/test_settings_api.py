"""API-006 / API-007 配置读取与乐观锁更新。"""

from __future__ import annotations


def test_settings_requires_authentication(client):
    response = client.get("/api/v1/settings")
    assert response.status_code == 401
    assert response.json()["code"] == "AUTH_REQUIRED"


def test_settings_secret_is_not_echoed(client, auth_headers):
    response = client.get("/api/v1/settings", headers=auth_headers)
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] >= 1
    items = {item["key"]: item for item in payload["items"]}
    assert "alarm.cooldown_ms" in items
    secret = items["security.admin_token"]
    assert secret["is_secret"] is True
    assert secret["value"] is None
    assert "is_configured" in secret


def test_settings_optimistic_lock_conflict(client, auth_headers):
    current = client.get("/api/v1/settings", headers=auth_headers).json()["version"]

    applied = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": current, "changes": {"alarm.cooldown_ms": 4200}},
    )
    assert applied.status_code == 200

    # 客户端仍持有旧版本号再次提交：必须冲突且不覆盖当前值
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": current, "changes": {"alarm.cooldown_ms": 9999}},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "CONFIG_VERSION_CONFLICT"

    items = {item["key"]: item for item in client.get("/api/v1/settings", headers=auth_headers).json()["items"]}
    assert items["alarm.cooldown_ms"]["value"] == 4200


def test_settings_rejects_out_of_range_value(client, auth_headers):
    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": version, "changes": {"risk.warning_threshold": 1.5}},
    )
    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "VALIDATION_ERROR"
    assert payload["details"][0]["field"] == "risk.warning_threshold"


def test_settings_rejects_unknown_key(client, auth_headers):
    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": version, "changes": {"unknown.key": 1}},
    )
    assert response.status_code == 422
    assert response.json()["details"][0]["reason"] == "unknown setting key"


def test_settings_rejects_cross_field_inconsistency(client, auth_headers):
    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={
            "version": version,
            "changes": {"risk.warning_threshold": 0.95, "risk.danger_threshold": 0.5},
        },
    )
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "risk.warning_threshold"


def test_settings_update_bumps_version_and_reports_restart(client, auth_headers):
    before = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": before, "changes": {"camera.device_index": 2, "alarm.cooldown_ms": 8000}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == before + 1
    assert payload["applied"]["camera.device_index"] == 2
    assert "camera.device_index" in payload["restart_required"]

    after = client.get("/api/v1/settings", headers=auth_headers).json()
    assert after["version"] == before + 1
    items = {item["key"]: item for item in after["items"]}
    assert items["alarm.cooldown_ms"]["value"] == 8000
