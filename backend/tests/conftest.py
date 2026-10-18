"""pytest 公共夹具。

测试基线见 `docs/testing/TEST_STRATEGY.md`：集成测试使用临时 SQLite。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ADMIN_TOKEN = "test-admin-token"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """启动完整应用（含推理引擎）的测试客户端。"""

    monkeypatch.setenv("DGB_DATABASE_URL", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    monkeypatch.setenv("DGB_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("DGB_INFERENCE_HZ", "50")
    monkeypatch.setenv("DGB_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("DGB_INFERENCE_SCENARIO", "default")
    monkeypatch.setenv("DGB_STATIC_DIR", str(tmp_path / "dist"))
    monkeypatch.setenv("DGB_MEDIA_DIR", str(tmp_path / "media"))

    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def app_context(client):
    return client.app.state.context


@pytest.fixture()
def auth_headers():
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture()
def database(tmp_path):
    """不启动推理引擎的独立数据库，用于服务层单元测试。"""

    from app.database import Database
    from app.repositories.models_registry import ModelRegistryRepository
    from app.repositories.sessions import SessionRepository
    from app.repositories.settings import SettingsRepository
    from app.timeutil import to_iso_z, utcnow

    db = Database(f"sqlite:///{(tmp_path / 'unit.db').as_posix()}")
    db.create_all()
    now_iso = to_iso_z(utcnow())
    with db.session() as session:
        SettingsRepository(session).ensure_defaults(now_iso=now_iso)
        ModelRegistryRepository(session).ensure_stub_models(now_iso=now_iso)
        SessionRepository(session).create(
            session_id="11111111-1111-1111-1111-111111111111",
            started_at=now_iso,
            camera_id="cam-unit",
            device_id="edge-unit",
            model_bundle_version="unit",
        )
    yield db
    db.dispose()
