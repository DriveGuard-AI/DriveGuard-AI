"""离线分析任务接口测试（M1）。

真实模型不参与：把 `app.main.build_inference` 换成脚本化端口，
让"帧数跑完"这个终态在毫秒级可控地发生。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models import MonitoringSession
from app.schemas import RiskSnapshot

ADMIN_TOKEN = "test-admin-token"
JPEG = b"\xff\xd8\xff\xd9"
JOBS = "/api/v1/analysis/jobs"


def _snapshot(session_id: str, sequence: int, now: datetime, position: int | None) -> RiskSnapshot:
    return RiskSnapshot(
        session_id=session_id,
        sequence=sequence,
        captured_at=now,
        risk_level="SAFE",
        confidence=0.9,
        quality="good",
        source_timestamp_ms=position,
        inference_mode="fake",
    )


class _IdlePort:
    """实时阶段的占位推理：一直在跑，不产生终态。"""

    mode = "test"
    degraded = False

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def status(self) -> dict:
        return {"state": "running", "frames_sent": 0}

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        return _snapshot(session_id, sequence, now, None)


class _ScriptedPort:
    """离线推理桩：tick 到 `frames` 次之后自报 completed。"""

    mode = "test"
    degraded = False

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.sent = 0

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def status(self) -> dict:
        state = "completed" if self.sent >= self.frames else "running"
        return {"state": state, "frames_sent": self.sent}

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        self.sent += 1
        return _snapshot(session_id, sequence, now, sequence * 40)


def _make_frames(media: Path, count: int) -> Path:
    directory = media / "analysis" / "job-under-test" / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(JPEG)
    return directory


@pytest.fixture()
def analysis_app(tmp_path, monkeypatch):
    media = tmp_path / "media"
    media.mkdir()
    # 帧数由用例在 POST 之前决定，构建器在请求线程里读取。
    script: dict[str, int] = {"frames": 3}

    monkeypatch.setenv("DGB_DATABASE_URL", f"sqlite:///{(tmp_path / 'analysis.db').as_posix()}")
    monkeypatch.setenv("DGB_ADMIN_TOKEN", ADMIN_TOKEN)
    monkeypatch.setenv("DGB_INFERENCE_HZ", "50")
    monkeypatch.setenv("DGB_HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("DGB_INFERENCE_SCENARIO", "default")
    monkeypatch.setenv("DGB_STATIC_DIR", str(tmp_path / "dist"))
    monkeypatch.setenv("DGB_MEDIA_DIR", str(media))

    import app.main as main_module

    def factory(config):
        if config.inference_mode == "live":
            return _ScriptedPort(script["frames"])
        return _IdlePort()

    monkeypatch.setattr(main_module, "build_inference", factory)
    application = main_module.create_app()
    with TestClient(application) as client:
        yield client, media, script


def _post(client: TestClient, path, headers, kind: str = "frames"):
    return client.post(
        JOBS,
        json={"source_kind": kind, "source_path": str(path)},
        headers=headers,
    )


def _wait_finished(client: TestClient, job_id: str, headers, timeout: float = 10.0):
    """轮询到终态，顺带收集 frames_done 序列用于单调性断言。"""

    deadline = time.monotonic() + timeout
    samples: list[int] = []
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS}/{job_id}", headers=headers).json()
        samples.append(body["frames_done"])
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body, samples
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束，采样 {samples}")


def test_create_job_runs_to_completion(analysis_app, auth_headers):
    client, media, script = analysis_app
    frames = _make_frames(media, 12)
    script["frames"] = 12

    response = _post(client, frames, auth_headers)
    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "running"
    assert created["frames_total"] == 12
    assert created["source_kind"] == "frames"
    assert created["source_path"] == str(frames.resolve())
    assert created["ended_at"] is None

    finished, samples = _wait_finished(client, created["job_id"], auth_headers)
    assert finished["status"] == "completed"
    assert finished["frames_done"] == 12
    assert finished["timeline_ms"] > 0
    assert finished["ended_at"] is not None
    # 进度只能前进：中途任何一次回退都会让页面进度条抖动。
    assert samples == sorted(samples)


def test_job_opens_new_session_and_closes_previous(analysis_app, auth_headers):
    client, media, script = analysis_app
    frames = _make_frames(media, 3)
    script["frames"] = 3
    context = client.app.state.context
    previous = context.engine.session_id
    assert previous

    created = _post(client, frames, auth_headers).json()
    assert created["session_id"] != previous

    with context.db.session() as session:
        old = session.get(MonitoringSession, previous)
        assert old is not None and old.ended_at is not None
        new = session.get(MonitoringSession, created["session_id"])
        assert new is not None and new.ended_at is None


def test_cancel_running_job_stops_engine(analysis_app, auth_headers):
    client, media, script = analysis_app
    frames = _make_frames(media, 5)
    # 让脚本远跑不完，确保取消发生在 running 状态。
    script["frames"] = 1_000_000
    context = client.app.state.context

    created = _post(client, frames, auth_headers).json()
    assert created["status"] == "running"

    cancelled = client.delete(f"{JOBS}/{created['job_id']}", headers=auth_headers)
    assert cancelled.status_code == 200
    body = cancelled.json()
    assert body["status"] == "cancelled"
    assert body["ended_at"] is not None
    assert context.engine.status()["state"] == "stopped"

    again = client.delete(f"{JOBS}/{created['job_id']}", headers=auth_headers)
    assert again.status_code == 409
    assert again.json()["code"] == "ANALYSIS_JOB_CONFLICT"


def test_missing_source_is_rejected(analysis_app, auth_headers):
    client, media, _ = analysis_app
    response = _post(client, media / "analysis" / "does-not-exist", auth_headers)
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"


def test_path_outside_media_root_is_rejected(analysis_app, auth_headers, tmp_path):
    client, _, _ = analysis_app
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "000000.jpg").write_bytes(JPEG)

    response = _post(client, outside, auth_headers)
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"


def test_unsupported_source_kind_is_rejected(analysis_app, auth_headers):
    client, media, _ = analysis_app
    frames = _make_frames(media, 2)
    response = _post(client, frames, auth_headers, kind="camera")
    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"


def test_unknown_job_returns_404(analysis_app, auth_headers):
    client, _, _ = analysis_app
    response = client.get(f"{JOBS}/00000000-0000-0000-0000-000000000000", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["code"] == "ANALYSIS_JOB_NOT_FOUND"


def test_list_filters_by_status(analysis_app, auth_headers):
    client, media, script = analysis_app
    frames = _make_frames(media, 4)
    script["frames"] = 4
    created = _post(client, frames, auth_headers).json()
    _wait_finished(client, created["job_id"], auth_headers)

    all_jobs = client.get(JOBS, headers=auth_headers).json()
    assert all_jobs["total"] == 1
    assert all_jobs["items"][0]["job_id"] == created["job_id"]

    completed = client.get(f"{JOBS}?status=completed", headers=auth_headers).json()
    assert completed["total"] == 1

    running = client.get(f"{JOBS}?status=running", headers=auth_headers).json()
    assert running["total"] == 0


def test_list_rejects_unknown_status(analysis_app, auth_headers):
    client, _, _ = analysis_app
    response = client.get(f"{JOBS}?status=whatever", headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_requires_authentication(analysis_app):
    client, media, _ = analysis_app
    frames = _make_frames(media, 1)
    assert client.get(JOBS).status_code == 401
    assert _post(client, frames, {}).status_code == 401
