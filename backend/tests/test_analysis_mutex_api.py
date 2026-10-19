"""模式互斥：实时在跑时不许静默被离线任务顶掉，以及切回实时的入口。

没有这道闸门的话，上传一个视频就会把台上的实时监控停掉，而调用方只会看到 201。
"""

from __future__ import annotations

import contextlib
import time

import pytest
from fastapi.testclient import TestClient

from tests.analysis_support import (
    JOBS,
    configure_env,
    install_ports,
    make_frames,
    write_sample_video,
)

UPLOAD = f"{JOBS}/upload"
RESUME = "/api/v1/monitoring/resume"
SYSTEM = "/api/v1/system/status"


@pytest.fixture()
def mutex_client(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def build(**overrides):
        media = tmp_path / "media"
        media.mkdir()
        configure_env(monkeypatch, tmp_path, media, **overrides)
        install_ports(monkeypatch, {"frames": None})

        import app.main as main_module

        with TestClient(main_module.create_app()) as client:
            yield client, media

    return build


def _wait_finished(client, job_id: str, headers, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS}/{job_id}", headers=headers).json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束")


def test_create_without_force_is_rejected_while_realtime_runs(mutex_client, auth_headers):
    with mutex_client() as (client, media):
        frames = make_frames(media, 3)
        context = client.app.state.context
        session_before = context.engine.session_id

        response = client.post(JOBS, json={"source_path": str(frames)}, headers=auth_headers)

        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "MONITORING_ACTIVE"
        assert "force" in body["message"]
        # 被拒绝的请求不能有任何副作用。
        assert context.engine.session_id == session_before
        assert client.get(JOBS, headers=auth_headers).json()["total"] == 0


def test_upload_without_force_does_not_even_decode(mutex_client, auth_headers):
    with mutex_client() as (client, media):
        video = write_sample_video(media / "clip.mp4", frames=10)

        with video.open("rb") as handle:
            response = client.post(
                UPLOAD,
                files={"file": (video.name, handle, "video/mp4")},
                data={"step_ms": "40"},
                headers=auth_headers,
            )

        assert response.status_code == 409
        assert response.json()["code"] == "MONITORING_ACTIVE"
        # 互斥在抽帧之前判定：不该为了一个会被拒绝的请求白跑一遍解码。
        assert not (media / "analysis").exists()


def test_force_takeover_then_resume_realtime(mutex_client, auth_headers):
    with mutex_client() as (client, media):
        frames = make_frames(media, 3)
        context = client.app.state.context
        realtime_session = context.engine.session_id

        created = client.post(
            JOBS, json={"source_path": str(frames), "force": True}, headers=auth_headers
        )
        assert created.status_code == 201
        job = created.json()
        assert context.engine.session_id != realtime_session
        status = client.get(SYSTEM, headers=auth_headers).json()
        assert status["pipeline"]["playback"] is True

        assert _wait_finished(client, job["job_id"], auth_headers)["status"] == "completed"

        resumed = client.post(RESUME, headers=auth_headers)
        assert resumed.status_code == 200
        payload = resumed.json()
        assert payload["playback"] is False
        assert payload["state"] == "running"
        assert payload["session_id"] and payload["session_id"] != job["session_id"]

        # 切回实时必须是“实时实现”在跑，而不是继续放刚才那段素材。
        # IdlePort 不提供素材时间轴，ScriptedPort 会给出 sequence*40。
        time.sleep(0.3)
        snapshot = client.get("/api/v1/monitoring/snapshot", headers=auth_headers).json()
        assert snapshot["source_timestamp_ms"] is None

        # 幂等：再点一次不会又开一个新会话。
        again = client.post(RESUME, headers=auth_headers).json()
        assert again["session_id"] == payload["session_id"]


def test_resume_conflicts_while_job_is_running(mutex_client, auth_headers):
    with mutex_client() as (client, media):
        frames = make_frames(media, 200)  # 20 Hz 下足够跑十几秒
        created = client.post(
            JOBS, json={"source_path": str(frames), "force": True}, headers=auth_headers
        ).json()
        assert created["status"] == "running"

        response = client.post(RESUME, headers=auth_headers)
        assert response.status_code == 409
        assert response.json()["code"] == "ANALYSIS_JOB_CONFLICT"

        client.delete(f"{JOBS}/{created['job_id']}", headers=auth_headers)


def test_resume_requires_admin_permission(mutex_client):
    with mutex_client() as (client, _media):
        # 匿名只读模式下不允许启动/停止管线。
        assert client.post(RESUME).status_code == 401
