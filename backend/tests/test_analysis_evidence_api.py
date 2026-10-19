"""回放证据接口测试：时间轴、事件帧区间、原片/帧媒体端点。"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.repositories.events import EventRepository
from app.repositories.sessions import SessionRepository
from app.schemas import RiskSnapshot
from tests.analysis_support import (
    ADMIN_TOKEN,
    JOBS,
    configure_env,
    install_ports,
    write_sample_video,
)

EVIDENCE = "/evidence"


@pytest.fixture()
def evidence_client(tmp_path, monkeypatch):
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


def _upload(client, headers, video: Path, **form):
    data = {"step_ms": "40", "size": "160x120", "quality": "85", "force": "true"}
    data.update({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in form.items()})
    with video.open("rb") as handle:
        return client.post(
            f"{JOBS}/upload",
            files={"file": (video.name, handle, "video/mp4")},
            data=data,
            headers=headers,
        )


def _wait_finished(client, job_id: str, headers, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS}/{job_id}", headers=headers).json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束")


def _seed_event(context, job, *, offset_ms: int, duration_ms: int) -> None:
    """在任务的会话里插入一条事件，位置精确可控。"""

    from app.timeutil import parse_iso, to_iso_z

    with context.db.session() as session:
        session_row = SessionRepository(session).get(job["session_id"])
        assert session_row is not None
        started_at = parse_iso(session_row.started_at)
        from datetime import timedelta

        event_start = started_at + timedelta(milliseconds=offset_ms)
        EventRepository(session).create(
            event_id="seed-event",
            session_id=job["session_id"],
            event_type="fatigue",
            snapshot=RiskSnapshot(
                session_id=job["session_id"],
                sequence=1,
                captured_at=event_start,
                risk_level="DANGER",
                confidence=0.9,
                quality="good",
                inference_mode="fake",
            ),
            now=event_start,
            model_version_id=None,
        )
        # create() 的时长是 0；直接把时长写成想要的值，模拟事件已经结束。
        from app.models import DrivingEvent

        record = session.get(DrivingEvent, "seed-event")
        assert record is not None
        record.duration_ms = duration_ms
        record.ended_at = to_iso_z(event_start + timedelta(milliseconds=duration_ms))
        # ck_events_active_boundary：有 ended_at 就不能还是 active。
        record.status = "ended"


def test_evidence_records_timeline_and_maps_event_frames(evidence_client, auth_headers):
    with evidence_client() as (client, media):
        video = write_sample_video(media / "clip.mp4", frames=50)
        job = _upload(client, auth_headers, video).json()
        finished = _wait_finished(client, job["job_id"], auth_headers)
        assert finished["status"] == "completed"

        evidence = client.get(f"{JOBS}/{job['job_id']}{EVIDENCE}", headers=auth_headers)
        assert evidence.status_code == 200, evidence.text
        body = evidence.json()
        assert body["step_ms"] == 40
        assert body["frames_total"] == job["frames_total"]
        assert body["duration_ms"] >= (job["frames_total"] - 1) * 40
        # 逐 tick 风险曲线必须落盘：完成的任务至少每帧一个点。
        assert len(body["timeline"]) > 0
        ms_values = [point["ms"] for point in body["timeline"]]
        assert ms_values == sorted(ms_values)
        assert {point["risk_level"] for point in body["timeline"]} <= {"SAFE", "WARNING", "DANGER"}
        assert all(point["frame"] == point["ms"] // 40 for point in body["timeline"])

        timeline_file = media / "analysis" / job["job_id"] / "timeline.ndjson"
        assert timeline_file.is_file()
        assert len(timeline_file.read_text(encoding="utf-8").splitlines()) == len(body["timeline"])

        # 事件 → 帧号的换算：400 ms / 40 = 第 10 帧，时长 800 ms = 20 帧
        _seed_event(client.app.state.context, job, offset_ms=400, duration_ms=800)
        again = client.get(f"{JOBS}/{job['job_id']}{EVIDENCE}", headers=auth_headers).json()
        assert len(again["events"]) == 1
        marker = again["events"][0]
        assert marker["started_ms"] == 400
        assert marker["start_frame"] == 10
        assert marker["end_frame"] == 30
        assert marker["duration_ms"] == 800


def test_source_endpoint_serves_video_with_range_when_retained(evidence_client, auth_headers):
    with evidence_client() as (client, media):
        video = write_sample_video(media / "keep.mp4", frames=20)
        job = _upload(client, auth_headers, video, retain_source=True).json()
        assert job["source_available"] is True
        _wait_finished(client, job["job_id"], auth_headers)

        full = client.get(f"{JOBS}/{job['job_id']}/source", headers=auth_headers)
        assert full.status_code == 200
        assert full.headers["content-type"].startswith("video/mp4")

        partial = client.get(
            f"{JOBS}/{job['job_id']}/source",
            headers={**auth_headers, "Range": "bytes=0-99"},
        )
        assert partial.status_code == 206
        assert len(partial.content) == 100

        frame = client.get(f"{JOBS}/{job['job_id']}/frames/0", headers=auth_headers)
        assert frame.status_code == 200
        assert frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"

        overflow = client.get(f"{JOBS}/{job['job_id']}/frames/99999", headers=auth_headers)
        assert overflow.status_code == 404
        assert overflow.json()["code"] == "ANALYSIS_SOURCE_MISSING"


def test_source_is_absent_by_default_and_media_requires_token(evidence_client, auth_headers):
    with evidence_client() as (client, media):
        video = write_sample_video(media / "drop.mp4", frames=12)
        job = _upload(client, auth_headers, video).json()
        assert job["source_available"] is False
        _wait_finished(client, job["job_id"], auth_headers)

        missing = client.get(f"{JOBS}/{job['job_id']}/source", headers=auth_headers)
        assert missing.status_code == 404
        assert missing.json()["code"] == "ANALYSIS_SOURCE_MISSING"

        # <video>/<img> 不能带请求头，所以支持 query token；但匿名必须被拒绝。
        anonymous = client.get(f"{JOBS}/{job['job_id']}/frames/0")
        assert anonymous.status_code == 401
        with_query = client.get(f"{JOBS}/{job['job_id']}/frames/0?access_token={ADMIN_TOKEN}")
        assert with_query.status_code == 200
