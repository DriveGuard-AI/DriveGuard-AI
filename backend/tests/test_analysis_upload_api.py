"""上传视频 → 抽帧 → 开始分析的接口测试（M2）。

用真实 mp4 走一遍解码与抽帧：这是"拖一个视频进来就能分析"的关键路径，
用假解码器测等于没测。推理侧仍是脚本化端口。
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.models import AuditLog
from app.services.frame_extraction import ExtractionError, extract_frames
from tests.analysis_support import (
    ADMIN_TOKEN,
    JOBS,
    configure_env,
    install_ports,
)

UPLOAD = f"{JOBS}/upload"


def _write_video(path: Path, *, frames: int = 10, fps: int = 25, width: int = 160, height: int = 120) -> Path:
    cv2, np = _cv2()
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():  # pragma: no cover - 取决于本机编码器
        pytest.skip("当前环境没有可用的 mp4 编码器")
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), (index * 20) % 255, dtype="uint8")
            writer.write(frame)
    finally:
        writer.release()
    return path


def _cv2():
    import cv2
    import numpy as np

    return cv2, np


def _upload(client: TestClient, headers, path: Path, **form):
    data = {"step_ms": "40", "size": "160x120", "quality": "85"}
    data.update({key: str(value) for key, value in form.items()})
    with path.open("rb") as handle:
        return client.post(
            UPLOAD,
            files={"file": (path.name, handle, "video/mp4")},
            data=data,
            headers=headers,
        )


def _wait_finished(client: TestClient, job_id: str, headers, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"{JOBS}/{job_id}", headers=headers).json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务 {job_id} 在 {timeout}s 内没有结束")


@pytest.fixture()
def upload_client(tmp_path, monkeypatch):
    """工厂夹具：必须在 `create_app()` 之前覆盖配置，否则读到的是旧值。"""

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


def test_upload_extracts_frames_and_runs(upload_client, auth_headers):
    with upload_client() as (client, media):
        video = _write_video(media / "clip.mp4", frames=10)

        response = _upload(client, auth_headers, video)
        assert response.status_code == 201, response.text
        job = response.json()

        assert job["source_kind"] == "frames"
        assert job["source_sha256"]
        assert job["frames_total"] >= 8  # 10 帧 / 25 fps / 40 ms ≈ 10
        job_dir = media / "analysis" / job["job_id"]
        frames = sorted((job_dir / "frames").glob("*.jpg"))
        assert len(frames) == job["frames_total"]
        assert frames[0].name == "000000.jpg"
        # retain_source 默认 false：抽帧后原始视频必须消失。
        assert not (job_dir / "source.mp4").exists()
        manifest = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        assert manifest["step_ms"] == 40
        assert manifest["size"] == "160x120"
        assert manifest["frames_total"] == job["frames_total"]

        finished = _wait_finished(client, job["job_id"], auth_headers)
        assert finished["status"] == "completed"
        assert finished["frames_done"] == finished["frames_total"]


def test_retain_source_keeps_video(upload_client, auth_headers):
    with upload_client(DGB_ANALYSIS_RETAIN_SOURCE="true") as (client, media):
        video = _write_video(media / "keep.mp4", frames=5)

        response = _upload(client, auth_headers, video)
        assert response.status_code == 201, response.text
        job_dir = media / "analysis" / response.json()["job_id"]
        assert (job_dir / "source.mp4").is_file()


def test_non_video_file_is_rejected(upload_client, auth_headers):
    with upload_client() as (client, media):
        text = media / "notes.txt"
        text.write_text("not a video", encoding="utf-8")

        response = client.post(
            UPLOAD,
            files={"file": ("notes.txt", text.read_bytes(), "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"


def test_broken_video_is_rejected_without_leftovers(upload_client, auth_headers):
    with upload_client() as (client, media):
        broken = media / "broken.mp4"
        broken.write_bytes(b"definitely not mpeg4")

        response = _upload(client, auth_headers, broken)
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"
        # 失败的上传不得留下半个任务目录。
        assert list((media / "analysis").iterdir()) == []


def test_upload_size_limit(upload_client, auth_headers):
    with upload_client(DGB_ANALYSIS_MAX_UPLOAD_MB="1") as (client, media):
        big = media / "big.mp4"
        big.write_bytes(b"\0" * (2 * 1024 * 1024))

        response = _upload(client, auth_headers, big)
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"
        assert list((media / "analysis").iterdir()) == []


def test_step_ms_must_stay_on_grid(upload_client, auth_headers):
    with upload_client() as (client, media):
        video = _write_video(media / "clip.mp4", frames=4)

        response = _upload(client, auth_headers, video, step_ms=30)
        assert response.status_code == 400
        assert "40" in response.text


def test_disk_budget_blocks_new_upload(upload_client, auth_headers):
    with upload_client(DGB_ANALYSIS_MAX_DISK_MB="1") as (client, media):
        (media / "blob.bin").write_bytes(b"\0" * (2 * 1024 * 1024))
        video = _write_video(media / "clip.mp4", frames=4)

        response = _upload(client, auth_headers, video)
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_ANALYSIS_SOURCE"


def test_purge_removes_only_expired_directories(upload_client):
    with upload_client() as (client, media):
        root = media / "analysis"
        stale = root / "stale-job"
        fresh = root / "fresh-job"
        stale.mkdir(parents=True)
        fresh.mkdir(parents=True)
        (stale / "000000.jpg").write_bytes(b"x")
        (fresh / "000000.jpg").write_bytes(b"x")
        old = time.time() - 48 * 3600
        os.utime(stale, (old, old))

        context = client.app.state.context
        removed = context.analysis._purge_once()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists()
        with context.db.session() as session:
            rows = (
                session.execute(
                    select(AuditLog).where(AuditLog.action == "analysis.purge")
                )
                .scalars()
                .all()
            )
            assert [row.resource_id for row in rows] == ["stale-job"]


def test_extract_frames_enforces_frame_cap(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=20)

    with pytest.raises(ExtractionError):
        extract_frames(
            video,
            tmp_path / "frames",
            step_ms=40,
            size="160x120",
            quality=85,
            max_frames=3,
        )
