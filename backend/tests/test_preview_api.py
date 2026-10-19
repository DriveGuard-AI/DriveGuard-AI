"""实时预览（API-010）与摄像头帧源测试。

关注点是**对齐**而不是画面好不好看：
- 预览分片里的字节/摘要必须等于送进推理的那一帧；
- 摄像头时间戳必须落在 25 Hz 网格上，慢于网格时丢时间片而不是积压；
- 帧源故障不能被当成"素材读完"，否则监控会静默停在最后一帧。
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from app.inference.camera_source import CameraSource
from app.inference.frame_source import FrameSourceError, PreviewTap
from app.inference.live_client import LiveCamera
from app.inference.live_port import LiveInference
from tests.test_live_inference import StubClient, result_with

JPEG = b"\xff\xd8\xff\xe0-preview-stand-in-\xff\xd9"
JPEG_SHA = hashlib.sha256(JPEG).hexdigest()


class ListSource:
    def __init__(self, frames: list[tuple[int, bytes]]) -> None:
        self._frames = list(frames)
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def next_frame(self):
        return self._frames.pop(0) if self._frames else None

    def close(self) -> None:
        self.closed = True


class RaisingSource:
    def start(self) -> None: ...

    def next_frame(self):
        raise FrameSourceError("摄像头读帧失败: device unplugged")

    def close(self) -> None: ...


# --------------------------------------------------------------- PreviewTap


def test_preview_tap_records_the_exact_bytes_sent_to_inference():
    source = ListSource([(0, JPEG), (40, JPEG + b"2")])
    tap = PreviewTap(source)

    assert tap.start() is None
    assert tap.next_frame() == (0, JPEG)
    frame = tap.latest()
    assert frame is not None
    assert frame.sequence == 0
    assert frame.timestamp_ms == 0
    # 同一个 bytes 对象：预览不是重新编码的副本
    assert frame.jpeg is JPEG
    assert frame.sha256 == JPEG_SHA
    assert frame.size_bytes == len(JPEG)

    assert tap.next_frame() == (40, JPEG + b"2")
    assert tap.latest().sequence == 1
    assert tap.stats()["frames_out"] == 2
    assert tap.stats()["source_timestamp_ms"] == 40


def test_preview_tap_surfaces_source_failures_without_inventing_a_frame():
    tap = PreviewTap(RaisingSource())
    with pytest.raises(FrameSourceError):
        tap.next_frame()
    assert tap.latest() is None
    assert "device unplugged" in str(tap.stats()["last_read_error"])


def test_preview_tap_reset_drops_previous_source_frames():
    tap = PreviewTap(ListSource([(0, JPEG)]))
    tap.next_frame()
    assert tap.latest() is not None
    tap.reset()
    assert tap.latest() is None
    assert tap.stats()["frames_out"] == 0


# --------------------------------------------------------------- CameraSource


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeCapture:
    def __init__(self, *, fail_after: int | None = None) -> None:
        self.reads = 0
        self.grabs = 0
        self.released = False
        self._fail_after = fail_after

    def isOpened(self) -> bool:  # noqa: N802 - 模拟 opencv 接口
        return True

    def set(self, prop, value) -> bool:
        return True

    def grab(self) -> bool:
        self.grabs += 1
        return True

    def read(self):
        self.reads += 1
        if self._fail_after is not None and self.reads > self._fail_after:
            return False, None
        # 8x8 纯色帧足够走通编码路径，不需要真实摄像头
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_camera_source_uses_a_strict_40ms_grid():
    clock = FakeClock()
    capture = FakeCapture()
    source = CameraSource(
        0,
        step_ms=40,
        capture_factory=lambda: capture,
        clock=clock,
        sleep=clock.sleep,
    )
    source.start()
    try:
        first_ts, first_jpeg = source.next_frame()
        assert first_ts == 0
        assert first_jpeg.startswith(b"\xff\xd8")

        clock.sleeps.clear()
        second_ts, _ = source.next_frame()
        assert second_ts == 40
        # 相机再快也不能超过网格：必须补眠到 +40ms
        assert clock.sleeps and clock.sleeps[0] == pytest.approx(0.04)

        third_ts, _ = source.next_frame()
        assert third_ts == 80
    finally:
        source.close()
    assert capture.released is True


def test_camera_source_skips_time_slices_when_running_behind():
    clock = FakeClock()
    capture = FakeCapture()
    source = CameraSource(
        0,
        step_ms=40,
        capture_factory=lambda: capture,
        clock=clock,
        sleep=clock.sleep,
    )
    source.start()
    try:
        assert source.next_frame()[0] == 0
        # 模拟处理抖动：墙钟已经越过两个时间片
        clock.now += 0.1
        clock.sleeps.clear()
        timestamp_ms, _ = source.next_frame()
        assert timestamp_ms == 40
        # 落后时不补眠，时间戳继续按网格走（不积压旧帧）
        assert clock.sleeps == []
        assert capture.grabs >= 1
    finally:
        source.close()


def test_camera_source_reports_failure_instead_of_completing():
    def factory() -> FakeCapture:
        return FakeCapture(fail_after=0)  # 预热就读不到帧

    source = CameraSource(
        0,
        capture_factory=factory,
        clock=FakeClock(),
        sleep=lambda seconds: None,
        read_retries=0,
        reopen_attempts=1,
        reopen_delay_seconds=0.0,
    )
    with pytest.raises(FrameSourceError):
        source.start()


def test_camera_read_failure_is_not_treated_as_source_completion():
    """拔线不能让监控"正常结束"：必须是可重试故障，最终 failed 且带错误原因。"""

    inference = LiveInference(
        StubClient(result_with()),
        LiveCamera(),
        RaisingSource(),
        retry_seconds=0.01,
        max_failures=1,
    )
    inference.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and inference.status()["state"] != "failed":
            time.sleep(0.02)
        status = inference.status()
        assert status["state"] == "failed"
        assert "FrameSourceError" in (status["last_error"] or "")
    finally:
        inference.close()


# --------------------------------------------------------------- API


def _enable_preview(client, headers, **extra) -> None:
    version = client.get("/api/v1/settings", headers=headers).json()["version"]
    payload = {"version": version, "changes": {"preview.enabled": True, **extra}}
    response = client.put("/api/v1/settings", headers=headers, json=payload)
    assert response.status_code == 200, response.text


def _attach_frame(context, jpeg: bytes = JPEG) -> PreviewTap:
    tap = PreviewTap(ListSource([(0, jpeg)]))
    tap.start()
    tap.next_frame()
    context.engine.set_preview_tap(tap)
    return tap


def test_preview_is_closed_by_default(client, auth_headers):
    response = client.get("/api/v1/monitoring/preview", headers=auth_headers)
    assert response.status_code == 403
    body = response.json()
    assert body["code"] == "FORBIDDEN"
    # 与"没有权限"区分开：管理员只是没打开开关
    assert body["details"][0]["reason"] == "preview.disabled"

    state = client.get("/api/v1/monitoring/preview/state", headers=auth_headers).json()
    assert state["enabled"] is False
    assert state["available"] is False
    assert state["pipeline"] == "realtime"


def test_preview_requires_a_frame_source(client, auth_headers):
    _enable_preview(client, auth_headers)
    # fake 推理没有帧源
    response = client.get("/api/v1/monitoring/preview", headers=auth_headers)
    assert response.status_code == 503
    assert response.json()["code"] == "PIPELINE_UNAVAILABLE"


def test_preview_requires_an_active_session(client, auth_headers, app_context):
    _enable_preview(client, auth_headers)
    _attach_frame(app_context)
    app_context.engine._session_id = None
    response = client.get("/api/v1/monitoring/preview", headers=auth_headers)
    assert response.status_code == 409
    assert response.json()["code"] == "MONITORING_NOT_ACTIVE"


def test_preview_rejects_when_client_limit_is_reached(client, auth_headers, app_context):
    _enable_preview(client, auth_headers, **{"preview.max_clients": 1})
    _attach_frame(app_context)
    app_context.preview._clients = 1
    response = client.get("/api/v1/monitoring/preview", headers=auth_headers)
    assert response.status_code == 503
    assert "上限" in response.json()["message"]


def test_preview_state_reports_the_frame_that_was_inferred(client, auth_headers, app_context):
    _enable_preview(client, auth_headers)
    _attach_frame(app_context)

    state = client.get("/api/v1/monitoring/preview/state", headers=auth_headers).json()
    assert state["enabled"] is True
    assert state["available"] is True
    assert state["sequence"] == 0
    assert state["source_timestamp_ms"] == 0
    assert state["sha256"] == JPEG_SHA
    assert state["bytes"] == len(JPEG)
    assert state["frames_out"] == 1


def test_preview_stream_parts_carry_the_inferred_frame_and_hash(client, auth_headers, app_context):
    _enable_preview(client, auth_headers)
    _attach_frame(app_context)

    service = app_context.preview
    real_stream = service.stream
    # 只为测试收束流：真实连接由客户端断开或空闲超时结束
    service.stream = lambda **kwargs: real_stream(max_parts=1, **kwargs)

    with client.stream("GET", "/api/v1/monitoring/preview", headers=auth_headers) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert response.headers["content-type"].endswith("boundary=frame")
        assert response.headers["cache-control"] == "no-store"
        body = b"".join(response.iter_bytes())

    assert body.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in body
    assert f"Content-Length: {len(JPEG)}\r\n".encode() in body
    assert b"X-Seq: 0\r\n" in body
    assert b"X-Timestamp-Ms: 0\r\n" in body
    assert f"X-Sha256: {JPEG_SHA}\r\n".encode() in body
    # 分片里的字节就是送进推理的那一份，逐字节相同
    assert JPEG in body
    assert body.endswith(b"--frame--\r\n")


def test_preview_turns_off_at_runtime_without_restart(client, auth_headers, app_context):
    _enable_preview(client, auth_headers)
    _attach_frame(app_context)
    assert client.get("/api/v1/monitoring/preview/state", headers=auth_headers).json()["enabled"] is True

    version = client.get("/api/v1/settings", headers=auth_headers).json()["version"]
    response = client.put(
        "/api/v1/settings",
        headers=auth_headers,
        json={"version": version, "changes": {"preview.enabled": False}},
    )
    assert response.status_code == 200
    assert client.get("/api/v1/monitoring/preview", headers=auth_headers).status_code == 403
