"""live 推理端口测试。

不依赖真实模型：映射表逐条断言，泵帧用桩客户端，HTTP 线格式用回环假服务验证。
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from app.inference.frame_source import JpegSequenceSource
from app.inference.live_client import LiveCamera, LiveClient, LiveError
from app.inference.live_port import LiveInference, summarize_tasks

NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)
TASKS = ("fatigue", "phone_use", "drinking", "distraction")


def task(*, available: bool = True, detected: bool = False, score: float | None = 0.1) -> dict:
    return {
        "available": available,
        "logit": 0.0 if score is None else score,
        "score": score,
        "threshold_logit": 0.0,
        "detected": detected if available else None,
        "source": "test",
    }


def result_with(**overrides: dict) -> dict:
    tasks = {name: task() for name in TASKS}
    tasks.update(overrides)
    return {"tasks": tasks, "gaze": {"available": False, "validation_status": "unvalidated_body_view"}}


class OneFrameSource:
    def __init__(self, frames: int = 1) -> None:
        self._frames = [(index * 40, b"\xff\xd8stub") for index in range(frames)]
        self.closed = False

    def start(self) -> None: ...

    def next_frame(self):
        return self._frames.pop(0) if self._frames else None

    def close(self) -> None:
        self.closed = True


class StubClient:
    base_url = "http://stub"

    def __init__(self, result: dict | None, *, stale_session: str | None = None) -> None:
        self._result = result
        self.stale_session = stale_session
        self.sent: list[tuple[str, int]] = []
        self.closed: list[str] = []

    def status(self) -> dict:
        return {"status": "idle", "session_id": self.stale_session}

    def open_session(self, camera: LiveCamera) -> str:
        return "stub-session"

    def send_frame(self, session_id: str, timestamp_ms: int, jpeg: bytes) -> dict:
        self.sent.append((session_id, timestamp_ms))
        return {"status": "ready", "result": self._result, "feature_slots": len(self.sent)}

    def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# --------------------------------------------------------------- 映射规则


def test_unavailable_evidence_is_never_reported_as_safe():
    summary = summarize_tasks({name: task(available=False, detected=False, score=None) for name in TASKS})
    assert summary["risk_level"] == "WARNING"
    assert summary["quality"] == "unknown"
    assert summary["reason_codes"] == ["EVIDENCE_UNAVAILABLE"]
    assert summary["risk_scores"] == {}


def test_missing_tasks_behave_like_unavailable_evidence():
    summary = summarize_tasks({})
    assert summary["risk_level"] == "WARNING"
    assert summary["quality"] == "unknown"


def test_one_detection_is_warning_two_are_danger():
    single = summarize_tasks({**{name: task() for name in TASKS},
                              "phone_use": task(detected=True, score=0.93)})
    assert single["risk_level"] == "WARNING"
    assert single["primary_risk"] == "phone_use"
    assert single["confidence"] == pytest.approx(0.93)
    assert single["reason_codes"] == ["PHONE_USE_DETECTED"]
    assert single["quality"] == "good"

    double = summarize_tasks({**{name: task() for name in TASKS},
                              "phone_use": task(detected=True, score=0.93),
                              "drinking": task(detected=True, score=0.42)})
    assert double["risk_level"] == "DANGER"
    assert double["primary_risk"] == "phone_use"
    assert double["reason_codes"] == ["DRINKING_DETECTED", "PHONE_USE_DETECTED"]


def test_nothing_detected_is_safe_only_when_all_tasks_available():
    full = summarize_tasks({name: task() for name in TASKS})
    assert full["risk_level"] == "SAFE"
    assert full["quality"] == "good"
    assert full["reason_codes"] == ["NO_BEHAVIOUR_DETECTED"]

    partial = summarize_tasks({"fatigue": task(), "phone_use": task(available=False, score=None)})
    assert partial["risk_level"] == "SAFE"
    assert partial["quality"] == "degraded"


# --------------------------------------------------------------- 端口行为


def test_pending_evidence_before_the_first_result():
    inference = LiveInference(StubClient(None), LiveCamera(), OneFrameSource())
    snapshot = inference.next_snapshot(session_id="s-1", sequence=0, now=NOW)
    assert snapshot.inference_mode == "real"
    assert snapshot.risk_level == "WARNING"
    assert snapshot.quality == "unknown"
    assert snapshot.reason_codes == ["EVIDENCE_PENDING"]
    assert snapshot.features.ear is None  # 不伪造特征值
    assert snapshot.detections == []


def test_result_timeline_is_surfaced_and_absent_without_evidence():
    """离线事件按素材时间轴计时，依赖快照里的 source_timestamp_ms。"""

    inference = LiveInference(StubClient(None), LiveCamera(), OneFrameSource())
    assert inference.next_snapshot(session_id="s-1", sequence=0, now=NOW).source_timestamp_ms is None

    result = result_with(phone_use=task(detected=True, score=0.88))
    result["source_timestamp_ms"] = 12345
    client = StubClient(result)
    inference = LiveInference(client, LiveCamera(), OneFrameSource(frames=2))
    inference.start()
    try:
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
        assert wait_for(lambda: inference.status()["state"] == "completed")
        snapshot = inference.next_snapshot(session_id="s-1", sequence=7, now=NOW)
        assert snapshot.source_timestamp_ms == 12345
    finally:
        inference.close()


def test_pumped_result_becomes_a_real_snapshot():
    result = result_with(phone_use=task(detected=True, score=0.88))
    client = StubClient(result)
    inference = LiveInference(client, LiveCamera(), OneFrameSource(frames=2))
    inference.start()
    try:
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
        assert wait_for(lambda: inference.status()["state"] == "completed")
        snapshot = inference.next_snapshot(session_id="s-1", sequence=7, now=NOW)
        assert snapshot.sequence == 7
        assert snapshot.inference_mode == "real"
        assert snapshot.risk_level == "WARNING"
        assert snapshot.primary_risk == "phone_use"
        assert snapshot.reason_codes == ["PHONE_USE_DETECTED"]
        assert snapshot.risk_scores["phone_use"] == pytest.approx(0.88)
        status = inference.status()
        assert status["frames_sent"] == 2
        assert status["degraded"] is False
        assert status["tasks"]["phone_use"]["detected"] is True
    finally:
        inference.close()
    assert client.closed == ["stub-session"]


def test_stale_result_is_reported_as_unknown_not_safe():
    client = StubClient(result_with())
    inference = LiveInference(client, LiveCamera(), OneFrameSource(), stale_seconds=0.0)
    inference.start()
    try:
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
        time.sleep(0.05)  # Windows 的 monotonic 粒度约 15ms，先跨过一个 tick
        snapshot = inference.next_snapshot(session_id="s-1", sequence=0, now=NOW)
        assert snapshot.risk_level == "WARNING"
        assert snapshot.quality == "unknown"
        assert snapshot.reason_codes == ["RESULT_STALE"]
    finally:
        inference.close()


def test_unavailable_tasks_mark_the_port_degraded():
    client = StubClient(result_with(drinking=task(available=False, detected=False, score=None)))
    inference = LiveInference(client, LiveCamera(), OneFrameSource())
    inference.start()
    try:
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
        snapshot = inference.next_snapshot(session_id="s-1", sequence=0, now=NOW)
        assert snapshot.quality == "degraded"
        assert inference.status()["degraded"] is True
        assert "drinking" not in snapshot.risk_scores
    finally:
        inference.close()


def test_stale_session_is_released_before_opening():
    client = StubClient(result_with(), stale_session="leftover-session")
    inference = LiveInference(client, LiveCamera(), OneFrameSource())
    inference.start()
    try:
        assert wait_for(lambda: "leftover-session" in client.closed)
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
    finally:
        inference.close()


class FlakyClient(StubClient):
    def __init__(self, result: dict, failures: int) -> None:
        super().__init__(result)
        self._failures = failures
        self._sessions = 0

    def open_session(self, camera: LiveCamera) -> str:
        if self._failures > 0:
            self._failures -= 1
            raise LiveError("创建会话失败（409）: A camera session is already active")
        self._sessions += 1
        return f"session-{self._sessions}"


class DropOnceClient(StubClient):
    """第一个会话在第 2 帧失败，用于验证重开会话后时间戳重新从 0 开始。"""

    def __init__(self, result: dict) -> None:
        super().__init__(result)
        self._sessions = 0
        self._sent_in_session = 0
        self.per_session_timestamps: list[list[int]] = []

    def open_session(self, camera: LiveCamera) -> str:
        self._sessions += 1
        self._sent_in_session = 0
        self.per_session_timestamps.append([])
        return f"session-{self._sessions}"

    def send_frame(self, session_id: str, timestamp_ms: int, jpeg: bytes) -> dict:
        self._sent_in_session += 1
        if self._sessions == 1 and self._sent_in_session == 2:
            raise LiveError("POST /frames 返回 503: detector unavailable")
        self.per_session_timestamps[-1].append(timestamp_ms)
        return super().send_frame(session_id, timestamp_ms, jpeg)


def test_startup_keeps_retrying_while_the_session_is_busy():
    client = FlakyClient(result_with(), failures=2)
    inference = LiveInference(client, LiveCamera(), OneFrameSource(), retry_seconds=0.01)
    inference.start()
    try:
        assert wait_for(lambda: inference.status()["results_seen"] >= 1)
        assert inference.status()["state"] in {"streaming", "completed"}
    finally:
        inference.close()


def test_reopened_session_restarts_timestamps_at_zero():
    # live 服务在推理错误后会关闭会话，并要求新会话的时间戳从 0 开始。
    client = DropOnceClient(result_with())
    inference = LiveInference(client, LiveCamera(), OneFrameSource(frames=4), retry_seconds=0.01)
    inference.start()
    try:
        assert wait_for(lambda: len(client.per_session_timestamps) >= 2)
        assert wait_for(lambda: inference.status()["state"] == "completed")
    finally:
        inference.close()
    assert client.per_session_timestamps[0] == [0]
    assert client.per_session_timestamps[1][0] == 0
    assert client.per_session_timestamps[1] == [0, 40]


# --------------------------------------------------------------- 帧源


def test_jpeg_sequence_source_orders_files_and_uses_grid_timestamps(tmp_path):
    for name in ("b.jpg", "a.JPG", "c.jpeg", "notes.txt"):
        (tmp_path / name).write_bytes(b"jpeg-" + name.encode("ascii"))
    source = JpegSequenceSource(tmp_path, step_ms=40)
    source.start()
    assert source.total == 3
    frames = []
    while (frame := source.next_frame()) is not None:
        frames.append(frame)
    assert [timestamp for timestamp, _ in frames] == [0, 40, 80]
    assert [data for _, data in frames] == [b"jpeg-a.JPG", b"jpeg-b.jpg", b"jpeg-c.jpeg"]
    assert source.remaining == 0
    source.close()


def test_jpeg_sequence_source_rejects_empty_or_missing_directory(tmp_path):
    with pytest.raises(ValueError):
        JpegSequenceSource(tmp_path / "missing").start()
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "readme.txt").write_text("no jpeg", encoding="ascii")
    with pytest.raises(ValueError):
        JpegSequenceSource(empty).start()


# --------------------------------------------------------------- 线格式


class _WireHandler(BaseHTTPRequestHandler):
    received: dict = {}

    def log_message(self, *args):  # noqa: D401 - 静默
        pass

    def _reply(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).received.setdefault("authorization", self.headers.get("Authorization"))
        if self.path == "/api/live/sessions":
            type(self).received["camera"] = body
            self._reply(201, {"session_id": "wire-session", "status": "warming"})
            return
        if self.path.endswith("/frames"):
            type(self).received.setdefault("frames", []).append(body)
            self._reply(200, {"status": "ready", "result": None, "feature_slots": 1})
            return
        self._reply(404, {"error": "unknown endpoint"})

    def do_DELETE(self):  # noqa: N802
        type(self).received["deleted"] = self.path
        self._reply(200, {"status": "stopped"})

    def do_GET(self):  # noqa: N802
        self._reply(200, {"status": "ok"})


@pytest.fixture()
def wire_server():
    _WireHandler.received = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _WireHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_live_client_wire_format(wire_server, tmp_path):
    token_file = tmp_path / "api.token"
    token_file.write_text("t" * 40, encoding="ascii")
    client = LiveClient.from_token_file(wire_server, token_file)

    session = client.open_session(LiveCamera())
    reply = client.send_frame(session, 40, b"\xff\xd8jpeg-bytes")
    client.close_session(session)

    received = _WireHandler.received
    assert session == "wire-session"
    assert reply["status"] == "ready"
    assert received["authorization"] == "Bearer " + "t" * 40
    assert received["camera"]["image_size"] == [1280, 720]
    assert received["camera"]["driver_region_confirmed_exclusive"] is True
    assert received["frames"][0]["timestamp_ms"] == 40
    assert base64.b64decode(received["frames"][0]["jpeg"]) == b"\xff\xd8jpeg-bytes"
    assert received["deleted"] == "/api/live/sessions/wire-session"


def test_live_client_rejects_short_token(tmp_path):
    token_file = tmp_path / "short.token"
    token_file.write_text("short", encoding="ascii")
    with pytest.raises(ValueError):
        LiveClient.from_token_file("http://127.0.0.1:1", token_file)


def test_live_client_reports_unreachable_service():
    client = LiveClient("http://127.0.0.1:9", "t" * 40, timeout=0.5)
    with pytest.raises(LiveError):
        client.health()
