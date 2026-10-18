"""WS-001 实时监控通道。"""

from __future__ import annotations

import pytest
from starlette.websockets import WebSocketDisconnect


def test_websocket_requires_valid_token(client):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/v1/monitoring") as websocket:
            websocket.receive_json()


def test_websocket_sends_ready_then_snapshots(client):
    with client.websocket_connect("/ws/v1/monitoring?access_token=test-admin-token") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "connection.ready"
        assert ready["schema_version"] == "1.0"
        assert "heartbeat_seconds" in ready["data"]

        first = websocket.receive_json()
        assert first["type"] in {"risk.snapshot", "heartbeat", "system.status"}

        snapshot = None
        for _ in range(20):
            message = websocket.receive_json()
            if message["type"] == "risk.snapshot":
                snapshot = message
                break
        assert snapshot is not None, "应收到 risk.snapshot"
        assert snapshot["data"]["inference_mode"] == "fake"
        assert snapshot["data"]["risk_level"] in {"SAFE", "WARNING", "DANGER"}
        assert isinstance(snapshot["sequence"], int)

        # 序列号必须单调递增
        second = None
        for _ in range(20):
            message = websocket.receive_json()
            if message["type"] == "risk.snapshot":
                second = message
                break
        assert second is not None
        assert second["sequence"] >= snapshot["sequence"]
        assert second["sent_at"].endswith("Z")


def test_websocket_token_in_subprotocol_is_accepted(client):
    with client.websocket_connect(
        "/ws/v1/monitoring",
        subprotocols=["test-admin-token"],
    ) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "connection.ready"


def test_envelope_sequence_is_strictly_monotonic_across_message_types(client):
    """REQ-007：同一连接上的消息序列号必须单调递增（不得回退）。"""

    with client.websocket_connect("/ws/v1/monitoring?access_token=test-admin-token") as websocket:
        sequences = [websocket.receive_json()["sequence"] for _ in range(25)]

    assert sequences == sorted(sequences), f"序列号出现回退：{sequences}"
    assert len(set(sequences)) == len(sequences), f"序列号出现重复：{sequences}"


def test_snapshot_payload_keeps_its_own_sequence(client):
    """快照自身序号保留在 data.sequence，供前端展示与丢帧判断。"""

    with client.websocket_connect("/ws/v1/monitoring?access_token=test-admin-token") as websocket:
        for _ in range(25):
            message = websocket.receive_json()
            if message["type"] == "risk.snapshot":
                assert isinstance(message["data"]["sequence"], int)
                assert "sequence" in message["data"]
                return
    raise AssertionError("未收到 risk.snapshot")
