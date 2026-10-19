"""离线分析测试的共用替身与装配。

真实模型不参与：把 `app.main.build_inference` 换成脚本化端口，
让"帧数跑完"这个终态在毫秒级可控地发生。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.schemas import RiskSnapshot

ADMIN_TOKEN = "test-admin-token"
JPEG = b"\xff\xd8\xff\xd9"
JOBS = "/api/v1/analysis/jobs"


def snapshot(
    session_id: str, sequence: int, now: datetime, position: int | None
) -> RiskSnapshot:
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


class IdlePort:
    """实时阶段的占位推理：一直在跑，不产生终态。"""

    mode = "fake"
    degraded = False

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def status(self) -> dict:
        return {"state": "running", "frames_sent": 0}

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        return snapshot(session_id, sequence, now, None)


class ScriptedPort:
    """离线推理桩：tick 到 `frames` 次之后自报 completed。"""

    mode = "fake"
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
        return snapshot(session_id, sequence, now, sequence * 40)


def make_frames(media: Path, count: int, *, name: str = "job-under-test") -> Path:
    directory = media / "analysis" / name / "frames"
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (directory / f"{index:06d}.jpg").write_bytes(JPEG)
    return directory


def configure_env(monkeypatch, tmp_path, media: Path, **overrides: object) -> None:
    env: dict[str, object] = {
        "DGB_DATABASE_URL": f"sqlite:///{(tmp_path / 'analysis.db').as_posix()}",
        "DGB_ADMIN_TOKEN": ADMIN_TOKEN,
        "DGB_INFERENCE_HZ": "50",
        "DGB_HEARTBEAT_SECONDS": "30",
        "DGB_INFERENCE_SCENARIO": "default",
        "DGB_STATIC_DIR": str(tmp_path / "dist"),
        "DGB_MEDIA_DIR": str(media),
    }
    env.update(overrides)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))


def install_ports(monkeypatch, script: dict) -> None:
    """把推理构建器换成脚本化端口。

    `script["frames"]` 由用例在请求前设置；设为 None 表示"按素材目录里的 JPEG 数跑"，
    上传用例只有抽帧完成后才知道帧数，用这个模式才能断言跑完。
    """

    import app.main as main_module

    def factory(config):
        if config.inference_mode != "live":
            return IdlePort()
        frames = script["frames"]
        if frames is None:
            frames = len(list(Path(config.frame_sequence_dir).glob("*.jpg")))
        return ScriptedPort(frames)

    monkeypatch.setattr(main_module, "build_inference", factory)


__all__ = [
    "ADMIN_TOKEN",
    "IdlePort",
    "JPEG",
    "JOBS",
    "ScriptedPort",
    "configure_env",
    "install_ports",
    "make_frames",
    "snapshot",
]
