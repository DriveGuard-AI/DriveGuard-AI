"""运行期切换推理源（离线分析的前置能力）。

离线分析要能反复跑不同素材：换素材 = 换一个推理源 + 开一个新会话。
这里不依赖真实模型服务，用桩验证生命周期正确性。
"""

from __future__ import annotations

import asyncio

import pytest

from app.inference.port import AnalysisSource
from app.schemas import RiskSnapshot
from app.services.monitoring import InferenceEngine
from app.timeutil import utcnow

SOURCE_B = AnalysisSource(kind="frames", frame_sequence="runs/frames/clip-b")


class StubInference:
    """最小推理实现：永远 SAFE，不产生事件。"""

    mode = "fake"
    degraded = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def close(self) -> None:
        self.closed += 1

    def status(self) -> dict:
        return {"state": "running", "name": self.name}

    def next_snapshot(self, *, session_id: str, sequence: int, now) -> RiskSnapshot:
        return RiskSnapshot(
            session_id=session_id,
            sequence=sequence,
            captured_at=now,
            risk_level="SAFE",
            primary_risk=None,
            confidence=0.0,
            quality="good",
            reason_codes=["NO_BEHAVIOUR_DETECTED"],
        )


class StubHub:
    client_count = 0

    async def broadcast_snapshot(self, snapshot) -> None: ...

    async def broadcast_event_changed(self, change) -> None: ...

    async def broadcast_heartbeat(self, now) -> None: ...


@pytest.fixture()
def engine(database, monkeypatch):
    monkeypatch.setenv("DGB_INFERENCE_HZ", "200")
    monkeypatch.setenv("DGB_HEARTBEAT_SECONDS", "3600")
    from app.config import load_config

    built: list[AnalysisSource] = []

    def builder(source: AnalysisSource) -> StubInference:
        built.append(source)
        return StubInference(source.frame_sequence or "")

    created = InferenceEngine(
        database, load_config(), StubHub(), StubInference("initial"), builder
    )
    created.built_sources = built  # type: ignore[attr-defined]
    return created


def test_run_analysis_replaces_source_and_opens_a_new_session(engine):
    async def scenario():
        await engine.start()
        await asyncio.sleep(0.05)
        first_session = engine.session_id
        first_sequence = engine.status()["sequence"]
        assert first_sequence > 0

        second_session = await engine.run_analysis(SOURCE_B)
        await asyncio.sleep(0.05)

        # 每个素材一个新会话：事件按会话隔离
        assert second_session and second_session != first_session
        assert engine.session_id == second_session
        # 序号与延迟统计随新素材归零
        assert engine.status()["sequence"] < first_sequence + 100
        assert engine.built_sources == [SOURCE_B]
        await engine.stop()

    asyncio.run(scenario())


def test_previous_inference_is_closed_when_switching(engine):
    async def scenario():
        original = engine._inference
        await engine.start()
        await engine.run_analysis(SOURCE_B)
        assert original.closed == 1
        assert engine._inference is not original
        assert engine._inference.name == "runs/frames/clip-b"
        assert engine._inference.started == 1
        await engine.stop()

    asyncio.run(scenario())


def test_switching_without_a_builder_is_rejected(database, monkeypatch):
    monkeypatch.setenv("DGB_INFERENCE_HZ", "200")
    from app.config import load_config

    engine = InferenceEngine(database, load_config(), StubHub(), StubInference("initial"))

    async def scenario():
        with pytest.raises(RuntimeError):
            await engine.run_analysis(SOURCE_B)

    asyncio.run(scenario())
