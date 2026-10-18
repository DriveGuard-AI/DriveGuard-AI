"""脚本化回放的推理实现（占位，非真实模型）。

明确声明：
- `mode == "fake"`、`degraded == True`
- 输出会带上 `inference_mode="fake"`，`/system/status` 也会暴露降级状态
- 不得被解读为真实模型的行为结论（业务规则 1）
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..schemas import Detection, FeatureSet, HeadPose, RiskSnapshot
from .scenarios import Segment, Scenario, get_scenario


class FakeInference:
    """Scripted, deterministic risk-snapshot producer."""

    mode = "fake"

    def __init__(self, scenario: Scenario | None = None) -> None:
        self._scenario = scenario or get_scenario("default")
        self._started_at: datetime | None = None

    # -------------------------------------------------------------- 端口实现

    @property
    def degraded(self) -> bool:
        return True

    @property
    def scenario_name(self) -> str:
        return self._scenario.name

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "degraded": self.degraded,
            "scenario": self._scenario.name,
            "description": self._scenario.description,
            "loop_seconds": self._scenario.loop_seconds,
        }

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        if self._started_at is None:
            self._started_at = now
        elapsed = max(0.0, (now - self._started_at).total_seconds())
        segment, progress = self._scenario.segment_at(elapsed)

        risk_level = segment.risk_level
        confidence = segment.confidence
        risk_scores = dict(segment.risk_scores)
        reason_codes = list(segment.reason_codes)

        if segment.escalate and progress >= 0.35:
            risk_level = "DANGER"
            confidence = min(0.98, segment.confidence + 0.3 * progress)
            primary = segment.primary_risk or "other"
            risk_scores[primary] = confidence
            if "RISK_ESCALATED" not in reason_codes:
                reason_codes.append("RISK_ESCALATED")

        return RiskSnapshot(
            schema_version="1.0",
            session_id=session_id,
            sequence=sequence,
            captured_at=now,
            risk_level=risk_level,  # type: ignore[arg-type]
            primary_risk=segment.primary_risk,
            confidence=round(confidence, 4),
            quality=segment.quality,  # type: ignore[arg-type]
            risk_scores={key: round(value, 4) for key, value in risk_scores.items()},
            features=self._features(segment, progress),
            detections=self._detections(segment, progress),
            reason_codes=reason_codes,
            inference_mode="fake",
        )

    # -------------------------------------------------------------- 内部构造

    def _features(self, segment: Segment, progress: float) -> FeatureSet:
        if segment.quality != "good":
            # 面部不可见时不伪造正常值（FR-002）
            return FeatureSet(ear=None, mar=None, perclos=None, head_pose=None)

        if segment.name == "fatigue":
            ear = 0.28 - 0.16 * progress
            mar = 0.15 + 0.45 * progress
            perclos = 0.10 + 0.55 * progress
            pose = HeadPose(yaw=2.0, pitch=5.0 + 18.0 * progress, roll=1.0)
        elif segment.name == "phone_use":
            ear = 0.27
            mar = 0.18
            perclos = 0.12
            pose = HeadPose(yaw=6.0 + 10.0 * progress, pitch=18.0 + 16.0 * progress, roll=3.0)
        elif segment.name == "drinking":
            ear = 0.29
            mar = 0.24 + 0.2 * progress
            perclos = 0.08
            pose = HeadPose(yaw=3.0, pitch=12.0 + 8.0 * progress, roll=2.0)
        else:
            ear = 0.30
            mar = 0.12
            perclos = 0.05
            pose = HeadPose(yaw=1.0, pitch=1.5, roll=0.5)

        return FeatureSet(
            ear=round(ear, 4),
            mar=round(mar, 4),
            perclos=round(perclos, 4),
            head_pose=pose,
        )

    def _detections(self, segment: Segment, progress: float) -> list[Detection]:
        if segment.quality != "good":
            return []
        detections: list[Detection] = []
        for index, (label, base) in enumerate(segment.detections):
            confidence = min(0.99, base + 0.12 * progress)
            detections.append(
                Detection(
                    label=label,
                    confidence=round(confidence, 4),
                    bbox=[0.42, 0.35 + 0.05 * index, 0.78, 0.92],
                )
            )
        return detections


__all__ = ["FakeInference"]
