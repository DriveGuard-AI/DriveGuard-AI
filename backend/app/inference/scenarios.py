"""脚本化场景定义。

无模型阶段使用可复现的确定性脚本，便于稳定复现 `TEST_STRATEGY.md §5` 的 E2E 场景，
尤其是 E2E-005（面部遮挡 → degraded，不得解释为 SAFE）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    name: str
    duration: float
    risk_level: str
    primary_risk: str | None
    quality: str
    confidence: float
    risk_scores: dict[str, float] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)
    detections: list[tuple[str, float]] = field(default_factory=list)
    escalate: bool = False
    """为真时，在后半段把风险升级为 DANGER 并提高置信度。"""


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    segments: tuple[Segment, ...]

    @property
    def loop_seconds(self) -> float:
        return sum(segment.duration for segment in self.segments)

    def segment_at(self, elapsed_seconds: float) -> tuple[Segment, float]:
        """返回当前片段及其内部进度 `p ∈ [0, 1)`。"""

        if self.loop_seconds <= 0:
            return self.segments[0], 0.0
        offset = elapsed_seconds % self.loop_seconds
        for segment in self.segments:
            if offset < segment.duration:
                return segment, offset / segment.duration if segment.duration else 0.0
            offset -= segment.duration
        last = self.segments[-1]
        return last, 1.0


SAFE_SEGMENT = Segment(
    name="safe",
    duration=6.0,
    risk_level="SAFE",
    primary_risk=None,
    quality="good",
    confidence=0.05,
    risk_scores={"fatigue": 0.05, "phone_use": 0.02, "drinking": 0.01, "distraction": 0.03},
    reason_codes=["NO_RISK"],
)

FATIGUE_SEGMENT = Segment(
    name="fatigue",
    duration=8.0,
    risk_level="WARNING",
    primary_risk="fatigue",
    quality="good",
    confidence=0.62,
    risk_scores={"fatigue": 0.62, "phone_use": 0.05, "drinking": 0.02, "distraction": 0.12},
    reason_codes=["EYES_CLOSED_SUSTAINED"],
    escalate=True,
)

PHONE_SEGMENT = Segment(
    name="phone_use",
    duration=8.0,
    risk_level="WARNING",
    primary_risk="phone_use",
    quality="good",
    confidence=0.58,
    risk_scores={"fatigue": 0.08, "phone_use": 0.58, "drinking": 0.02, "distraction": 0.44},
    reason_codes=["PHONE_INTERACTION_SUSTAINED"],
    detections=[("phone", 0.81)],
    escalate=True,
)

DRINKING_SEGMENT = Segment(
    name="drinking",
    duration=6.0,
    risk_level="WARNING",
    primary_risk="drinking",
    quality="good",
    confidence=0.55,
    risk_scores={"fatigue": 0.06, "phone_use": 0.04, "drinking": 0.55, "distraction": 0.3},
    reason_codes=["DRINKING_ACTION"],
    detections=[("bottle", 0.74)],
)

DEGRADED_SEGMENT = Segment(
    name="degraded",
    duration=6.0,
    # 业务规则 2：输入质量不足时不得默认输出 SAFE
    risk_level="WARNING",
    primary_risk=None,
    quality="degraded",
    confidence=0.0,
    risk_scores={},
    reason_codes=["INPUT_DEGRADED"],
)

SCENARIOS: dict[str, Scenario] = {
    "default": Scenario(
        name="default",
        description="完整演示序列：安全 → 疲劳 → 安全 → 手机 → 饮水 → 输入降级",
        segments=(
            SAFE_SEGMENT,
            FATIGUE_SEGMENT,
            SAFE_SEGMENT,
            PHONE_SEGMENT,
            DRINKING_SEGMENT,
            DEGRADED_SEGMENT,
        ),
    ),
    "safe": Scenario(
        name="safe",
        description="仅安全状态，用于验证不产生误报",
        segments=(SAFE_SEGMENT,),
    ),
    "fatigue": Scenario(
        name="fatigue",
        description="安全后进入持续疲劳，用于验证告警与事件聚合",
        segments=(SAFE_SEGMENT, FATIGUE_SEGMENT),
    ),
    "degraded": Scenario(
        name="degraded",
        description="安全后输入降级，用于验证不得把缺失解释为 SAFE",
        segments=(SAFE_SEGMENT, DEGRADED_SEGMENT),
    ),
}


def get_scenario(name: str) -> Scenario:
    return SCENARIOS.get(name, SCENARIOS["default"])


__all__ = ["SCENARIOS", "Scenario", "Segment", "get_scenario"]
