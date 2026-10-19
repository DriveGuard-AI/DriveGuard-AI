"""推理端口。

前后端只依赖 `RiskSnapshot` 契约；`InferencePort` 是模型实现与业务逻辑之间的唯一边界。
将来接入 TCN / Risk Fusion 时，只需提供新的 `InferencePort` 实现，
`MonitoringService`、REST 与前端不需要任何改动（见 `docs/architecture/ARCHITECTURE.md` §5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from ..schemas import RiskSnapshot


@dataclass(frozen=True)
class AnalysisSource:
    """一次分析的素材来源描述。

    引擎只认这个描述，不认具体采集方式，这样三类来源可以共存而不改动引擎：

    - `frames`：预抽好的 JPEG 序列目录（离线分析，服务端拉帧）；
    - `camera`：设备或网络摄像头（实时，服务端拉流）；
    - `push`：浏览器推流（实时，帧由 HTTP 打进来）。

    目前只实现 `frames`。新增来源时提供新的 `InferencePort` 实现即可，
    `InferenceEngine` 的接口不变。
    """

    kind: str
    frame_sequence: str | None = None


class InferencePort(Protocol):
    """风险推理端口。

    实现可以额外提供两个**可选**生命周期方法，`InferenceEngine` 会按需调用：

    - `start()`：启动后台采集（例如从 live 服务拉帧）；
    - `close()`：释放会话与线程。
    """

    @property
    def mode(self) -> str:
        """`real` 或 `fake`；必须在对外状态中如实暴露。"""

    @property
    def degraded(self) -> bool:
        """当前是否处于降级实现（例如脚本化回放）。"""

    def status(self) -> dict[str, Any]:
        """运行状态摘要，用于 `/api/v1/system/status`。"""

    def next_snapshot(self, *, session_id: str, sequence: int, now: datetime) -> RiskSnapshot:
        """产出下一个版本化风险快照。"""


__all__ = ["InferencePort"]
