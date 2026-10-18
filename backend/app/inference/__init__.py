"""推理实现包。

当前仅提供脚本化占位实现；真实模型接入后新增实现并通过 `create_app` 注入即可。
"""

from .fake import FakeInference
from .port import InferencePort

__all__ = ["FakeInference", "InferencePort"]
