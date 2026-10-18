"""DriveGuard AI 后端应用包。

本包实现 `docs/requirements/REQUIREMENTS.md`、`docs/api/API_DESIGN.md` 与
`docs/database/DATABASE_DESIGN.md` 定义的应用层与数据层契约。

注意：模型识别与判断暂未实现，推理由 `app.inference.fake` 的脚本化回放提供，
并且会在 `RiskSnapshot.inference_mode` 与 `/system/status` 中显式声明为 `fake`，
不得冒充真实模型输出。
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
