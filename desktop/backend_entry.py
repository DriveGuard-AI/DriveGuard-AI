"""DriveGuard AI 桌面包的后端入口。

PyInstaller 冻结后 `backend/app` 与 `frontend/dist` 都在 `_MEIPASS` 下；
Electron 会显式传入静态目录、数据库目录与模型服务配置，这里只做兜底。
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _resource_root()
    # Electron 会覆盖这些值；直接运行本文件时也能起一个本地演示实例。
    os.environ.setdefault("DGB_STATIC_DIR", str(root / "frontend" / "dist"))
    os.environ.setdefault("DGB_AUTO_CREATE_SCHEMA", "true")
    os.environ.setdefault("DGB_INFERENCE_MODE", "fake")

    port = int(os.environ.get("DGB_DESKTOP_PORT", "8000"))
    if not (1024 <= port <= 65535):
        raise SystemExit(f"DGB_DESKTOP_PORT 非法: {port}")

    from app.main import app  # noqa: PLC0415 - 保证 env 先于应用导入

    import uvicorn  # noqa: PLC0415

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level=os.environ.get("DGB_DESKTOP_LOG_LEVEL", "info"),
        access_log=False,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
