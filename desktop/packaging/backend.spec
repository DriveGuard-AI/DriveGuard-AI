# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec：把 FastAPI 后端与前端静态产物打成 onedir 目录。

构建命令（仓库根目录）：
    pyinstaller desktop/packaging/backend.spec --noconfirm \
        --distpath desktop/dist --workpath desktop/build/pyinstaller
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]  # desktop/packaging -> 仓库根

hiddenimports = (
    collect_submodules("app")
    + collect_submodules("uvicorn")
    + [
        "anyio._backends._asyncio",
        "multipart",
        "multipart.multipart",
        "pydantic.deprecated.decorator",
        "sqlalchemy.dialects.sqlite",
        "sqlalchemy.sql.default_comparator",
    ]
)

a = Analysis(
    [str(ROOT / "desktop" / "backend_entry.py")],
    pathex=[str(ROOT / "backend")],
    binaries=[],
    datas=[
        (str(ROOT / "frontend" / "dist"), "frontend/dist"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "torch", "torch_npu", "training", "inference"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="driveguard-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="backend",
)
