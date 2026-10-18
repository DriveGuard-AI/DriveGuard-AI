"""Alembic 环境配置。

`docs/database/DATABASE_DESIGN.md` 第 9 节要求 schema 变更使用单调递增 migration，
且不在数据库上手工修改。
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database import Base  # noqa: E402
from app import models  # noqa: F401,E402  (注册元数据)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.environ.get("DGB_DATABASE_URL")
if database_url:
    config.set_main_option("sqlalchemy.url", database_url)

target_metadata = Base.metadata


def _ensure_sqlite_directory() -> None:
    """SQLite 需要父目录存在；迁移前自动创建，避免 `unable to open database file`。"""

    url = config.get_main_option("sqlalchemy.url") or ""
    if not url.startswith("sqlite"):
        return
    path = url.split("sqlite:///", 1)[-1]
    if not path or path == ":memory:":
        return
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = BACKEND_DIR / candidate
    candidate.parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    _ensure_sqlite_directory()
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
