"""SQLite 引擎与会话管理。

遵循 `docs/database/DATABASE_DESIGN.md` 第 2 节：UTC 时间、WAL、短事务、
稳定 UUID 主键，且推理热路径不直接写库（写入经由 repository 层）。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def build_engine(database_url: str) -> Engine:
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(database_url, future=True, connect_args=connect_args)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _record) -> None:  # pragma: no cover - 驱动回调
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, future=True, expire_on_commit=False, autoflush=False)


class Database:
    """引擎 + 会话工厂的轻量封装。"""

    def __init__(self, database_url: str) -> None:
        self.engine = build_engine(database_url)
        self.session_factory = build_session_factory(self.engine)

    def create_all(self) -> None:
        # 导入以注册 ORM 元数据
        from . import models  # noqa: F401

        self._ensure_sqlite_directory()
        Base.metadata.create_all(self.engine)

    def _ensure_sqlite_directory(self) -> None:
        url = self.engine.url
        if url.get_backend_name() != "sqlite":
            return
        database = url.database
        if not database or database == ":memory:":
            return
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    def dispose(self) -> None:
        self.engine.dispose()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """短事务上下文：显式提交/回滚，避免事务悬挂。"""

        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def is_writable(self) -> bool:
        from sqlalchemy import text

        try:
            with self.session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
