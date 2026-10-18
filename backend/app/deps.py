"""依赖注入与应用上下文。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from .config import AppConfig
from .database import Database
from .security import Principal, ensure_permission, principal_from_request
from .services.events import EventService
from .services.monitoring import InferenceEngine
from .services.settings import SettingsService
from .services.statistics import StatisticsService
from .services.system import SystemService
from .ws.hub import ConnectionManager


@dataclass
class AppContext:
    """应用级单例集合：由 `create_app` 构造并挂到 `app.state.context`。"""

    config: AppConfig
    db: Database
    hub: ConnectionManager
    engine: InferenceEngine
    events: EventService
    settings: SettingsService
    statistics: StatisticsService
    system: SystemService


def get_context(request: Request) -> AppContext:
    return request.app.state.context


def get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def require_permission(permission: str):
    """返回校验指定权限的依赖；失败时抛出统一错误。"""

    def dependency(request: Request, context: AppContext = Depends(get_context)) -> Principal:
        principal = principal_from_request(context.config, request)
        ensure_permission(principal, permission)
        return principal

    return dependency


__all__ = ["AppContext", "get_context", "get_request_id", "require_permission"]
