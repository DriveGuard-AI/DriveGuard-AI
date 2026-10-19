"""鉴权与授权。

`docs/api/API_DESIGN.md` 第 6 节：
- `/health` 匿名可用
- 其余接口在生产部署使用 Bearer Token
- 权限：`monitoring:read`、`settings:read`、`settings:write`、`models:read`

本实现保持轻量：配置了 `DGB_ADMIN_TOKEN` 时，携带正确 Token 的请求获得全部权限；
未配置 Token 时视为本地可信只读演示模式，匿名仅获得读权限，管理写操作仍返回 401。
前端隐藏按钮不构成授权（NFR-004 / 架构 §8）。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request, WebSocket

from .config import AppConfig
from .errors import AuthRequiredError, ForbiddenError

PERMISSION_MONITORING_READ = "monitoring:read"
PERMISSION_MONITORING_WRITE = "monitoring:write"
PERMISSION_SETTINGS_READ = "settings:read"
PERMISSION_SETTINGS_WRITE = "settings:write"
PERMISSION_MODELS_READ = "models:read"
PERMISSION_ANALYSIS_READ = "analysis:read"
PERMISSION_ANALYSIS_WRITE = "analysis:write"

ANONYMOUS_PERMISSIONS = frozenset({PERMISSION_MONITORING_READ, PERMISSION_MODELS_READ})
ADMIN_PERMISSIONS = frozenset(
    {
        PERMISSION_MONITORING_READ,
        PERMISSION_MONITORING_WRITE,
        PERMISSION_SETTINGS_READ,
        PERMISSION_SETTINGS_WRITE,
        PERMISSION_MODELS_READ,
        PERMISSION_ANALYSIS_READ,
        PERMISSION_ANALYSIS_WRITE,
    }
)


@dataclass(frozen=True)
class Principal:
    subject: str
    authenticated: bool
    permissions: frozenset[str]


def _extract_bearer(header_value: str | None) -> str | None:
    if not header_value:
        return None
    parts = header_value.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def resolve_principal(config: AppConfig, token: str | None) -> Principal:
    if config.admin_token and token and token == config.admin_token:
        return Principal(subject="admin", authenticated=True, permissions=ADMIN_PERMISSIONS)
    if config.auth_enabled:
        # 已配置 Token：除健康检查外一律要求鉴权（NFR-004 生产部署必须启用鉴权）。
        # 提供错误凭证同样视为未认证，避免静默降级。
        return Principal(subject="anonymous", authenticated=False, permissions=frozenset())
    # 本地可信只读演示模式：匿名仅获得读权限
    return Principal(subject="anonymous", authenticated=False, permissions=ANONYMOUS_PERMISSIONS)


def principal_from_request(config: AppConfig, request: Request) -> Principal:
    token = _extract_bearer(request.headers.get("authorization"))
    return resolve_principal(config, token)


def principal_from_websocket(config: AppConfig, websocket: WebSocket) -> Principal:
    token = websocket.query_params.get("access_token")
    if not token:
        # 优先使用短期凭证子协议，避免 Token 进入 URL 日志
        protocol = websocket.headers.get("sec-websocket-protocol")
        token = protocol.strip() if protocol else None
    return resolve_principal(config, token)


def ensure_permission(principal: Principal, permission: str) -> None:
    if permission in principal.permissions:
        return
    if not principal.authenticated:
        raise AuthRequiredError()
    raise ForbiddenError()


__all__ = [
    "ADMIN_PERMISSIONS",
    "ANONYMOUS_PERMISSIONS",
    "PERMISSION_ANALYSIS_READ",
    "PERMISSION_ANALYSIS_WRITE",
    "PERMISSION_MODELS_READ",
    "PERMISSION_MONITORING_READ",
    "PERMISSION_MONITORING_WRITE",
    "PERMISSION_SETTINGS_READ",
    "PERMISSION_SETTINGS_WRITE",
    "Principal",
    "ensure_permission",
    "principal_from_request",
    "principal_from_websocket",
    "resolve_principal",
]
