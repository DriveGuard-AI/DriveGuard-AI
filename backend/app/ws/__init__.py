"""WebSocket 包。

只导出 Hub；路由模块由 `app.main` 直接导入，避免包初始化时产生循环依赖。
"""

from .hub import ConnectionManager

__all__ = ["ConnectionManager"]
