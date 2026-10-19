"""应用入口。

`create_app` 组装配置、数据层、服务层、REST 路由、WebSocket 路由与静态前端。
默认单进程模块化单体（`docs/architecture/ARCHITECTURE.md` ADR-002），
静态前端由 FastAPI 托管（前端执行 `npm run build` 后产物位于 `frontend/dist`）。
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from .api.v1 import api_v1_router
from .config import AppConfig, load_config
from .database import Database
from .deps import AppContext
from .errors import register_exception_handlers
from .inference.factory import build_inference
from .inference.port import InferencePort
from .services.analysis import AnalysisService
from .services.events import EventService
from .services.monitoring import InferenceEngine
from .services.settings import SettingsService
from .services.statistics import StatisticsService
from .services.system import SystemService
from .ws.hub import ConnectionManager
from .ws.routes import router as ws_router


class RequestIDMiddleware(BaseHTTPMiddleware):
    """为每个请求分配/回传 `X-Request-ID`，用于日志与错误关联。"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def build_context(
    config: AppConfig,
    inference: InferencePort | None = None,
) -> AppContext:
    db = Database(config.database_url)
    hub = ConnectionManager(config)
    inference_impl = inference or build_inference(config)
    # 离线分析需要在运行期切换素材；测试注入固定推理时不给构建器。
    # 入参是素材来源描述，新增摄像头/浏览器推流时只需在这里多一个分支，
    # 引擎接口不变。
    builder = None
    if inference is None:
        def builder(source):  # noqa: F811 - 覆盖上面的占位
            if source.kind != "frames" or not source.frame_sequence:
                raise ValueError(f"暂不支持的素材来源: {source.kind}")
            return build_inference(
                replace(config, inference_mode="live", frame_sequence_dir=source.frame_sequence)
            )

    engine = InferenceEngine(db, config, hub, inference_impl, builder)

    return AppContext(
        config=config,
        db=db,
        hub=hub,
        engine=engine,
        events=EventService(db),
        settings=SettingsService(db),
        statistics=StatisticsService(db),
        system=SystemService(db, config, engine),
        analysis=AnalysisService(db, config, engine),
    )


def create_app(
    config: AppConfig | None = None,
    inference: InferencePort | None = None,
) -> FastAPI:
    resolved_config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        context: AppContext = app.state.context
        if resolved_config.auto_create_schema:
            context.db.create_all()
        await context.engine.start()
        # 保留期清理：NFR-003 要求启用帧持久化时必须配套保留期。
        context.analysis.start()
        try:
            yield
        finally:
            # 先取消进度跟踪任务，再停引擎：跟踪任务会读引擎状态，
            # 反过来顺序会留下一个读已停止引擎的野任务。
            await context.analysis.shutdown()
            await context.engine.stop()
            context.db.dispose()

    app = FastAPI(
        title="DriveGuard AI",
        version=resolved_config.service_version,
        description=(
            "驾安智护后端：实现 REST/WebSocket 契约与事件、配置、统计服务。"
            "当前推理为脚本化占位（inference_mode=fake），模型识别与判断尚未实现。"
        ),
        lifespan=lifespan,
    )
    app.state.context = build_context(resolved_config, inference)

    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)

    app.include_router(api_v1_router)
    app.include_router(ws_router)

    index_file = resolved_config.static_dir / "index.html"
    if index_file.exists():
        app.mount("/", StaticFiles(directory=resolved_config.static_dir, html=True), name="dashboard")
    else:

        @app.get("/", include_in_schema=False)
        def dashboard_not_built() -> JSONResponse:
            return JSONResponse(
                status_code=200,
                content={
                    "service": "DriveGuard AI",
                    "dashboard": "not-built",
                    "hint": "运行 `npm install && npm run build`（frontend/）后由 FastAPI 托管静态产物",
                    "api": "/api/v1/health",
                    "docs": "/docs",
                },
            )

    return app


app = create_app()


__all__ = ["app", "build_context", "create_app"]
