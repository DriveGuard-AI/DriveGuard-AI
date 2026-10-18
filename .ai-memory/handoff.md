## Handoff Checkpoint

**更新时间**: 2026-09-11
**当前目标**: 按文档需求交付前后端 MVP 骨架（模型识别与判断暂不实现）。
**当前阶段**: 已完成
**完成度**: 后端 API/WS/DB/测试 + 前端四域 Dashboard 全部可运行；模型部分明确未实现。

### 已完成

- 后端：FastAPI + SQLite，API-001~009、WS-001、事件生命周期、配置乐观锁、统计、鉴权与统一错误码。
- 数据层：5 张表与 CHECK 约束、索引、Alembic `0001_initial`、WAL、临时 SQLite 集成测试。
- 推理端口：`InferencePort` + `FakeInference`（确定性脚本场景），快照携带 `inference_mode=fake`。
- 前端：React + TS + Vite + ECharts 四域 Dashboard，WS 重连/序列跳跃补快照/背压提示；构建产物由 FastAPI 托管。
- 验证：`pytest` 40 passed；`alembic upgrade head` 成功；`vite build` 成功；端到端启动验证通过。
- 文档：新增 `docs/implementation/MVP_STATUS.md`，更新 `docs/README.md` 索引与根 `README.md` 快速开始。

### 未完成

- 模型识别与判断（Object Detector / Face Landmarker / Head Pose / TCN / Risk Fusion）。
- 摄像头采集 `FramePacket` 适配器（FR-001）与摄像头状态事件。
- 结构化日志与 Prometheus 风格指标；E2E-001~004 与性能基准需真实模型与目标硬件。

### 关键决策

- 无模型阶段采用"前后端先行 + 契约化占位推理"，并强制显式声明降级（见 `project_memory.md`）。

### 恢复入口

- **首读文件**: `docs/implementation/MVP_STATUS.md`
- **关键命令**: `cd backend && python -m pytest -q`、`alembic upgrade head`、`cd frontend && npm run build`
- **验证路径**: 新增模型实现时提供 `InferencePort` 实现并在 `create_app` 注入，然后核对 REST/WS 契约未变化。

### 阻塞项

- 无开发阻塞；本机缺 draw.io CLI，仅影响架构图 PNG 导出。
