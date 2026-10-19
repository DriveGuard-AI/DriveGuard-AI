# DriveGuard AI MVP 实现状态

> 本文记录 **2026-09-11** 交付的前后端 MVP 骨架与文档的关系。
> 相关信息可信度分级沿用 `docs/README.md`：**设计基线 / 已实现 / 待实现**。

## 1. 本次实现范围

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| 后端 REST API（API-001 ~ API-009） | **已实现** | 路径、字段、错误码与 `docs/api/API_DESIGN.md` 一致 |
| 离线分析任务（API-010 ~ API-018，2026-10-19） | **已实现** | 创建/列表/详情/取消 + 上传自动抽帧 + 回放证据 + 与实时互斥（409 + `force`，`/monitoring/resume` 切回实时）；`analysis_jobs` 表（迁移 `0002`/`0003`）；上传配套保留期清理与磁盘上限（NFR-003） |
| WebSocket（WS-001 `/ws/v1/monitoring`） | **已实现** | envelope、`sequence` 单调、心跳、10 Hz 节流、慢客户端丢旧快照、`event.changed` 优先 |
| 事件服务（FR-006） | **已实现** | 生命周期 `candidate → active → ended / discarded`，同类同会话聚合，最短持续时间校验 |
| 配置服务（FR-009） | **已实现** | 键注册表校验（类型/范围/跨字段）、乐观锁、审计日志（secret 去敏） |
| 统计服务（FR-008） | **已实现** | 总量/类型/风险等级/按天趋势；空数据返回 0 与空数组 |
| 数据层 | **已实现** | 6 张表 + CHECK 约束 + 索引，Alembic `0001_initial` / `0002_analysis_jobs`，WAL |
| 鉴权与权限 | **已实现** | Bearer Token、`monitoring:read / monitoring:write / settings:read / settings:write / models:read / analysis:read / analysis:write`；配置 Token 后除 `/health` 外一律要求鉴权 |
| 可观测性（NFR-006） | **部分实现** | `/system/status` 暴露 FPS/延迟 P50/P95/丢帧率/WS 客户端数/DB 可写性；结构化日志待补 |
| 前端 Dashboard | **已实现** | 实时监控 / 离线分析 / 事件历史 / 统计 / 系统设置五域，React + TS + Vite + ECharts，构建产物由 FastAPI 托管；事件与统计支持按会话过滤，监控页在 `pipeline.playback=true` 时提示“离线回放” |
| 模型识别与判断（FR-002 ~ FR-004） | **未实现** | 见下节 |
| 摄像头采集（FR-001） | **未实现** | 无 `FramePacket` 采集适配器 |
| 训练与评估（FR-011） | **未实现** | 无 DMD 数据处理与训练代码 |

## 2. 模型未实现期间的替代方案（重要）

- 新增推理端口 `backend/app/inference/port.py`：`InferencePort` 是模型与业务逻辑之间的唯一边界。
- 当前注入 `FakeInference`（`backend/app/inference/fake.py`）：按 **确定性脚本场景** 回放
  （`safe → fatigue → safe → phone_use → drinking → degraded`），可稳定复现 E2E 场景。
- **如实声明降级**，避免被误读为真实模型能力：
  - 每个快照携带 `inference_mode = "fake"`
  - `/api/v1/system/status` 返回 `pipeline.inference_mode = "fake"`、`pipeline.degraded = true`、`model.loaded = false`
  - 前端顶部与实时监控页均有醒目提示；`model_versions` 中登记的是 `*-stub` 占位模型
- 接入真实模型时只需提供一个 `InferencePort` 实现并在 `create_app` 注入，**REST 契约、数据库结构与前端不需要改动**（`docs/architecture/ARCHITECTURE.md` §5）。

## 3. 与文档的偏差与说明

| 项 | 文档基线 | 实现取值 | 说明 |
| --- | --- | --- | --- |
| `RiskSnapshot` 字段 | API-009 | 新增可选字段 `inference_mode` | `/api/v1` 内允许新增可选响应字段（API 设计第 8 节）；用于声明降级 |
| 事件统计口径 | FR-008 | 排除 `discarded` 事件 | `discarded` 未达到最短持续时间，不应计入危险事件数 |
| 未配置 Token 时的行为 | 生产必须鉴权 | 未配置 Token 时匿名仅具读权限（本地可信演示） | 配置 `DGB_ADMIN_TOKEN` 后除 `/health` 外全部要求鉴权 |
| 数据库初始化 | 版本化 migration | 同时支持 `DGB_AUTO_CREATE_SCHEMA`（开发便利）与 `alembic upgrade head` | 生产应以 migration 为准 |
| 部署方式 | 单机边缘、进程化 | 未引入容器 | 与 `docs/testing/TEST_STRATEGY.md` 一致，见下节"文档影响" |

## 4. 验证证据

```text
后端：pytest 40 passed（契约 / 集成 / WS / 隐私 / 事件生命周期）
迁移：alembic upgrade head → 0001_initial (head)
前端：tsc -b && vite build 成功（dist/）
端到端：uvicorn 启动后 GET / 返回 Dashboard，
        GET /api/v1/system/status 报告 inference_mode=fake、processed_fps≈4.9，
        GET /api/v1/monitoring/snapshot 返回带 sequence 的版本化快照
```

## 5. 文档影响（按 `docs/governance/DOCUMENTATION_GOVERNANCE.md` §4.3）

```markdown
## 文档影响

- 影响判断：需要更新
- 关联需求或 Spec：REQ-001/002/003/004（部分）、REQ-006、REQ-007、REQ-008、REQ-010、FR-006、FR-008、FR-009、FR-010、NFR-002、NFR-003、NFR-004、NFR-006
- 已同步文件：docs/README.md（索引）、README.md（状态与快速开始）、.gitignore、本文件
- 未同步项及原因：REQUIREMENTS.md / ARCHITECTURE.md / TEST_STRATEGY.md 正文未改动——
  实现遵循既有契约，未发生契约变化，因此按"轻量影响检查"仅记录结论
- 验证：pytest 40 passed；OpenAPI 覆盖 API-001~009；WebSocket 消息 schema_version=1.0
```

## 6. 后续建议顺序

1. 接入真实 `InferencePort` 实现（Object Detector / Face Landmarker / Head Pose / TCN / Risk Fusion）。
2. 补 `FramePacket` 采集适配器与摄像头状态事件（FR-001）。
3. 结构化日志与 Prometheus 风格指标（NFR-006 剩余部分）。
4. 按 `docs/testing/TEST_STRATEGY.md` 补齐 E2E-001~008 与性能基准（需真实模型与目标硬件）。
