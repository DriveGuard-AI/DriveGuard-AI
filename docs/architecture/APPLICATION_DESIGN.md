# 应用层与技术选型

版本：0.2.0 · 2026-09-18 · 状态：设计基线，未实现。来源：PPT 第 9、10、12 页；本文件补充实现边界，不表示依赖已安装。

## 1. 选型与代价

| 领域 | 决策 | 原因与接受的代价 | 重评条件 |
| --- | --- | --- | --- |
| 后端 | Python 3.11、FastAPI、Pydantic、Uvicorn | 与模型 Python 生态一致，REST/WS 与 schema 校验集中；异步不等于推理自动非阻塞 | 可量化吞吐/资源瓶颈 |
| 前端 | React + TypeScript + Vite + ECharts | 页面组件、类型与动态图表职责明确；需独立构建和契约同步 | UI 需求或维护成本变化 |
| 数据访问 | SQLAlchemy 2.x + Alembic（工程补充选型） | Repository 隔离数据访问、版本化迁移；不能保证切库零改动 | SQLite/PostgreSQL 双库测试反馈 |
| 存储 | SQLite 本地 MVP；PostgreSQL 集中多设备阶段 | 单机部署简单；接受单写者和备份管理成本 | 多节点写入、远程数据库或并发瓶颈 |
| 通信 | REST 查询/配置；WebSocket 状态/事件通知 | 不轮询逐帧状态；需处理慢客户端和重连 | 多设备跨节点扩展 |
| 视频预览 | 独立鉴权 MJPEG HTTP 流，默认关闭 | MVP 易于接入；带宽/编码成本较高，仅本地/受控局域网 | 低带宽或远程多人预览时评估 WebRTC |
| 打包 | Docker 分平台镜像，Ubuntu/Jetson 部署 | 可复现进程环境；驱动、JetPack/CANN 仍需宿主机提供 | 目标设备兼容性与运维约束 |

不采用 PPT 第 3 页的 PyQt、SSD/VGG、MTCNN：该页是旧方案回顾。当前不引入 Redis、Celery、Kafka、Kubernetes 或多服务治理，训练也不通过业务 API 发起。

新增数据访问选型的接口依据：[SQLAlchemy 2.0 ORM](https://docs.sqlalchemy.org/en/20/orm/quickstart.html)、[Alembic 迁移教程](https://alembic.sqlalchemy.org/en/latest/tutorial.html)。具体依赖版本在实现阶段锁定，不将选型文档当作已安装环境。

## 2. 进程与调用边界

“模块化单体”指单仓与统一发布单元，不要求所有工作挤入同一事件循环。首版运行两个受监护进程：

- `inference`：独占摄像头和模型，完成感知、TCN、融合、风险事件边界和本地告警；API 崩溃不停止它。
- `api`：单个 Uvicorn worker，接收快照/事件，管理 REST/WS、配置、数据库和前端静态文件；不在路由里解码视频、运行训练或推理。

本机 IPC 使用有界队列/本地 socket 适配器：快照是 latest-only；事件以稳定 event_id 与更新序号重试并在数据库幂等写入。MVP 不宣称进程崩溃期间事件零丢失：缓冲满或 API 不可用须记录计数/状态，告警继续；可靠磁盘 outbox 作为后续增强。原始帧不随业务事件发送。

配置先校验并持久化，再向推理进程申请生效；未确认应用的值返回 restart_required/待生效，不谎报 applied。涉及模型组合的更改在会话边界重启/切换，不在半个窗口里混用 profile。

FastAPI worker 是独立进程；首版使用一个 worker，避免内存 WS Hub、快照及 IPC 所有权分裂。将来增加 workers 前须引入共享状态/消息协调，而非只调启动参数。[FastAPI worker 文档](https://fastapi.tiangolo.com/deployment/server-workers/)。

## 3. 后端模块

| 层/服务 | 职责 | 约束 |
| --- | --- | --- |
| Routers + schemas | 请求解析、鉴权、状态码、响应 schema | 不直连模型或写 SQL |
| MonitoringService | 最新有效快照、系统状态、预览开关 | 区分进程存活、推理可用、数据新鲜度 |
| EventService | 接收风险事件边界、幂等持久化、历史查询 | 不在 API 重算风险，不逐帧建事件 |
| SettingsService | 白名单键、权限、范围、乐观锁、应用确认 | 记录审计，不允许任意宿主机路径 |
| StatisticsService | 按时间/类型/风险聚合事件 | 无数据为零，查询范围受限 |
| ModelRegistry | 列出专家版本、解析组合 manifest | 本期只读接口，不隐式远程下载/训练 |
| WebSocketHub | 版本、序列、心跳、背压、鉴权失效 | 快照可丢旧保新，事件通知溢出显式断开并提示 REST 补查 |
| Repositories | 事务、查询、SQLite/PostgreSQL 差异 | 推理热路径不等待数据库；阻塞 DB 操作不占用 async 事件循环 |

数据库采用短事务、单写入调度、WAL 和有限重试；不把 SQLite 文件放网络共享盘。WAL 允许读写并行但仍是单写者，需监控 WAL 大小和 checkpoint。[SQLite WAL 文档](https://www.sqlite.org/wal.html)。

## 4. 前端页面与数据来源

| 页面 | 展示 | 数据来源 |
| --- | --- | --- |
| 实时监控 | 风险/质量、四任务分数、EAR/MAR/PERCLOS/头姿曲线、检测结果、可选视频 | API-009 初始快照 + WS-001；API-010 视频 |
| 历史事件 | 筛选、分页、持续时间、原因与版本、详情 | API-003/004 |
| 统计分析 | 类型分布、次数、时段趋势 | API-005 |
| 系统/设置 | 设备状态、模型清单、配置与生效状态 | API-001/002/006/007/008 |

TypeScript 类型以 OpenAPI/WS schema 为准；ECharts 环形缓冲只保留有限时间点，页面隐藏时降低绘图频率。数据过期、断连、warming_up 或任务缺失必须可见；空分数显示“不可用”，不能显示为 0 或绿色安全。UI 不二次计算风险等级。

WS-001 最大默认 10 Hz，不等于模型处理 FPS。序列以会话为范围单调递增；新会话重置，前端清理旧曲线。快照跳号用 API-009 补最新；事件通知跳号或溢出用事件列表重新查询，不承诺逐帧回放。

## 5. 实时视频与隐私

PPT 的实时视频需要独立数据通道，不能只定义 JSON 状态推送。首版设计 `GET /api/v1/monitoring/preview`（API-010）：MJPEG、无磁盘落盘、受控降分辨率/帧率、慢客户端丢旧帧。该通道不承诺 25 FPS，关闭预览不影响检测。

默认 `preview.enabled=false`；管理员开启后仍需 `monitoring:read`。浏览器通过携带 Authorization 的 fetch 流读取与绘制，不把长效 Token 放 img URL；切页/断连取消请求，禁止缓存。局域网开放前配置 TLS/Origin 白名单。视频外传云端不是默认功能，远程预览需要另行明确授权与保留策略。

预览与视频片段保存是两个开关：开启预览不等于授权保存。禁止在 WebSocket JSON 里塞 base64 连续帧，也不默认上传原始帧。生产日志去除凭证和图像。

## 6. 存储与版本追溯

沿用会话、危险事件、设置、模型版本、审计五表。实时逐帧特征和视频不入库，训练特征也不进入业务库。会话绑定不可变 `model_bundle_version`，模型组合 manifest 位于受控工件目录；事件主时序模型外键加会话组合版本共同追溯全部专家。

SQLite 到 PostgreSQL 是演进项目：需要迁移 UUID/时间/JSON 类型、索引、事务与部署配置，并跑双库契约测试，不承诺替换连接字符串即完成。详见[数据库设计](../database/DATABASE_DESIGN.md)。

## 7. 依赖与代码边界

规划目录 `backend/`、`frontend/`、`desktop/`、`inference/`、`training/`、`configs/`、`models/releases/`；`desktop/` 是 v0.1 新增的 Windows 桌面封装层。

现有根 requirements 仅覆盖训练基础依赖，不能启动完整应用。实现时按训练、推理、后端分别维护 requirements 清单，前端使用 package.json/锁文件；NPU 插件仅进 NPU 环境。具体包版本在实际目标环境解析后锁定，本次不安装、不改已有依赖清单。

后端模块负责人尚未指定姓名；shadow 提供模型工件、特征协议、评估与加载说明，不默认为全部前后端开发负责人。接口详见 [API 设计](../api/API_DESIGN.md)及 [OpenAPI](../api/openapi.yaml)。

## 8. Windows 桌面封装（v0.1）

v0.1 以 Electron 提供 Windows 桌面壳，PyInstaller 打包 FastAPI 后端与前端静态产物；
渲染进程仍复用现有 React Dashboard，但用户看到的是无地址栏的原生窗口。

- 主进程负责内置配置、随机本地管理 Token、后端进程生命周期、窗口与日志；
- `preload` 只暴露本地 Token、版本和打开数据/日志目录，不暴露 Node 或模型服务 Token；
- 模型服务仍为外部依赖，桌面包不包含权重、CANN/NPU 运行时；
- 首发功能收敛为 4 页签：实时监控、离线分析、事件历史、系统设置；统计页不进入 v0.1；
- 运行数据写入 `%APPDATA%\DriveGuardAI`，不写安装目录。

详细构建、配置与验收见 [Windows 桌面版](../deployment/DESKTOP_APP.md)。
