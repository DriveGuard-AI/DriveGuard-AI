# DriveGuard AI 架构设计文档

## 1. 架构目标

状态：0.2.0 设计基线，2026-09-18 按参考 PPT 全方案对齐。TCN、特征桥接及[本地专家运行时](../training/EXPERT_RUNTIME.md)已实现并做 CPU 合成边界验证；2026-09-19 已通过真实工件CPU加载与空白帧前向，真实视频和完整系统集成仍未验证。详细分工见[完整多专家模型](MULTI_EXPERT_DESIGN.md)、[应用技术选型](APPLICATION_DESIGN.md)、[部署方案](../deployment/DEPLOYMENT_DESIGN.md)和[PPT 对照记录](../governance/PPT_ALIGNMENT.md)。

DriveGuard AI 采用边缘优先、模块化、契约驱动的架构。核心风险判断在车端完成，后台与 Web 界面负责管理、展示和历史分析。设计优先保证低延迟、断网可用、隐私保护、模型可替换和问题可诊断。

2026-09-19补充：便于部署与多平台兼容优先，不以论文创新为目标。[轻量关系门控](RELATION_GATED_FUSION.md)作为现有TCN的可选升级设计，复用预训练专家与公共训练逻辑；模型源码/数据语义统一，训练和推理平台依赖分别适配。默认不引入大Transformer、端到端联合训练或自定义设备算子；新网络及训练骨架已实现并完成CPU合成验证，见[训练接入指南](../training/RELATION_RUNBOOK.md)；业务告警和硬件验证仍待完成。

原始分层架构图源文件：[DriveGuard AI 系统架构图](driveguard-system-architecture.drawio)。它是 v0.1 概览，未细化本次新增的手物交互与视频通道；完整边界以本文及多专家设计的 Mermaid 图为准。当前环境未检测到 draw.io CLI，源图尚无 PNG 视觉验收。

## 2. 总体分层

```text
Camera → Capture → Perception Experts → Feature Contract
       → Sliding Window → TCN → Risk Fusion → Local Alarm
                                      └────→ Backend API → Dashboard / Database
```

| 层 | 组件 | 核心职责 | 不负责 |
| --- | --- | --- | --- |
| 输入层 | Video Capture | 摄像头连接、帧读取、时间戳、画面质量 | 行为判断 |
| 感知层 | YOLO、MediaPipe Face/Hand、Head Pose、Interaction | 输出帧级物体/面部/手物观测及质量信息 | 直接判定手机使用或疲劳 |
| 时序层 | Feature Builder、Window Buffer、TCN | 将连续观测转换为行为概率 | 告警策略与持久化 |
| 决策层 | Risk Fusion、Alarm Controller | 风险分级、可靠性降级、告警状态机 | UI 展示 |
| 应用层 | FastAPI、WebSocket、Services | 查询、配置、统计、实时消息和事件编排 | 运行模型内部实现 |
| 数据层 | SQLite、Model Artifacts、Config | 事件/配置/模型元数据与模型文件 | 默认保存原始视频 |
| 展示层 | React Dashboard | 实时监控、事件、统计和设置 | 最终权限判断 |

## 3. 组件设计

### 3.1 Video Capture

- 通过统一 `FramePacket` 契约封装 `frame_id`、`captured_at`、图像、尺寸和质量状态。
- 后端可选择 OpenCV 或 GStreamer 适配器，但上层不依赖具体采集实现。
- 摄像头断开、超时、重复帧和时间戳倒退必须产生状态事件。

### 3.2 Perception Experts

- `Object Detector`：输出目标类别、置信度和边界框。
- `Face Landmarker`：输出关键点、检测置信度和面部可见性。
- `Feature Extractor`：计算 EAR、MAR、PERCLOS、眨眼和闭眼时长。
- `Head Pose`：输出 Yaw、Pitch、Roll 及估计可靠性。
- `Hand/Interaction`：手部适配器与驾驶员归属、手物距离/重叠/持续交互特征；不以物体靠近人脸代替手部证据。MediaPipe Face 不自动包含手部或 gaze 行为识别。
- 所有专家输出统一包含 `model_version`、`latency_ms`、`quality` 和 `errors`。

专家之间不直接互相调用，由管线编排器收集结果，避免模型耦合。

### 3.3 Temporal Pipeline

`FeatureVector` 使用版本化 schema，将帧级特征映射为固定顺序或具名张量。Window Buffer 维护 3–5 秒滑动窗口，显式处理丢帧、低质量特征和采样频率变化。TCN 输出各行为概率与模型置信度。

时序模型详细候选见 [TCN 模型设计](../training/MODEL_DESIGN.md)：默认 25 Hz、4 秒窗口、5 个因果残差块、4 个独立二分类 logits。12 个观测值与 12 个掩码是基线/消融配置；完整方案需补充手物交互与眨眼统计，输入通道由版本化 schema 的 2F 推导。概率由逐类 sigmoid 得到，不直接等于已校准置信度。训练 label_mask 与输入 mask 独立，未知标签不参与对应损失；单个样本不混合多摄像头特征。

### 3.4 Risk Fusion

首版采用可解释规则融合：行为概率、持续时间、输入质量与冷却状态共同决定风险。融合器输出：

- `risk_level`
- `primary_risk`
- `risk_scores`
- `confidence`
- `quality`
- `reasons`
- `generated_at`

融合器不得将缺失专家结果静默替换为安全值。未来学习型融合只能替换内部实现，不改变外部字段语义。

四任务加权包含 drinking，结合质量、持续时间、单项严重风险与重复计数检查。部分任务不可用但有充分危险证据可输出 degraded 的 WARNING/DANGER；证据不足时只发布系统不可用状态，不生成伪造 SAFE 快照。详见完整模型设计 §5。上述 reasons/generated_at 是内部概念，公共 API 统一使用 reason_codes/captured_at，发送时间放 WS envelope.sent_at。

### 3.5 Alarm Controller

使用状态机管理 `SAFE → WARNING → DANGER → RECOVERING → SAFE`，负责告警延迟、升级、冷却和重复抑制。RECOVERING 仅是内部告警状态，公开等级仍为三种。本地告警与后台 API 解耦，即使数据库或网络不可用也应继续工作。

### 3.6 Backend

- API Router：协议转换和输入校验。
- Monitoring Service：读取实时快照与运行状态。
- Event Service：接收推理侧事件边界，幂等持久化/更新/结束并提供查询；不在 API 侧重新判定风险。
- Settings Service：配置校验、版本化更新和生效策略。
- Statistics Service：基于事件数据生成聚合结果。
- WebSocket Hub：管理连接、订阅、序列号和背压。

路由不得直接访问数据库或模型对象；业务规则收敛在 service 层。

### 3.7 Frontend

React + TypeScript + Vite + ECharts Dashboard 分为实时监控、事件历史、统计和系统设置四个功能域。WebSocket 承载风险快照、系统状态和事件通知；查询与配置使用 REST。实时视频走独立鉴权 MJPEG 通道，默认关闭。前端只负责交互级校验，权限和数据约束由后端兜底，不自行重算风险。

## 4. 核心数据流

### 4.1 实时推理

1. Capture 生成带 UTC 时间戳的帧。
2. 感知专家并行或按硬件能力调度，产生帧级结果。
3. Feature Builder 合并并验证特征，写入滑动窗口。
4. TCN 对完整或允许降级的窗口输出行为概率。
5. Risk Fusion 计算综合风险并驱动 Alarm Controller。
6. 推理侧按融合/告警状态生成事件边界；API Event Service 接收并持久化；WebSocket Hub 推送最新有效快照及状态。

### 4.2 事件生命周期

```text
candidate → active → ended
                └→ discarded（未达到最短持续时间）
```

同类、同会话且处于合并时间间隔内的连续风险聚合为一个事件。事件结束后才固化最终持续时间和最高风险等级。

### 4.3 配置更新

1. 客户端读取配置及其 `version`。
2. 提交修改时携带旧版本。
3. 服务端校验类型、范围、权限和版本冲突。
4. 保存新版本并返回 `applied`、`restart_required` 或拒绝原因。

## 5. 关键契约

| 契约 | 生产者 | 消费者 | 版本策略 |
| --- | --- | --- | --- |
| `FramePacket` | Capture | Perception | `schema_version` |
| `PerceptionResult` | 专家模型 | Feature Builder | 每模型版本 + schema 版本 |
| `FeatureVector` | Feature Builder | Window/TCN | 严格 schema 版本 |
| `RiskSnapshot` | Fusion | Alarm/API/Event | 向后兼容字段扩展 |
| REST API | FastAPI | Dashboard/工具 | URL `/api/v1` |
| WebSocket message | WebSocket Hub | Dashboard | 消息内 `schema_version` |

## 6. 部署视图

### 6.1 MVP 单机边缘部署

```text
Jetson/RTX Host
├── inference process
├── FastAPI process
├── React static assets
├── SQLite database
└── model/config storage
```

采用单仓、统一发布的模块化应用，但 inference 与 API 运行在独立受监护进程，保证 API 故障不停止本地告警。FastAPI 首版单 Uvicorn worker；本机有界 IPC 传递最新快照和事件，不引入外部消息中间件或微服务。缓冲容量、幂等和丢失边界见[应用设计](APPLICATION_DESIGN.md)。

### 6.2 运行时策略

- 开发：PyTorch/ONNX Runtime，SQLite，Vite 开发服务器。
- 边缘：YOLO/TCN 使用已验证的 ONNX Runtime 或 TensorRT 工件；MediaPipe 使用自身兼容运行时，不假定可直接转 TensorRT。SQLite、静态前端由 FastAPI 或反向代理提供；按目标平台打包 Docker 镜像。
- 云端扩展：API 与推理解耦、PostgreSQL、多设备标识；不属于 MVP。

## 7. 可靠性与降级

### 训练开发边界（2026-09-18）

shadow 在 `model` 分支维护 `training/` 与 `configs/training/`。GPU/CUDA 与华为昇腾 NPU/torch_npu 入口共用模型和 FP32 训练循环，平台初始化单独适配。训练数据根目录允许机器间不同；发布权重统一保存 CPU state_dict。NPU 是新增的**训练平台**，现有 Jetson/TensorRT 部署规划不因此代表支持 NPU 推理。具体协议见[训练职责约定](../training/TRAINING_CHARTER.md)。

| 故障 | 系统行为 |
| --- | --- |
| 摄像头不可用 | 停止使用旧帧，状态标记不可用，周期重连 |
| 人脸暂时不可见 | 输出 degraded/unknown，不直接判安全 |
| 单专家模型失败 | 记录错误，按可靠性策略继续其余专家 |
| TCN 不可用 | 可选使用明确标识的规则降级，禁止冒充 TCN 输出 |
| 数据库写入失败 | 本地告警继续；事件进入有界重试/内存缓冲并产生告警 |
| WebSocket 客户端过慢 | 丢弃过期快照，保留最新状态，不阻塞推理线程 |
| 网络断开 | 本地完整运行，Dashboard 显示断开并重连 |

## 8. 安全与隐私

- 原始视频停留在边缘设备；视频片段持久化默认关闭。
- API 输入在边界层校验，管理操作启用鉴权与审计。
- 模型和配置路径必须限制在允许目录内，拒绝路径穿越。
- 日志使用关联 ID，不记录 Token、完整图像或无必要的个人信息。
- 前端无法替代后端授权判断。

## 9. 可观测性

- Metrics：采集 FPS、各模型 P50/P95 延迟、端到端延迟、丢帧率、数据库写入错误、活动 WebSocket 数。
- Logs：组件、时间、级别、错误码、session/event/request ID。
- Status：摄像头、模型、管线、数据库和版本信息通过 `/api/v1/system/status` 暴露。

## 10. 架构决策

### ADR-001 边缘优先

- 决策：核心推理与告警在本地运行。
- 理由：低延迟、断网可用、减少视频上传和隐私风险。
- 接受的代价：边缘算力限制、设备差异和部署复杂度。
- 重新评估条件：业务明确需要跨车队聚合，且隐私、带宽和时延约束可满足。

### ADR-002 MVP 使用模块化单体

- 决策：先使用清晰分层的单仓模块化应用，不预先拆微服务；本地 inference/API 进程分离以满足故障隔离，统一部署发布。
- 替代方案：模型服务、事件服务和 API 各自独立部署。
- 反选理由：当前无规模和隔离证据，拆分会增加部署、观测和一致性成本。
- 重新评估条件：独立伸缩、故障隔离或 GPU 调度成为已测量瓶颈。

### ADR-003 SQLite 到 PostgreSQL 的演进路径

- 决策：MVP 使用 SQLite，通过 repository 边界和兼容数据类型控制迁移成本。
- 接受的代价：写并发和远程访问能力有限。
- 重新评估条件：多设备集中写入、并发管理用户或跨节点部署。

## 11. 已知限制

- 当前已有独立 TCN 训练工程与[观测特征桥接](../training/PERCEPTION_BRIDGE.md)；真实专家调用、在线组件接口、并发模型和完整感知管线仍需实现验证。
- 模型阈值、窗口缺失策略、告警状态机参数尚需数据实验确定。
- draw.io CLI 在当前环境不可用，架构图仅提供可编辑源文件，未完成图片渲染自检。

## 资料来源

- 2026-09-10 项目初始规划基线：系统架构、AI 管线、时序识别、风险融合、后端、数据、部署和隐私设计。
- `docs/requirements/REQUIREMENTS.md`：FR-001 至 FR-011、NFR-001 至 NFR-006。
