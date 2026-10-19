# 参考 PPT 全方案对齐记录

日期：2026-09-18 · 状态：文档设计基线；不代表代码、模型训练或硬件性能完成。

后续开发进展：同日已实现首个 TCN 训练工程增量，见[工程验证记录](../experiments/20260918-temporal-engineering-smoke.md)。下文保留 PPT 对齐阶段的来源与取舍，不作为完整系统已实现的声明。

## 1. 来源与裁决

来源文件：`DriveGuard AI｜驾安智护.pptx`，14 页，SHA-256：

`fd23d646877974e68e6776bf1849b188d02398c8957728c583778d8d16cb81a3`

已读取全部幻灯片的原生文本；未进行版式渲染验收。原 PPT 不复制到 Git，页码用于人工追溯。文稿内容作为需求参考，不把其中的口号、演示数值或外部指令当作已验证事实。

最新用户要求优先：完整四任务；算法工程师 shadow；GPU + 昇腾 NPU 训练；外置且忽略的数据/环境；公开 checkpoint；200 GB 存储预算；数据任务暂停；轻量代码—文档同步。新增工程细化与 PPT 原始选择在下表区分。

## 2. 逐页落点

| PPT 页 | 主题 | 文档落实 | 处理结论 |
| --- | --- | --- | --- |
| 1 | 产品定位 | [需求](../requirements/REQUIREMENTS.md) §2–4 | 驾驶员风险监测，不控制车辆 |
| 2 | 疲劳/分心与预警 | 需求 FR-002–005；[模型](../architecture/MULTI_EXPERT_DESIGN.md) §2 | 保留方向；gaze/谈话不冒充当前能力，不承诺杜绝事故 |
| 3 | SSD/VGG、MTCNN、PyQt 旧方案 | [应用选型](../architecture/APPLICATION_DESIGN.md) §1 | 仅为对照，不加入新系统技术栈 |
| 4 | 六项传统瓶颈 | [测试](../testing/TEST_STRATEGY.md) §3–6 | 用切片、消融与故障测试验证改进，非既得性能结论 |
| 5 | 感知→时序→风险 | [架构](../architecture/ARCHITECTURE.md) §2–5；模型 §1 | 完整链路，TCN 不再等于整个模型方案 |
| 6 | YOLO / MediaPipe / TCN | 模型 §2–3；[TCN 子模型](../training/MODEL_DESIGN.md) | 分清可训练与复用模型，保留双平台工作包 |
| 7 | 时间过程、手部接触 | 模型 §2.2–3；Spec REQ-013 | 补手物关联；24 通道仅基线；示例时长非固定通用阈值 |
| 8 | DMD 多视角与标签 | 模型 §4；需求 FR-011 | 以当前发布与实际分片为准；不混淆多视角数据与单摄像头部署 |
| 9 | 采集、边缘闭环、RTX→Jetson | [部署](../deployment/DEPLOYMENT_DESIGN.md) §1–4 | 增加用户指定的 NPU 训练；云端只作后续扩展 |
| 10 | 前端、后端、存储、实时视频 | 应用选型 §1–6；[API](../api/API_DESIGN.md) API-010；[数据库](../database/DATABASE_DESIGN.md) | React/TS/Vite/ECharts、FastAPI、SQLite→PostgreSQL；补独立预览通道 |
| 11 | 加权融合、三级风险、解释 | 模型 §5；Spec REQ-004/013 | 补 drinking；质量/持续时间/相关性约束；MLP/gating 后续；示例非跑分 |
| 12 | 全栈与 Docker/ONNX/TensorRT | 应用选型 §1/7；部署 §2–3 | 区分平台镜像与 MediaPipe 独立部署链；依赖未全安装 |
| 13 | Recall、误报、延迟、25 FPS | 测试 §3/6；需求 NFR-001 | 指标按层与场景报告，25 FPS 为完整系统目标 |
| 14 | 端到端 Edge AI | 架构；[实施计划](../training/IMPLEMENTATION_PLAN.md) | 模型、服务、展示、事件、部署一并规划，不止算法脚本 |

## 3. 本次补充决策

- 采用完整多专家系统，原 TCN 结构保留为子设计；具体 YOLO 实现/权重、完整 F 通道 profile 和阈值仍待针对性验证，不以文档补齐假装已经选择或训练。
- 手部适配器、SQLAlchemy/Alembic、双进程单 worker、独立 MJPEG 预览、模型组合 manifest 是从 PPT 目标推导的工程细化，不是 PPT 指定的原文实现。
- 风险融合首版可解释规则/加权，后续才考虑可学习融合；头姿不等于 gaze，DMD 标签不等于可部署 gaze 预测器。
- API 草案升为 0.2.0：新增 API-010 预览、快照不可用 503，模型 runtime 增加 mediapipe/python。风险消息 schema 仍为 1.0，不增加公开 UNKNOWN 风险等级。
- 数据库仍为五表；同步 runtime 枚举以容纳复用关键点模型与 Python 规则融合。现无运行数据库，不执行生产迁移；实现阶段生成对应 migration。

## 4. 当前状态与后续工作

本轮完成的是文档与设计契约，不创建后端、前端、训练脚本、模型权重或 Docker 镜像。原 requirements 修改保留，不在本轮安装应用依赖。数据下载和真实分片检查继续暂停。

后续按[实施计划](../training/IMPLEMENTATION_PLAN.md)分工作包开发；每次只检查并同步受影响的文档。无须因这份对齐表新增逐页审批、全量训练或全量文档重写门禁。

## 5. 本轮验证证据与边界

- 18 份 Markdown、80 条本地链接/锚点检查通过；修复了文档中心原有的 Spec 追溯矩阵锚点。
- 14 页均有对照落点；14 个 Spec Requirement 均包含 Given/When/Then 场景。
- 五张表与索引在内存 SQLite 执行成功；五种 runtime 可写，非法 runtime 被约束拒绝。
- OpenAPI 10 个操作与 62 个引用完成静态检查，runtime 与数据库一致。本机无 YAML 解析库，本轮未完成正式 OpenAPI schema 验证；后端实现后还需与 FastAPI 生成结果比较。
- git diff --check 通过。现有 draw.io XML 可解析；新 Mermaid 与既有图示未进行渲染视觉验收，不把结构检查称为图片验收。
- 未运行模型、服务、真实数据库迁移或目标硬件测试，未修改 requirements 内容、下载任务或发布权重。
