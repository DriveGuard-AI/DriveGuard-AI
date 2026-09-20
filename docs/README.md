# DriveGuard AI 文档中心

本目录定义 DriveGuard AI 的需求、架构、行为规格、数据模型、接口契约、测试策略与文档协作方式。当前以**部署方便、多平台兼容和可验证收益**为原则，不追求论文式架构创新。当前处于 **Early Development**：已有 TCN、感知桥接及本地视频专家调用代码；三个预训练工件通过独立Windows CPU加载检查，真实视频/精度验收、融合、应用和部署仍未完成。文档中的系统目标不等于已上线能力。

## 文档索引

| 文档 | 主要读者 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| [PPT 全方案对齐记录](governance/PPT_ALIGNMENT.md) | 全体贡献者 | 14 页来源、设计落点、差异与状态 | Baseline v0.2 |
| [部署优先的轻量关系门控](architecture/RELATION_GATED_FUSION.md) | 算法、部署、测试 | 四证据流、共享TCN、任务门控、平台边界与回退 | CPU骨架与合成训练已验证 |
| [完整多专家模型](architecture/MULTI_EXPERT_DESIGN.md) | 算法、推理、测试 | YOLO/MediaPipe/交互特征/TCN/融合整体设计 | Baseline v0.3 |
| [应用层与技术选型](architecture/APPLICATION_DESIGN.md) | 前后端、架构 | FastAPI、React、视频预览、数据访问和进程边界 | Baseline v0.2 |
| [训练与边缘部署](deployment/DEPLOYMENT_DESIGN.md) | 算法、部署 | GPU/NPU 训练、Jetson 推理、Docker、导出回滚 | Baseline v0.3 |
| [需求文档](requirements/REQUIREMENTS.md) | 产品、算法、开发、测试 | 业务目标、范围、需求和验收指标 | Baseline v0.3 |
| [架构设计](architecture/ARCHITECTURE.md) | 架构、开发、运维 | 分层架构、组件边界、数据流和部署设计 | Baseline v0.2 |
| [项目 Spec](specs/PROJECT_SPEC.md) | 开发、测试、评审者 | RFC 2119 行为规则和 Given/When/Then 场景 | Baseline v0.3 |
| [数据库设计](database/DATABASE_DESIGN.md) | 后端、数据、测试 | SQLite MVP 表结构、约束、索引和迁移策略 | Baseline v0.2 |
| [API 设计](api/API_DESIGN.md) | 前后端、测试 | REST 与 WebSocket 契约、错误码和联调规则 | Baseline v0.2 |
| [OpenAPI 草案](api/openapi.yaml) | 前后端、工具链 | 可机读的 REST API 契约 | Draft v0.2 |
| [测试策略](testing/TEST_STRATEGY.md) | 开发、测试、算法 | 分层测试、数据切分、性能与验收方案 | Baseline v0.2 |
| [文档协作规范](governance/DOCUMENTATION_GOVERNANCE.md) | 全体贡献者 | 代码迭代时的文档同步与审核流程 | Active v0.1 |
| [shadow 训练职责约定](training/TRAINING_CHARTER.md) | 算法、推理集成、项目负责人 | 训练分工、GPU/NPU、数据与权重交付 | Active v0.2 |
| [TCN 子模型设计](training/MODEL_DESIGN.md) | 算法 | 因果网络、24 通道基线、部分监督与 checkpoint | Baseline v0.2 |
| [分阶段实施计划](training/IMPLEMENTATION_PLAN.md) | 各模块维护者 | 完整方案工作包与 shadow 的训练增量 | P2/P3已有工程基础，关系门控CPU工程验证完成 |
| [关系门控训练与服务器接入](training/RELATION_RUNBOOK.md) | shadow、服务器操作者 | 模型文件、五数组、GPU/NPU命令、续训与验证边界 | CPU合成闭环通过，硬件待验收 |
| [服务器初始化](training/SERVER_SETUP.md) | 服务器操作者 | Git缺文件排查、NPU依赖安装、环境预检、被忽略工件准备 | 操作指南，待服务器执行 |
| [TCN 训练指南](training/RUNBOOK.md) | 算法、服务器操作者 | 配置预检、合成验证、双入口与恢复 | 已实现，硬件训练未验证 |
| [特征数据契约](training/DATA_CONTRACT.md) | 算法、特征提取维护者 | 录制级数组、manifest、mask 与切分 | v1/v2已实现，真实标注映射待接入 |
| [感知特征桥接](training/PERCEPTION_BRIDGE.md) | 算法、感知维护者 | 专家观测到19特征、时序统计、流式导出 | 已实现，不含模型推理调用 |
| [感知桥接验证](experiments/20260918-perception-bridge.md) | 算法、评审者 | 几何/缺失/导出与38通道接线 | CPU 合成验证 |
| [本地专家运行指南](training/EXPERT_RUNTIME.md) | 算法、感知维护者 | YOLOX/Face/Hand工件、PTS视频入口、独立依赖 | 真实模型CPU加载通过，视频待验收 |
| [本机感知环境准备](training/PERCEPTION_SETUP.md) | 算法、本机操作者 | 独立环境、实际工件、已知警告、复现方式 | 第1步已完成，仅Windows CPU |
| [预训练工件清单](../models/PRETRAINED_ASSETS.md) | 算法、发布维护者 | 固定来源、hash及许可状态 | YOLOX权重再发布许可待确认 |
| [专家运行时实施计划](training/EXPERT_RUNTIME_IMPLEMENTATION_PLAN.md) | 开发、评审者 | 本轮范围、接口、测试与验收边界 | 工程增量 |
| [专家运行时验证](experiments/20260918-expert-runtime.md) | 算法、评审者 | 43项新增测试、已知限制与审查修复 | 合成API边界验证 |
| [首轮工程验证](experiments/20260918-temporal-engineering-smoke.md) | 算法、评审者 | 29 项测试、合成续训与验证边界 | CPU 合成验证通过 |
| [权重发布目录](../models/releases/README.md) | 算法、使用者 | 正式权重、来源与许可交付要求 | 尚无正式权重 |

## 信息可信度

本轮模型调整从[轻量关系门控设计](architecture/RELATION_GATED_FUSION.md)进入，图示同时提供SVG和可编辑draw.io。原有API与数据库契约未因该内部模型设计而变更。

完整产品背景建议从 [PPT 全方案对齐](governance/PPT_ALIGNMENT.md)进入，再按算法、应用、部署分工阅读。TCN 仅是完整多专家中的时序子模块；不能以其 24 通道基线代替全部模型设计。本地工程测试不证明 DMD 精度或 GPU/NPU 兼容性。

- **已确认基线**：项目目标、技术栈、总体架构、MVP 范围、隐私原则和性能目标来自 2026-09-10 建立文档集时的项目规划；根目录 `README.md` 现作为极简概览和导航入口。
- **设计基线**：为支持后续实现而定义的数据结构、接口、错误码、模块职责和验证规则。
- **已实现**：共享 TCN、部分监督训练、特征清单校验、双平台入口和 checkpoint；轻量关系门控/v2质量与年龄契约/共享训练和CPU合成验证；几何/时序桥接、本地YOLOX/Face/Hand适配、PTS因果采样及保守跟踪。实际验证范围见工程记录与运行指南。
- **待实现/验证**：真实视频/精度验收与检测器训练、自动归属/头姿、业务风险融合/告警、前后端、数据库迁移、部署及正式模型训练和权重公开；工件加载通过不等于这些能力已验证。
- **待确认参数**：模型阈值、窗口长度、告警延迟、保留周期等通过配置管理，并须由实验或产品验收确定最终值。

## 需求追溯

需求使用 `FR-*`、`NFR-*` 编号；Spec 使用 `REQ-*`；API 使用 `API-*`；数据库表使用稳定英文表名。映射关系维护在 [项目 Spec 的追溯矩阵](specs/PROJECT_SPEC.md#7-需求追溯矩阵)中。

## 变更原则

代码迭代涉及外部行为、数据结构、API、配置、部署方式或质量指标时，应在同一变更中检查相关文档并按影响同步更新。纯内部重构且不改变契约时，可以只在评审说明中记录“已检查，无需更新文档”。详细流程见[文档协作规范](governance/DOCUMENTATION_GOVERNANCE.md)。

## 资料来源

- `README.md`：当前项目简介、核心技术、数据集、模型和文档入口。
- 本文档集：基于 README 形成的第一版可执行设计契约，版本日期为 2026-09-10。
- 2026-09-18 参考 PPT 全部 14 页与用户最新约束：详见 PPT 对齐记录中的来源 hash 和取舍。
