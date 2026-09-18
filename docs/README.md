# DriveGuard AI 文档中心

本目录定义 DriveGuard AI 的需求、架构、行为规格、数据模型、接口契约、测试策略与文档协作方式。当前仓库处于 **Planning / Early Development** 阶段，因此除 `README.md` 已明确的项目定位外，本文档集中的技术细节均属于待实现的设计基线，不表示代码已经存在。

## 文档索引

| 文档 | 主要读者 | 作用 | 当前状态 |
| --- | --- | --- | --- |
| [需求文档](requirements/REQUIREMENTS.md) | 产品、算法、开发、测试 | 业务目标、范围、需求和验收指标 | Baseline v0.1 |
| [架构设计](architecture/ARCHITECTURE.md) | 架构、开发、运维 | 分层架构、组件边界、数据流和部署设计 | Baseline v0.1 |
| [项目 Spec](specs/PROJECT_SPEC.md) | 开发、测试、评审者 | RFC 2119 行为规则和 Given/When/Then 场景 | Baseline v0.1 |
| [数据库设计](database/DATABASE_DESIGN.md) | 后端、数据、测试 | SQLite MVP 表结构、约束、索引和迁移策略 | Baseline v0.1 |
| [API 设计](api/API_DESIGN.md) | 前后端、测试 | REST 与 WebSocket 契约、错误码和联调规则 | Baseline v0.1 |
| [OpenAPI 草案](api/openapi.yaml) | 前后端、工具链 | 可机读的 REST API 契约 | Draft v0.1 |
| [测试策略](testing/TEST_STRATEGY.md) | 开发、测试、算法 | 分层测试、数据切分、性能与验收方案 | Baseline v0.1 |
| [文档协作规范](governance/DOCUMENTATION_GOVERNANCE.md) | 全体贡献者 | 代码迭代时的文档同步与审核流程 | Active v0.1 |
| [shadow 训练职责约定](training/TRAINING_CHARTER.md) | 算法、推理集成、项目负责人 | 训练分工、GPU/NPU、数据与权重交付 | Active v0.2 |

## 信息可信度

模型详细设计见 [TCN 模型设计](training/MODEL_DESIGN.md)：完整四任务方向已确认，结构和训练契约为待评审草案，尚未实现或训练。

- **已确认基线**：项目目标、技术栈、总体架构、MVP 范围、隐私原则和性能目标来自 2026-09-10 建立文档集时的项目规划；根目录 `README.md` 现作为极简概览和导航入口。
- **设计基线**：为支持后续实现而定义的数据结构、接口、错误码、模块职责和验证规则。
- **待实现**：仓库尚无源代码、数据库迁移、模型文件或部署配置，所有实现状态以代码与测试结果为准。
- **待确认参数**：模型阈值、窗口长度、告警延迟、保留周期等通过配置管理，并须由实验或产品验收确定最终值。

## 需求追溯

需求使用 `FR-*`、`NFR-*` 编号；Spec 使用 `REQ-*`；API 使用 `API-*`；数据库表使用稳定英文表名。映射关系维护在 [项目 Spec 的追溯矩阵](specs/PROJECT_SPEC.md#需求追溯矩阵)中。

## 变更原则

代码迭代涉及外部行为、数据结构、API、配置、部署方式或质量指标时，应在同一变更中检查相关文档并按影响同步更新。纯内部重构且不改变契约时，可以只在评审说明中记录“已检查，无需更新文档”。详细流程见[文档协作规范](governance/DOCUMENTATION_GOVERNANCE.md)。

## 资料来源

- `README.md`：当前项目简介、核心技术、数据集、模型和文档入口。
- 本文档集：基于 README 形成的第一版可执行设计契约，版本日期为 2026-09-10。
