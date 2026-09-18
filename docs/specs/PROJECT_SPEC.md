# DriveGuard AI 项目 Spec

## 1. Proposal

**Intent**：以可测试的行为契约定义 DriveGuard AI MVP，避免感知模型、时序模型、后端、前端和数据层在独立迭代时产生语义漂移。

**Scope**：覆盖单车单摄像头的实时检测、时序判断、风险融合、本地告警、事件记录、REST/WebSocket、系统配置、模型信息、训练评估和文档同步；不覆盖车辆控制、多租户云平台和法规认证。

**Approach**：采用边缘优先的模块化单体，通过版本化特征、风险、数据和 API 契约连接各模块，以 Given/When/Then 场景作为开发与评审依据。

## 2. 规范用语

本文使用 RFC 2119 语义：

- **MUST / MUST NOT**：不可省略的约束。
- **SHALL / SHALL NOT**：系统必须满足的可验收行为。
- **SHOULD / SHOULD NOT**：强烈建议；偏离时需在变更说明中解释原因。
- **MAY**：可选能力。

## 3. ADDED Requirements

### REQ-001：边缘实时处理

系统 SHALL 在边缘设备完成视频采集、推理、风险融合与本地告警，核心闭环 SHALL NOT 依赖互联网或 Dashboard 可用性。

#### Scenario：正常实时处理

- GIVEN 摄像头、模型和推理管线均可用
- WHEN 连续视频帧进入系统
- THEN 系统生成带时间戳的风险快照
- AND 在目标硬件基准中测量并报告 FPS 与端到端延迟

#### Scenario：网络中断

- GIVEN 系统正在进行实时监测
- WHEN 网络连接或 Dashboard 连接中断
- THEN 本地推理和告警 SHALL 继续运行
- AND 连接状态 SHALL 被记录并在恢复后重新建立推送

### REQ-002：帧级感知质量

系统 SHALL 输出物体、人脸特征和头部姿态观测，并 MUST 为每类观测附带置信度、质量状态与模型版本。

#### Scenario：有效人脸观测

- GIVEN 画面中驾驶员面部清晰可见
- WHEN Face Landmarker 和特征计算完成
- THEN 输出 EAR、MAR、PERCLOS、Yaw、Pitch、Roll 及有效性信息

#### Scenario：面部不可见

- GIVEN 驾驶员面部被遮挡或离开画面
- WHEN 特征无法可靠计算
- THEN 相关字段 SHALL 标记为不可用或低质量
- AND 系统 MUST NOT 用正常值填充缺失特征

### REQ-003：时序行为识别

系统 SHALL 基于连续 3–5 秒的版本化特征窗口识别疲劳、手机使用、饮水与分心行为，MUST NOT 将单帧物体出现直接等同于行为成立。

#### Scenario：正常眨眼

- GIVEN 驾驶员短暂闭眼后迅速恢复
- WHEN 滑动窗口完成一次时序推理
- THEN 该过程 SHALL 可被识别为普通眨眼
- AND 不得仅因单帧闭眼创建长时间闭眼危险事件

#### Scenario：持续使用手机

- GIVEN 手机与驾驶员交互特征在多个连续时间步出现
- WHEN TCN 对完整窗口进行推理
- THEN 系统输出手机使用概率及其置信度
- AND Risk Fusion 结合持续时间决定风险等级

### REQ-004：风险融合与可解释性

系统 SHALL 仅输出 `SAFE`、`WARNING`、`DANGER` 三种风险等级，并 SHALL 提供主风险、各行为分数、置信度、输入质量和判定原因。

#### Scenario：多风险同时出现

- GIVEN 疲劳和视觉分心概率同时升高
- WHEN Risk Fusion 计算综合风险
- THEN 输出一个综合风险等级和明确的主风险
- AND 保留各子风险分数供 UI 展示和审计

#### Scenario：关键输入降级

- GIVEN 一个或多个关键专家模型不可用
- WHEN Risk Fusion 生成结果
- THEN 输出质量 SHALL 标记为 `degraded`
- AND 系统 MUST NOT 把输入缺失解释为 `SAFE` 证据

### REQ-005：本地告警状态

系统 SHALL 按可配置的持续时间、升级阈值、恢复阈值和冷却时间控制本地告警。

#### Scenario：风险升级与恢复

- GIVEN 风险从 `WARNING` 持续升至 `DANGER`
- WHEN 满足对应持续时间条件
- THEN Alarm Controller 触发升级告警
- AND 风险恢复后按恢复阈值退出告警而非单帧立即清除

### REQ-006：事件生命周期与隐私

系统 SHALL 记录危险事件元数据并聚合同一连续事件；默认 MUST NOT 持久化原始视频或完整帧。

#### Scenario：连续事件聚合

- GIVEN 同一监测会话中同类风险持续存在
- WHEN 多个连续风险快照达到记录条件
- THEN 系统更新同一事件的持续时间和最高风险
- AND 不为每帧创建独立事件

#### Scenario：默认隐私配置

- GIVEN 管理员未显式启用事件视频片段
- WHEN 危险事件结束
- THEN 数据库只保存事件元数据
- AND 不生成视频片段路径

### REQ-007：实时与查询接口

系统 SHALL 通过版本化 REST API 提供事件、统计、配置、模型和状态查询，并通过版本化 WebSocket 推送实时快照。

#### Scenario：事件分页查询

- GIVEN 数据库存在多条事件
- WHEN 客户端请求事件列表并提供合法分页参数
- THEN 返回当前页、总数和稳定排序的事件数据

#### Scenario：WebSocket 重连

- GIVEN 客户端曾连接并记录最后序列号
- WHEN 连接中断后重新建立
- THEN 服务端发送最新完整快照
- AND 后续消息序列号保持单调递增

### REQ-008：配置安全更新

系统 SHALL 校验配置类型、范围、权限和版本；失败更新 MUST NOT 覆盖当前有效配置。

#### Scenario：乐观锁冲突

- GIVEN 客户端读取配置版本 3，服务端已更新为版本 4
- WHEN 客户端携带版本 3 提交修改
- THEN 服务端返回 `409 CONFIG_VERSION_CONFLICT`
- AND 当前版本 4 的配置保持不变

#### Scenario：非法阈值

- GIVEN 风险阈值超出定义范围
- WHEN 管理员提交配置
- THEN 服务端返回 `422 VALIDATION_ERROR`
- AND 响应指出具体字段

### REQ-009：训练、版本与评估

训练流程 MUST 按驾驶员主体划分 DMD 训练、验证和测试集合；模型工件 SHALL 记录数据版本、训练配置、评估指标、运行时格式和校验值。

#### Scenario：主体泄漏检查

- GIVEN 已生成训练、验证和测试清单
- WHEN 执行数据切分审计
- THEN 三个集合的驾驶员主体交集 MUST 为空

#### Scenario：模型交付

- GIVEN 候选模型完成训练和导出
- WHEN 模型进入部署评审
- THEN 提供 Precision、Recall、F1、False Alarm/Hour、Detection Delay 和 FPS
- AND 记录 ONNX/TensorRT 工件版本与校验值

### REQ-010：代码与文档同步

每次代码迭代 SHOULD 检查是否影响需求、架构、数据结构、API、配置、部署或测试契约。发生契约变化时，相关文档与可机读规范 SHALL 在同一迭代中同步；不影响契约时 MAY 只在变更说明中记录“已检查，无需更新”。

#### Scenario：API 字段变更

- GIVEN 代码变更新增、删除或改变对外 API 字段
- WHEN 变更进入评审
- THEN API 文档、OpenAPI 与相关 Scenario 同步更新
- AND 评审者核对实现和文档的一致性

#### Scenario：内部重构

- GIVEN 代码只调整内部实现且外部行为、数据和配置契约不变
- WHEN 变更进入评审
- THEN 贡献者记录文档影响检查结果
- AND 可以不修改文档正文

#### Scenario：发现已有偏差

- GIVEN 迭代过程中发现实现与文档不一致
- WHEN 团队确认正确行为
- THEN 在当前变更或明确关联的后续任务中修正实现或文档
- AND 在修正前不得把存在偏差的条目标记为“已对齐”

### REQ-011：shadow 训练工程与双平台交付

shadow SHALL 维护公共 JSON 超参数、模型结构、GPU 和华为昇腾 NPU 独立训练入口及共享训练逻辑。数据根目录 SHALL 可外置，原始数据与派生特征 MUST 从 Git 排除；正式 checkpoint 权重 SHALL 公开交付并附配置、结构、数据切分摘要、模型卡、来源许可与校验和。

#### Scenario：本地与服务器路径不同

- GIVEN 相同相对路径清单和不同机器数据根目录
- WHEN 通过命令行或环境变量选择根目录
- THEN 得到相同数据协议与清单校验结果
- AND 不要求下载完整数据到本地工作区

#### Scenario：请求的加速器不可用

- GIVEN 用户选择 GPU 或 NPU 入口但对应设备/依赖不可用
- WHEN 初始化训练设备
- THEN 明确报告缺失平台依赖，不静默切换 CPU
- AND 无需导入 NPU 插件即可执行 GPU 入口的无依赖配置预检

#### Scenario：checkpoint 跨平台交接

- GIVEN 合法训练已产生 checkpoint
- WHEN shadow 交付该模型版本
- THEN 权重保存为 CPU state_dict 并附可重建模型配置、训练状态与 SHA-256
- AND 分别记录 GPU/NPU 的实际验证状态，未运行的平台不能标记通过

#### Scenario：权重公开许可尚未确认

- GIVEN 正式权重有第三方数据或预训练权重来源
- WHEN 来源许可尚不能确认满足公开交付要求
- THEN 在模型卡记录 pending_permission 和待核实来源
- AND 不把仅存在本地的 checkpoint 标记为已开源，继续完成可独立开展的训练工程工作

## 4. MODIFIED Requirements

当前尚无历史 Spec，本节无修改项。后续变更必须保留旧行为摘要，并说明兼容策略。

## 5. REMOVED Requirements

当前尚无移除项。后续移除行为必须给出原因、替代路径和废弃周期。

## 6. 跨层一致性规则

1. 风险枚举在模型输出、Python 类型、数据库约束、REST、WebSocket 和 TypeScript 类型中保持一致。
2. 所有时间戳以 UTC、ISO 8601 传输，数据库保存 UTC。
3. 数据库迁移先更新数据库设计，再同步 ORM/Repository、API schema 和测试。
4. API 变化同步更新 `API_DESIGN.md` 与 `openapi.yaml`；WebSocket 变化同步更新消息 schema 示例。
5. 模型特征变化同步更新 FeatureVector 版本、训练输入、推理适配和兼容策略。
6. 文档同步用于维持代码迭代的有效对齐，不把不相关的文字调整升级为开发阻塞项。

## 7. 需求追溯矩阵

| Requirement | 需求来源 | 架构组件 | 数据/API | 验证 |
| --- | --- | --- | --- | --- |
| REQ-001 | FR-001、FR-005、NFR-001/002 | Capture、Pipeline、Alarm | status、WS | E2E、断网、性能 |
| REQ-002 | FR-002 | Perception Experts | WS snapshot | 单元、录像回放 |
| REQ-003 | FR-003/004 | Window、TCN | model version | 时序模型评估 |
| REQ-004 | FR-004 | Risk Fusion | events、WS | 规则表、边界用例 |
| REQ-005 | FR-005 | Alarm Controller | settings | 状态机测试 |
| REQ-006 | FR-006、NFR-003 | Event Service | driving_events | DB 集成、隐私检查 |
| REQ-007 | FR-007/008/010 | API、WebSocket Hub | `/api/v1/*`、`/ws/v1/*` | 契约、集成测试 |
| REQ-008 | FR-009、NFR-004 | Settings Service | system_settings、PUT settings | 权限、冲突、校验 |
| REQ-009 | FR-011 | Training/Model Registry | model_versions | 数据审计、基准报告 |
| REQ-010 | AC-008、NFR-005 | 全模块 | 文档与契约文件 | 评审检查 |
| REQ-011 | FR-011、shadow 任务分配 | training、configs、models | 训练协议与 checkpoint | 路径/切分测试、设备错误、合成训练/续训、真实双平台验证 |

## 8. Spec 验证

- [x] 每个新增 Requirement 至少包含一个 Given/When/Then Scenario。
- [x] 主路径、失败或降级路径均有覆盖。
- [x] 风险、时序、隐私与性能模糊表述已给出判定方式或明确标记为实验参数。
- [x] Spec 描述行为契约，不绑定具体类名或函数名。
- [x] 文档同步规则允许内部重构声明“无需更新”，符合轻量迭代要求。

## 资料来源

- 2026-09-10 项目初始规划基线：项目目标、MVP 路线图、架构、通信方式、数据集、指标和隐私原则。
- `docs/requirements/REQUIREMENTS.md`：FR、NFR 与验收标准。
