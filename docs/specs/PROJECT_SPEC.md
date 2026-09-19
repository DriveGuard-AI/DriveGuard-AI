# DriveGuard AI 项目 Spec

## 1. Proposal

状态：0.3.0 设计基线，2026-09-19；新增部署优先关系门控场景，尚未实现；来源与取舍见[PPT 全方案对齐](../governance/PPT_ALIGNMENT.md)和[关系门控设计](../architecture/RELATION_GATED_FUSION.md)。公共API契约版本不随本文同步升级。

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

系统 SHALL 输出物体、人脸特征和头部姿态观测，并 MUST 为每类观测附带质量状态与模型版本；上游实际提供置信度时保留其定义，不提供时 MUST NOT 虚构置信度。未实现或不可解的姿态/观测标记不可用。

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
- THEN 有有效结论时服务端发送最新完整快照，否则发送不可用系统状态
- AND 后续消息序列号在同一 session 内保持单调递增，新会话允许重置

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
- AND 记录实际交付运行时工件的格式、版本与校验值；仅在启用对应部署后端时要求ONNX/TensorRT工件

上述为完整部署候选的评估范围；单独 YOLO/TCN 开发增量按对应层报告，不具备完整视频/融合链路的事件与系统指标写“未测”，不可填造数据。MediaPipe 记录自身运行时工件，不强制套用 ONNX/TensorRT 格式。

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

#### Scenario：训练平台与感知运行平台分离

- GIVEN 昇腾服务器使用预先提取、校验过的录制级特征训练融合网络
- WHEN 初始化公共模型与NPU训练入口
- THEN 训练 SHALL 不依赖导入MediaPipe或YOLOX运行时
- AND GPU/NPU共享结构、损失与数据语义；分别记录实机证据，不声称MediaPipe已迁移NPU

#### Scenario：显式混合后端与导出选择

- GIVEN 部署配置明确指定MediaPipe CPU及其他模块的实际backend
- WHEN 使用该已验证组合运行
- THEN 混合后端 SHALL 被记录，不将其视作静默回退或承诺全链路加速
- AND PyTorch参考运行不强制先生成TensorRT；选用的导出格式须另有数值与业务验证

### REQ-012：四任务与部分监督（新增设计细化）

TCN SHALL 保留疲劳、手机使用、饮水和视觉分心四个可同时成立的输出。训练 SHALL 区分输入有效性与标签已知性；未标注类别 MUST NOT 自动成为负样本。具体候选见 [TCN 模型设计](../training/MODEL_DESIGN.md)。

#### Scenario：子集缺少行为标注

- GIVEN 一个 gaze 窗口没有疲劳、手机或饮水标注
- WHEN 构建监督信号与计算损失、指标
- THEN 未知类别的 label_mask 为 0，不贡献对应分类损失或指标
- AND 不将子集来源名作为模型输入或行为标签

#### Scenario：没有有效监督

- GIVEN 一个 batch 的启用任务全部没有已知标签
- WHEN 执行训练循环
- THEN 跳过该 batch 的参数更新并计数
- AND 整个 epoch 无有效更新时报告失败，不以零损失标记训练成功

#### Scenario：跨子集主体泄漏

- GIVEN 同一驾驶员出现在多个子集、分片或视角
- WHEN 生成训练、验证与测试划分
- THEN 该驾驶员全部录制及派生窗口归入同一集合

#### Scenario：未训练任务与正式交付

- GIVEN 部分任务实验只训练了四类中的一部分
- WHEN 评估或交付模型
- THEN 明确记录 active_tasks，其余任务不可作为有效预测使用
- AND 不将部分任务成绩称为完整四任务模型成绩

### REQ-013：完整多专家与质量契约

完整系统 SHALL 包含 YOLO 物体感知、MediaPipe 面部感知、驾驶员手物交互证据、TCN 与风险融合；MUST NOT 以 TCN 单模型或旧 24 通道基线代替完整方案验收。头部姿态 MUST NOT 冒充已经验证的 gaze 预测能力。

#### Scenario：背景物体与交互区分

- GIVEN 手机出现在背景或副驾驶区域，缺乏驾驶员手物交互证据
- WHEN 感知与时序模块形成观测
- THEN 手机存在不能直接构成 phone_use 结论
- AND 手部不可见时标记交互特征缺失，不以确定负例代替

#### Scenario：特征扩展与组合版本

- GIVEN 完整 profile 新增手物交互或眨眼通道
- WHEN 加载模型组合
- THEN 提取器、normalizer、输入维数与 checkpoint schema SHALL 一致
- AND 旧 24 通道权重不能静默加载到新输入；组合 manifest 追溯所有专家

#### Scenario：没有足够证据

- GIVEN 窗口未满、全部输入过期或剩余任务不足以判断当前风险
- WHEN 客户端获取状态
- THEN API-009 返回 503 PIPELINE_UNAVAILABLE，WS 发布 system.status 而非伪造 SAFE 快照
- AND 已有快照标记过期；部分任务缺失但有充分危险证据时只允许 degraded 的 WARNING/DANGER

#### Scenario：关系证据依赖失效

- GIVEN 手部观测缺失、过期或驾驶员归属未确认
- WHEN 计算手物关系、时序统计或执行专家缺失训练增强
- THEN 所有依赖该观测的字段 SHALL 同步失效，受影响历史统计重算或保守屏蔽
- AND 独立有效的面部/物体证据可保留，不残留来自完整观测的交互时长

#### Scenario：任务门控与全部证据缺失

- GIVEN 采用关系门控模型，batch中包含部分样本全部证据流无效
- WHEN 计算各任务门控权重与输出
- THEN 无效流权重为0，全部无效样本不产生NaN/Infinity且标记不可用
- AND 不将分类头bias或门控权重作为有效行为结论、校准置信度或SAFE证据

#### Scenario：新旧融合组合切换

- GIVEN 保留拼接TCN基线与关系门控候选的独立版本组合
- WHEN 显式切换或回退模型
- THEN 模型类型、质量profile、normalizer、工件和实际backend SHALL 一致，缺少年龄元数据的旧缓存不能冒充新schema
- AND 清空窗口与计时状态重新预热，不因设计升级修改现有公共风险枚举

### REQ-014：应用层隔离与受控实时预览

系统 SHALL 使用 FastAPI REST/WS、React/TypeScript/Vite/ECharts 和 SQLite MVP 持久化；inference 与 API SHALL 分进程运行，推理热路径 MUST NOT 等待 HTTP、数据库统计或视频预览消费。视频预览默认关闭且鉴权，与片段保存分别授权。

#### Scenario：后端故障与慢客户端

- GIVEN inference 正在完成本地告警闭环
- WHEN API 停止或 WS/视频客户端消费过慢
- THEN 本地推理和告警 SHALL 继续，快照/预览使用有界缓冲丢旧保新
- AND 事件缓冲溢出或持久化失败须报告缺口，不宣称零丢失

#### Scenario：预览启用与隐私

- GIVEN preview.enabled=false 或请求者未通过鉴权
- WHEN 请求 API-010
- THEN 返回 403 或 401，不发送视频帧
- AND 管理员开启后，仅授权用户可收取内存 MJPEG，默认不生成录像文件

#### Scenario：预览中途失败

- GIVEN 已建立预览连接
- WHEN 管理员关闭预览或摄像头不可用
- THEN 关闭预览流并通过系统状态解释原因，不向 JPEG 流写入 JSON 错误
- AND 不把预览关闭等同于停止监测

## 4. MODIFIED Requirements

REQ-012 为监督契约，REQ-013/014 将 PPT 完整多专家和应用层方案落入可测试行为；原有四任务目标不变。原来笼统的“重连总是有快照”细化为有效快照或不可用状态，避免未知时伪造安全。API 草案 0.2.0 增加预览、快照 503 和运行时枚举；无已发布客户端，后续实现保持 schema 同步，不另加审批门禁。

2026-09-19：REQ-011/013补充轻量关系门控、多平台边界和组合回退；REQ-009仅要求实际启用的导出格式，REQ-002禁止伪造上游未提供的置信度。新增场景为预期行为，不代表现有代码已实现。无新REST/WS字段、数据库表或风险等级，API与DDL本轮无需修改。

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
| REQ-012 | FR-004/011 | TCN、标签适配、评估 | feature/label schema | 未知标签、任务可用性、跨子集切分 |
| REQ-013 | FR-002/003/004/010、PPT 6/7/11 | YOLO、MediaPipe、Interaction、Fusion | model bundle、API-002/008/009 | 交互反例、schema 不匹配、不可用快照 |
| REQ-014 | FR-006/007、NFR-002/003、PPT 9/10/12 | inference、FastAPI、React、Repository | API-010、WS、settings | 预览鉴权/关闭、API 故障、背压/事件缺口 |

## 8. Spec 验证

REQ-012 追溯 FR-004/011，涉及 TCN、标签适配和评估；验证包括未知标签梯度、全未知 batch、跨子集主体泄漏和任务可用性，均为后续实现测试。

- [x] 每个新增 Requirement 至少包含一个 Given/When/Then Scenario。
- [x] 主路径、失败或降级路径均有覆盖。
- [x] 风险、时序、隐私与性能模糊表述已给出判定方式或明确标记为实验参数。
- [x] Spec 描述行为契约，不绑定具体类名或函数名。
- [x] 文档同步规则允许内部重构声明“无需更新”，符合轻量迭代要求。

## 资料来源

- 2026-09-10 项目初始规划基线：项目目标、MVP 路线图、架构、通信方式、数据集、指标和隐私原则。
- `docs/requirements/REQUIREMENTS.md`：FR、NFR 与验收标准。
