# DriveGuard AI 项目记忆

### Decision Record: 独立CPU感知工件准备（第1步）

**日期**: 2026-09-19
**选择**: 用户要求先做第1步，仅准备本机三个真实模型、独立.venv-perception及模型路径/hash/来源/许可，不开始真实视频、DMD处理或训练；原训练/NPU依赖不动。YOLOX0.3.0固定官方提交，Windows只排除演示文档符号链接，模型源码不改。
**证据/边界**: 新环境torch2.7.1+cpu下105项unittest通过；三个真实模型初始化及两帧空白图前向单独通过，均非精度验收。pip check仍有4项上游依赖声明警告，详见准备记录；这是受限CPU推理运行时，非完整训练/ONNX环境。YOLOX权重再发布许可pending_permission，不可作为项目正式公开权重交付。
**恢复入口**: `docs/training/PERCEPTION_SETUP.md`、`models/PRETRAINED_ASSETS.md`。下一步真实短视频需实际ROI/录制信息，不能沿用640×480测试ROI。文件均未提交。

## Glossary

### Glossary Entry: 风险快照

**规范名称**: RiskSnapshot / 风险快照  
**别名/变体**: 实时状态、风险状态  
**定义**: 某一采集时刻由风险融合层输出的版本化实时风险、特征质量和解释信息。  
**适用范围**: 推理、告警、REST、WebSocket、前端  
**禁止使用**: 将单帧检测结果称为最终风险结论  
**来源**: `docs/specs/PROJECT_SPEC.md`

### Glossary Entry: 危险事件

**规范名称**: DrivingEvent / 危险事件  
**别名/变体**: 风险事件、告警记录  
**定义**: 达到记录条件并按连续时间聚合的驾驶风险生命周期记录。  
**适用范围**: Event Service、数据库、REST、统计  
**禁止使用**: 将每一帧风险快照分别称为危险事件  
**来源**: `docs/database/DATABASE_DESIGN.md`

## Decision Records

### Decision Record: 文档先行的分层设计基线

**日期**: 2026-09-10  
**问题**: 项目只有 README，后续多模块开发缺少一致的需求、数据和接口契约。

#### 选项分析

| 选项 | 优势 | 劣势 | 复杂度 |
| --- | --- | --- | --- |
| 单一大文档 | 建立快速 | 难以维护和分工 | 低 |
| 分层文档与 OpenAPI | 边界清晰、可追溯、可机读 | 需要维护跨文档一致性 | 中 |
| 完整 OpenSpec 工件流 | 治理严谨 | 项目早期成本偏高 | 高 |

#### 决策

**选择**: 分层文档与 OpenAPI。  
**理由**: 与感知、时序、后端、前端和数据层边界一致，同时保持早期项目的维护成本可控。  
**Trade-offs**: 接受跨文档同步成本，通过追溯矩阵和轻量影响检查降低漂移。  
**影响范围**: `docs/` 全部设计文档和后续代码评审流程。  
**撤销条件**: 文档数量和变更频率导致维护成本持续高于收益，或团队正式采用其他 Spec 平台。

## Project Conventions

### Decision Record: 本地专家运行时增量

**日期**: 2026-09-18
**选择**: 用户批准后接入YOLOX0.3.0 Nano/Tiny、MediaPipe Face/Hand CPU VIDEO、PyAV PTS因果重采样、保守IoU跟踪与19特征缓存。工件须显式本地路径/hash/来源/许可，不自动下载。默认驾驶员归属未知，人工exclusive声明不等于自动身份；头姿保持mask0。候选感知依赖独立，未安装进NPU训练环境。
**证据/边界**: 全量105项测试通过（新增43）；独立规格/质量审查修复FP32、多人脸手归属、发布前关闭资源、解码播放列表溯源问题。外部模型/解码API替身，不是实际权重、PyAV或GPU/NPU运行验证；无真实跑分/正式权重。未提交，数据任务仍暂停。
**恢复入口**: `docs/training/EXPERT_RUNTIME.md`、`docs/experiments/20260918-expert-runtime.md`。下一步准备工件及独立环境，用小视频核对特征覆盖与时间坐标，不自动启动昂贵训练。

### Decision Record: 模型无关的感知桥接增量

**日期**: 2026-09-18
**选择**: 用户要求继续后，先实现专家packet到几何/时序/手物特征的桥接；19特征/38通道为独立候选，不改旧24通道。JSONL导出标签全未知，必须接真实标注和主体划分才可训练。
**证据/边界**: 全量62项测试及独立规格/质量复核通过；未下载模型或安装依赖，不等于YOLO/MediaPipe实际调用已完成。数据任务继续暂停，代码在model分支未提交。
**恢复入口**: `docs/training/PERCEPTION_BRIDGE.md`、`docs/experiments/20260918-perception-bridge.md`。

### Decision Record: 首个独立 TCN 训练工程增量

**日期**: 2026-09-18
**选择**: 用户批准开始开发后，沿用 model 分支先实施 P2 及 P1 基础特征清单；共享训练循环、CUDA/NPU 独立设备初始化、单卡 FP32；不恢复真实数据任务。
**证据**: 29 项 unittest 与独立规格/质量复核通过，本地 CPU 合成两轮续训至第三轮成功；参数与连续训练逐位一致。代码和相关文档未提交，工件在忽略的 runs 下。
**限制**: 本地 torch 2.14.0+cpu 测试不代表目标 torch 2.7.1 或 GPU/NPU 验证；没有 DMD 精度、正式权重或完整多专家系统。YOLO/MediaPipe 特征提取为下一工作包。
**恢复入口**: `docs/training/RUNBOOK.md`、`docs/training/DATA_CONTRACT.md`、`docs/experiments/20260918-temporal-engineering-smoke.md`。

### Decision Record: PPT 完整方案不等于 TCN 子模型

**日期**: 2026-09-18
**问题**: 原 24 通道 TCN 设计缺少完整感知/交互与应用、部署落点。
**选项**: 仅维护 TCN 结构简单但范围不足；完整多专家与应用分层增加协议维护成本，但符合用户提供的 PPT 与最新要求。
**选择**: YOLO + MediaPipe Face/Hand 适配 + 手物交互 + TCN + 规则融合；24 通道仅基线/消融，正式通道由 profile 决定。FastAPI 与推理分进程，React/TS/Vite/ECharts，SQLite→PostgreSQL，受鉴权预览默认关闭；GPU/昇腾训练与 Jetson 部署分开。
**理由/Trade-offs**: 保留完整产品闭环，同时承认手部遮挡、目标平台兼容及未验证指标的边界；不为文档对齐增加审批门禁。
**影响范围**: docs 下模型、应用、部署、需求、Spec、API/OpenAPI、数据库、训练计划与测试。
**撤销条件**: 用户明确改变产品范围，或目标硬件/数据验证证明某个适配器不可行；对应更新 profile、契约和文档，不静默缩减为 TCN。
**恢复入口**: `docs/governance/PPT_ALIGNMENT.md`、`docs/architecture/MULTI_EXPERT_DESIGN.md`、`docs/architecture/APPLICATION_DESIGN.md`。

### Glossary Entry: 完整多专家与时序子模型

**规范名称**: 完整多专家模型系统 / TCN 时序子模型
**定义**: 前者含感知、交互、时序与风险融合；后者仅从版本化特征序列学习四类行为。
**适用范围**: 模型设计、训练实施与产品验收。
**禁止使用**: 以 TCN 已实现/24 通道权重等同于整个系统已完成；以头姿等同 gaze 预测。
**来源**: 2026-09-18 用户参考 PPT 与全方案文档对齐要求。

### Project Convention: 主 README 保持极简

**类型**: 文档规范  
**适用范围**: 根目录 `README.md`

#### 规范内容

主 README 只保留项目简介、核心能力、工作流程、技术栈、数据集与模型、隐私原则、当前状态和文档导航。需求、架构、数据库、API、测试和路线图细节统一放入 `docs/`。

#### 示例

**正确**: README 用一段话概括 TCN，并链接到架构或需求文档。  
**错误**: 在 README 重复完整 API 字段、数据库 DDL 或逐项路线图。

#### 发现来源

2026-09-10 用户选择“极简版”README。

### Project Convention: 代码迭代中的文档同步

**类型**: 工程协作规范  
**适用范围**: 所有影响需求、架构、数据、API、配置、部署或测试契约的变更。

#### 规范内容

每次代码迭代检查文档影响。契约变化时同步受影响文档；纯内部重构可以记录“已检查，无需更新”。文档同步用于有效对齐，不把不相关文案差异扩大为开发阻塞。

#### 示例

**正确**: API 新增字段时同步 `API_DESIGN.md`、`openapi.yaml` 和契约测试。  
**错误**: API 已改变但文档保持旧字段，或内部变量重命名却无差别修改所有文档。

#### 发现来源

2026-09-10 文档基线建立任务，详见 `docs/governance/DOCUMENTATION_GOVERNANCE.md`。

### Decision Record: 部署与多平台兼容优先的关系门控

**日期**: 2026-09-19
**来源**: 用户确认轻量门控方向并明确不要求论文创新。
**选择**: 保留现有19特征/38通道TCN基线；新增四证据流、共享窄因果TCN、任务级质量门控，先冻结YOLOX/MediaPipe。新结构仍是设计，不是已训练模型。
**理由/取舍**: 对比仅拼接TCN和重型图网络/Cross-Attention，采用可小步验证的工程增量；共享权重减少参数但不代表四流只算一次。收益不足即回退，不为架构名称增加复杂度。
**兼容边界**: 共享模型和数据语义，GPU/NPU入口与依赖隔离；感知可在另一机器提取，MediaPipe可保留CPU。ONNX/TRT按需，不承诺通用engine或已验证跨平台。
**文档影响**: 需求、架构、Spec、数据v2草案、部署、训练与测试；公共API/数据库与当前配置未改变。同步只覆盖影响面，不新增审批门禁。
**撤销条件**: 实际效果不值资源成本或目标平台不能可靠支持；回退完整版本组合并重置时序状态。
**恢复入口**: `docs/architecture/RELATION_GATED_FUSION.md`。

### Glossary Entry: 学习门控与风险融合

**规范名称**: 关系门控学习层 / 风险规则与告警状态机。
**定义**: 前者在有效证据流之间学习各行为的权重；后者处理可用性、持续时间和告警恢复。门控权重不是校准置信度或因果解释，不能覆盖缺失mask或把无证据输出为SAFE。

### Decision Record: 关系门控CPU模型骨架交付

**日期**: 2026-09-19
**依据**: 用户批准推进到CPU合成训练，目标是准备模型结构和训练文件供服务器接入特征路径。
**实现**: relation_gated_v1默认13,524参数，v2配置/manifest含profile/hash和三专家真实年龄；shared训练循环与CPU/GPU/NPU入口，旧v1保留。输入预处理不使用标签/子集，原始标签和不可用正例单独统计。
**证据**: 两个现有CPU环境各128项测试；合成2轮续训到3轮与连续3轮权重/history完全一致、best重载有限；独立规格/质量审查通过。
**边界**: 可在准备好v2特征、真实标签、主体清单后指定服务器路径训练；不能只输入DMD原始视频目录。未验证GPU/NPU、不自动恢复数据任务、不安装依赖或发布合成权重。业务任务支持/告警、DMD标签解析、可选随机缺失增强另行实施。
**恢复入口**: docs/training/RELATION_RUNBOOK.md、docs/experiments/20260919-relation-training.md。

### Decision Record: 本机 CPU 联调基线 + 真实视频验证

**日期**: 2026-10-18
**问题**: 无 NPU 的本机要验证 live 复合服务与真实视频链路，并明确离应用层接入还差什么；同时不能污染发布工件。
**选择**: (1) 发布 bundle 绑定 `code_identity` 与 NPU，禁止改发布目录，改用 `scripts/local/prepare_dev_bundle.py` 生成 `runs/` 下的开发副本（硬链接权重、backend=cpu、刷新 identity）；(2) 三解释器：`.venv` 主服务/测试、`.venv-perception` MediaPipe、`.venv-detector` YOLO11；(3) 真实视频按 40ms 网格喂入（离线分析，非实时）；(4) 发现的运行时缺陷直接修，模型质量问题只记录不调阈值。
**证据/边界**: 真实素材 1114 帧、0 中断、1014 个结果、四任务全程 available、p50 250ms（0.14× 实时）。发现并修复：检测器单线程接收导致连接超时（改线程化接收 + 推理锁）、会话巡检抢锁导致 409、`core.autocrlf` 破坏发布包校验、开发副本硬链接反向写坏发布文件。未解决已记录：body 视角 `drinking` 在"手持手机贴耳"素材上 1014/1014 误报、`fatigue` 29 秒单点、body gaze 标记 unvalidated、CPU 无法满足 25Hz 输入（疲劳门控要求连续 32 个 head 槽全新鲜）。均为 CPU 环境结论，未验证 NPU/GPU 帧率与长时间稳定性。
**撤销条件**: 目标加速器实测达到 25Hz 且丢帧≈0，或产品决定放弃实时摄像头只做离线分析时，对应更新本基线。
**恢复入口**: `docs/testing/KNOWN_LIMITATIONS.md`、`docs/experiments/20261018-live-video-validation.md`、`docs/deployment/LOCAL_CPU_DEV.md`。

### Decision Record: 感知管线剖析 + 检测器异步化

**日期**: 2026-10-19
**问题**: 单帧 250ms 的构成此前只有一个与端到端数字不自洽的分阶段表（MediaPipe 17ms、eye 4ms），无法判断该优化谁；同时疲劳门控要求 8Hz 头部采样。
**选择**: (1) 先剖析再改：`scripts/local/profile_perception.py` 包裹真实 `CameraPerception` 的四个协作者，测完再决定；(2) 只把最慢的协作者（检测器，占 55%）移出关键路径，其余不动；(3) 异步策略是「忙时复用上次结果，不排队」，而非加深队列；(4) 精度影响用 `scripts/local/compare_windows.py` 按 41 个窗口逐一比对，不用行数统计。
**证据/边界**: 实测单帧 242.7ms = 解码 13.9 + 检测器 133.6 + MediaPipe 57.5 + eye 22.6 + 特征/跟踪 0.4 + 未归因 14.6。旧记录的 17ms/4ms 是单阶段空跑值，与并发成本差 3.4×/5.7×。异步后全视频 p50 282→172ms、墙钟 366.8→237.6s（1.55×），41 个窗口中 39 个完全一致；代价是 `phone_use`/`drinking` 可用率 100%→97.6%、1 个窗口 `distraction` 翻转，已记 KL-08。仍不足 8Hz（需 125ms），瓶颈转为 MediaPipe+eye+解码，全在 CPU。偶发 `[WinError 10060]` 两次运行一次，已加 30s 死锁保护快速失败（KL-09，根因未除）。未验证真实 webcam 长稳与 `ages` 长期贴近 81ms 的影响。
**撤销条件**: 换执行后端或融合导出后关键路径降至 125ms 以下，或异步的 2.4% 证据缺失在真机不可接受时回退为串行。
**恢复入口**: `docs/experiments/20261019-perception-pipeline-async.md`、`docs/deployment/LOCAL_CPU_DEV.md` §6、`inference/live/perception_worker.py`。
