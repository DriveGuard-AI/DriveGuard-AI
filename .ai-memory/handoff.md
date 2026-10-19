## Handoff Checkpoint

> 本仓库现在同时包含两个子项目：模型侧（`inference/`、`training/`、`configs/`、`models/`）
> 与应用侧（`backend/`、`frontend/`）。下面两段交接各自保留，互不覆盖。
> 应用侧的最新进度请看 `docs/implementation/MVP_STATUS.md`（截至 2026-10-19 已完成离线分析 M1–M5）。

### 应用侧交接（2026-09-11）

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
### 模型侧交接（2026-09-19）
**更新时间**: 2026-09-19
**当前目标**: shadow在model分支按部署方便、多平台兼容、可验证收益推进轻量关系门控；不以论文创新为目标。
**当前阶段**: 用户要求推进到CPU合成训练验证。本轮已实现关系门控模型、v2数据/质量契约、年龄导出、共享训练和CPU/GPU/NPU入口、模型工厂与重载/续训；不再停留在设计阶段。数据/服务器任务仍不自动恢复。
**完成度**: CPU模型骨架交付完成，默认13,524参数；两个现有CPU环境各128项测试通过。真实DMD、GPU/NPU、业务告警/任务支持组合仍未完成。改动保留model分支工作区，未提交。

**本轮验证**: .venv-perception torch2.7.1+cpu为128项/50.454秒，.venv-training torch2.14.0+cpu为128项/45.817秒。独立spec与quality审查通过。runs/relation-smoke-20260919报告2轮→续训3轮与连续3轮权重/history完全一致，best重载有限值。文档/图示静态检查通过，OpenAPI仍非完整schema验证。

### 已完成

- 2026-09-19 15:03：关系门控CPU工程已实现，新增23项测试；修复不合格流历史影响其他门控、__init__提前导入torch、无输入正标签误满足监督覆盖。原105项测试保留。新训练独立JSON/三平台入口，旧v1继续可用；提取显式--group-profile才产生v2。
- 新入口：docs/training/RELATION_RUNBOOK.md、docs/experiments/20260919-relation-training.md。特征训练无需感知包；server必须准备v2五数组、真实标签和主体清单，不是只指定原始DMD目录。可选缺失增强/真实标注解析器/任务支持部署适配尚未实现。下列早期记录保留历史背景。

- 2026-09-19部署优先调整：新增RELATION_GATED_FUSION设计及可编辑图源，同步需求/架构/Spec/数据/训练/部署/测试；保留38通道TCN，冻结感知模型，选用四流共享窄TCN和任务门控。导出按目标平台需要安排，不增审批门禁。

- 2026-09-19：新增被忽略的.venv-perception（Python3.12.14、torch2.7.1+cpu、NumPy1.26.4、MediaPipe0.10.21、YOLOX0.3.0固定官方源码）；三个模型位于models/local，runtime.local.json填好来源/hash/许可状态。工件清单models/PRETRAINED_ASSETS.md、安装记录docs/training/PERCEPTION_SETUP.md、Windows版本快照均已同步。原训练环境与NPU依赖未改，未连接远程服务器。
- 当前pip check有4项已记录的YOLOX上游依赖声明警告（onnx、onnx-simplifier、onnxruntime、opencv-python）；实际cv2由opencv-contrib提供。本次只是CPU感知运行时，不是完整YOLOX训练/导出环境。以下2026-09-18记录保留其历史上下文，不代表当前仍缺本地工件。
- 新增 inference/experts 工件校验、YOLOX0.3.0 Nano/Tiny安全加载/FP32/CPU NMS、MediaPipe Face/Hand CPU VIDEO；新增本地视频PTS采样、IoU稳定ID、runtime配置/CLI与record.provenance。新增43项测试，均以合成外部API边界为限，没有真实模型运行。
- 修复review发现：全局float64模型dtype、多脸时手归属、资源关闭失败仍发布索引、ffconcat二次文件未纳入溯源。限制MOV/MP4、Matroska/WebM、AVI/file协议与MOV外部轨道。requirements-perception.txt为独立候选清单，未安装；原NPU依赖未改。models/local和perception/*.local.json忽略，正式releases不忽略。

- 新增完整多专家、应用选型、部署与 PPT 14 页追溯文档；旧 24 通道 TCN 降为基线/消融子设计。
- 需求/架构/Spec/训练职责与计划/测试/导航同步；Spec 14 项均有 Scenario。
- API 草案 0.2.0：10 个操作，新增预览与无效快照 503，runtime 与五表 DDL 同步支持 mediapipe/python。
- 内存 SQLite 执行 DDL/索引，并检查五种 runtime 接受、非法 runtime 拒绝；OpenAPI 62 个引用静态解析到目标。
- 新增 training/common、training/temporal、configs/training、tests：2F 可配因果 TCN、masked BCE、录制级 mmap 特征、训练主体归一化、JSON 校验、共享 GPU/NPU 入口与 epoch 恢复。
- 29 项测试通过，CPU 两轮续训至第三轮与连续运行参数逐位相同；独立审查通过。修复跳过未知 batch 后覆盖计数遗漏并保留回归测试。
- 已有本地环境 Python 3.12.14 / torch 2.14.0+cpu / NumPy 2.3.5；未安装依赖。双入口 -S 预检通过，不初始化设备。
- 合成工件位于 runs/temporal-smoke-20260918，Git 忽略；新增 RUNBOOK、DATA_CONTRACT、工程实验记录和 models/releases 公开交付说明；文档/空白/忽略检查通过。
- 新增 inference/perception 的 geometry/profile/sequence/export：19特征38通道候选、严格packet/因果统计、流式未标注数组、record索引。33项新测试；修复几何溢出、眨眼窗口起点、失败索引发布和可变ROI元数据。指导见PERCEPTION_BRIDGE，工件均为临时合成数据。

### 未完成

- YOLOX权重来源已记录，但再发布许可仍pending_permission；真实视频/PyAV解码验收、检测器双平台训练、自动驾驶员归属/头姿、真实profile标定、推理质量适配、融合、应用/部署未完成。两帧空白图不证明检测/坐标精度；本地配置录制ID/ROI仅为测试值。
- 数据任务保持暂停；Gaze 两片/Drows 一片/Dist 一片尚未核对覆盖。未做真实训练或 GPU/NPU 实机训练验证；本机已在PyTorch2.7.1+cpu跑测试，但不是远程NPU验证；没有正式权重或 DMD 跑分。
- 本机无可用 YAML 解析库，OpenAPI 本轮仅做静态结构/引用与枚举检查，尚需正式 schema validator 和实现后的 FastAPI diff。旧 draw.io 标记为原概览/TCN基线；本轮新增关系门控图单独检查，不代表旧图全量验收。

### 关键决策

- 文档同步采用轻量影响检查；契约变化才同步受影响文档，内部重构可声明无需更新。
- 保留 model 分支及此前 requirements 修改；shadow 负责算法工件，不自动承担全部前后端任务。正式权重仍必须公开，来源许可待核实时记录 pending_permission。

### 恢复入口

最新先读`docs/training/RELATION_RUNBOOK.md`及`docs/experiments/20260919-relation-training.md`；CPU烟测使用training.relation.smoke和新的runs目录。现有tests/忽略规则未改，若需Git同步测试须由用户决定取消忽略，勿自动覆盖其规则。

- **首读文件**: `docs/architecture/RELATION_GATED_FUSION.md`、`docs/training/PERCEPTION_SETUP.md`、`models/PRETRAINED_ASSETS.md`、`docs/training/EXPERT_RUNTIME.md`、`docs/training/IMPLEMENTATION_PLAN.md`。
- **关键命令**: `.venv-perception/Scripts/python.exe -B -m unittest discover -s tests -v`；`.venv-perception/Scripts/python.exe -B -S -m inference.perception.video --config configs/perception/runtime.local.json --check-config`；`python -S -m training.temporal.npu_train --check-config`；`git -c core.autocrlf=false diff --check`。训练仍只接收manifest；结构预检不代替工件校验/前向。复验脚本须使用新的--output路径，原始报告不可覆盖。
- **验证路径**: 文档本地链接/锚点、REQ 场景、SQLite DDL、OpenAPI schema/生成差异；各模型与系统指标分层验证。

### 阻塞项

- 当前设计无硬件阻塞，契约和纯CPU合成模型测试可先推进。真实视频验收仍需用户选择短视频并确认录制ID/视角/驾驶员ROI；未经确认不能把驾驶员独占标志设为true。自动身份/头姿、目标Jetson参数、YOLO训练兼容/许可、真实数据与权重发布许可仍待核对。
