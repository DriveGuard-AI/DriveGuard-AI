# 完整方案分阶段实施计划与 shadow 训练工作包

日期：2026-09-18。训练工作包：shadow；应用/部署由相应维护者承担；分支 `model`。TCN 训练工程与观测特征桥接已实现，分别见[TCN 工程记录](../experiments/20260918-temporal-engineering-smoke.md)和[感知桥接记录](../experiments/20260918-perception-bridge.md)。

目标：落实PPT完整多专家、应用与部署方案，不将项目缩减为TCN。2026-09-19进一步明确部署便利、多平台兼容优先，不要求论文创新。NPU基础Linear测试已通过，不追加远程硬件排查。数据任务暂搁置；P2、P1/P3基础代码及三个真实工件CPU加载已完成，真实视频与目标硬件仍待验证，不代表完整交付。

方案：共享 PyTorch 模型/训练循环 + CUDA/torch_npu 双入口 + JSON 配置 + 外置特征清单；YOLO + MediaPipe + 手物交互 + TCN + Fusion 的完整设计见[多专家文档](../architecture/MULTI_EXPERT_DESIGN.md)。首版单卡 FP32，TCN 仍可作为独立代码增量，但不代表整体交付完成。

## 1. 完整工作包与进度

| 顺序/工作包 | 主要负责角色 | 依赖与交付 |
| --- | --- | --- |
| P1 特征与平台契约 | shadow，推理维护者协作 | 已有基线与19特征交互候选、严格观测协议/manifest/导出；真实专家工件组合与数据标定待完成 |
| P2 TCN 训练工程 | shadow | 已实现结构、mask 监督、配置/双入口与 checkpoint，24 通道基线通过 CPU 工程验证；目标硬件未验证 |
| P3 感知与交互 | shadow，推理维护者协作 | 几何/统计/关联、YOLOX/MediaPipe适配、PTS与保守跟踪已实现，工件CPU加载通过；真实视频、自动归属/头姿、检测器训练待完成 |
| P3.1 轻量关系门控 | shadow，推理维护者协作 | F/O/HO/FO、质量/依赖契约、共享窄TCN、任务门控与旧基线共存；模型/v2数据/双入口已实现，CPU合成训练与恢复通过；硬件和真实数据待验收 |
| P4 数据与真实训练 | shadow | 用户恢复数据任务后检查实际分片、统一主体划分、训练完整 profile 与四任务；200 GB 容量核算 |
| P5 融合与本地闭环 | 推理维护者，shadow 协作 | 质量/持续时间/加权规则、事件边界、告警、版本组合与离线回放 |
| P6 后端与数据库 | 后端维护者（姓名未指定） | FastAPI 服务/单 worker、IPC、Repository/migration、API-001–010、WS、鉴权与预览 |
| P7 前端 | 前端维护者（姓名未指定） | React/TS/Vite/ECharts 四页面，类型契约、过期/未知显示、重连与受控预览 |
| P8 部署与验收 | 部署/测试维护者，shadow 协作 | 平台镜像、目标运行时、MediaPipe、长稳/FPS/事件评估与权重公开；ONNX/目标TensorRT按需，不是全平台前置条件 |

P6/P7 可在契约确定后用模拟消息与临时数据库推进，不需要等训练完成；正式集成验收依赖 P5。P3 与 P2 可分增量实施，但完整 profile 数据必须在正式训练前对齐。以上是工作划分，不为每个包增加审批手续。

## 2. shadow 首个训练工程增量

以下勾选项保留2026-09-18首轮交付的历史范围；最新感知状态见第1节和本节末的2026-09-19记录。

- [x] 文档对齐：Requirements/Spec/Architecture/Testing/Governance 引用 shadow 职责；之后创建 model 分支。
- [x] 编写 [TCN 子模型设计](MODEL_DESIGN.md)、完整多专家/应用/部署方案与 PPT 对照；不代表模型实现完成。
- [x] 数据契约：manifest 引用录制级数组，按配置生成窗口，避免重复落盘；单窗 `[100,2F]`、基线 F=12、四类 target/label_mask；主体泄漏、非法路径和维度测试通过。完整交互/眨眼提取仍待 P3，不扫描真实数据。
- [x] 配置与双入口：公共 JSON + gpu/npu 小配置；CLI `--check-config` 不依赖 torch 或数据；设备不可用时失败，参数与路径测试通过。
- [x] TCN：因果膨胀 Conv1d 残差块、末时间步分类；形状、参数量、因果性和有限梯度通过。
- [x] 训练与 checkpoint：Masked BCE、Adam、全验证集逐类聚合选 best、epoch resume、CPU state_dict；合成两轮续训至第三轮通过，并测试与连续训练逐位一致。全未知 batch 不更新但保留覆盖计数。
- [x] 文档与交接：[RUNBOOK](RUNBOOK.md)、[数据契约](DATA_CONTRACT.md)、模型结构、工程结果页和权重发布目录；没有真实跑分或正式权重。
- [x] 完成验证：29 项 unittest、双入口预检、规格/质量审查通过；SQLite/API 契约无需因纯 TCN 增量改变，文档状态与静态检查同步。

现有入口：`python -m unittest discover -s tests -v`，`python -m training.temporal.gpu_train --check-config`，`python -m training.temporal.npu_train --check-config`。实际训练还需要符合数据契约的特征，不接收原始视频。

2026-09-19：用户指定第1步已完成，三个真实工件、独立Windows CPU感知环境及本地配置就绪，加载/空白帧前向与105项测试通过；YOLOX权重再发布许可仍pending_permission，见[准备记录](PERCEPTION_SETUP.md)。没有读取真实视频或恢复DMD任务。

恢复方式：最新主线为[轻量关系门控设计](../architecture/RELATION_GATED_FUSION.md)。本轮已完成[CPU模型骨架与训练入口](RELATION_RUNBOOK.md)；真实短视频验收仍是独立未完成项，不自动恢复下载或启动昂贵训练。数据库/接口仍为草案，硬件兼容和权重公开与加载smoke分开记录。

## 3. 关系门控增量顺序

- [x] 确认工程目标与方向，保留38通道拼接TCN；不做论文新颖性承诺。
- [x] 形成架构、平台边界、数据扩展与Spec场景；CPU阶段已落地，剩余阶段不因工程验证而自动完成。
- [x] 质量/依赖契约与v2校验器：年龄、分组、mask传播、时序统计重置、新旧缓存拒绝混用；先合成边界测试。
- [x] 四流小投影、共享窄TCN、有限值masked门控与四任务头；CPU因果性/梯度/全缺失混合batch测试。
- [x] 复用公共训练循环、GPU/NPU入口及checkpoint，增加模型类型/profile一致性，旧基线继续可运行。
- [ ] 实际可用设备上做固定小样本前向/反向/保存加载；不可用平台标未验证，不自动升级服务器环境。
- [ ] 用户恢复数据后，小规模真实提取/标注/主体划分与训练，按相同输入证据比较基线；按问题需要补对照，不默认大规模消融。
- [ ] 明确目标硬件后测试总链路误报/漏报、可用率与P50/P95；有收益才切换组合，必要时再做ONNX/TensorRT优化。

[最小代码实施计划](RELATION_IMPLEMENTATION_PLAN.md)已执行至CPU合成阶段，详见[运行指南](RELATION_RUNBOOK.md)；不把新增模型默认键直接加入当前严格v1配置。shadow负责模型与工件，应用/车端集成不因本次设计变更转为shadow独占职责。
