# TCN 训练工程 Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans；每项先写失败测试再实现。沿用用户指定的 model 分支，保留现有未提交文档，不另建工作树。

Goal: 完成可独立运行的特征 TCN 训练、验证、保存及 epoch 边界续训；不启动 DMD 数据任务或目标硬件训练。

Architecture: 依赖最小的 JSON 配置/CLI 预检，录制级 NumPy mmap 特征与 mask，共享 PyTorch 模型/循环，CUDA/NPU 设备初始化隔离。12 特征仅基线，模型支持版本化 2F；合成 smoke 不构成真实模型成绩。

Tech Stack: Python 标准库 unittest、NumPy、PyTorch；服务器目标 torch2.7.1，现有本地 CPU 环境 torch2.14.0 仅用于兼容范围内工程测试。

## 任务与文件边界

- [x] T1 模型/损失：training/temporal/model.py、losses.py；tests/test_temporal_core.py。先验证31204参数、输出与因果性，再覆盖 mask 梯度、逐类平均、全未知 None。
- [x] T2 配置/入口：training/common/config.py、device.py，temporal/cli.py、gpu_train.py、npu_train.py，configs/training/*.json；tests/test_config_cli.py。先断言无 torch 的 --check-config 可运行、配置错误失败、平台覆盖不得改结构，再实现。
- [x] T3 数据：training/common/data.py、normalization.py；tests/test_training_data.py。manifest引用同一录制 values/masks/targets/label_masks 的.npy文件，按末槽标签取窗口；拒绝跨主体划分、路径逃逸、NaN、维度/mask错误；只用train唯一槽统计。
- [x] T4 循环/工件：training/temporal/train.py、metrics.py，common/checkpoint.py；tests/test_training_loop.py。先测试未知batch不更新、epoch全未知失败、验证聚合不受batch分组影响，再实现Adam、best/last、CPU state_dict与严格resume。
- [x] T5 合成闭环/文档：training/temporal/smoke.py，RUNBOOK.md、DATA_CONTRACT.md、实验记录与models/releases/README.md。生成D盘临时数据，跑2epochs后resume第3个并与连续运行对照；同步实际实现状态，保留GPU/NPU未验证。
- [x] T6 独立审查与复验：先对Spec再对代码质量，修复问题后全量unittest、双入口 -S 预检、git diff --check；不自动合并/发布。

执行结果：2026-09-18 完成；29 项测试通过、独立审查通过。详见[工程验证记录](../experiments/20260918-temporal-engineering-smoke.md)。完成范围仅本计划，不等于完整 model 分支或多专家系统交付完成。

## 测试命令与期望

使用已有忽略环境 `.venv-training/Scripts/python.exe -B -m unittest discover -s tests -v`；单项通过 `-p test_config_cli.py` 等模式选择。RED 应因目标行为尚不存在而失败，GREEN 所有相关测试通过。

预检：`python -S -m training.temporal.gpu_train --check-config`、`python -S -m training.temporal.npu_train --check-config`；两者应打印有效配置且不加载 torch/torch_npu，也不要求存在数据或设备。

CPU smoke：`python -m training.temporal.smoke --output-dir runs/temporal-smoke`，只允许新输出目录。真实训练入口需要 --manifest、--data-root 和新的 --output-dir；GPU/NPU缺失明确失败，不回退CPU。

## 固定接口

TemporalTCN(input_channels,channels,dilations,kernel_size,dropout,window_steps) 输出[B,4] logits。masked_bce_with_logits返回loss或None、逐类loss分子和有效监督数。配置为严格JSON，active_tasks保持固定四类中的唯一子集；平台文件只覆盖backend/device_index/路径。

manifest为driveguard-feature-manifest-v1，含dataset_version、feature_schema、extractor_profile、label_mapping、records；schema特征名/采样率须与配置一致；record含record_id、subject_id、recording_id、view、subset、split、values/masks/targets/label_masks相对.npy路径。窗口由window_steps和stride生成，不复制重叠数据；训练/验证每启用任务须有正负例。

checkpoint为基础类型/tensor组成的dict，显式weights_only加载，含模型/优化器CPU状态、epoch/step/best、配置和协议/hash、normalizer、数据摘要、Git/env与RNG。resume只允许平台/路径和总epoch增加，其余配置或数据内容变化拒绝；最佳模型工件需可随输出目录迁移保留。
