# 完整 gaze 联合训练实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development；逐任务 TDD，先规格再质量审核。用户指定 model 分支，原地工作，不自动提交、合并或推送。

**Goal:** 将已批准的四行为 + 九类 gaze 方案接入真实特征提取、全量准备、联合训练、续训及推理。

**Architecture:** 新版显式 profile/config/record/manifest，不冒充旧 19 维数据；原四流网络共享面部编码，增加 gaze 头；保留旧入口与默认行为。训练相关文件与感知/DMD 文件按所有权拆开，边界通过版本化常量、配置和数组接口连接。

**Tech Stack:** 现有 NumPy/PyTorch、MediaPipe Tasks、PyAV，FP32 CPU/CUDA/torch_npu；不安装或升级依赖。

## 契约与文件所有权

- 训练子任务：`training/gaze/`、`training/common/`、必要的 `training/relation/` 兼容扩展、`configs/training/relation_gaze.json`、`tests/test_gaze_training.py`。先给出可依赖的公共 API。
- 主任务：`inference/experts/gaze_geometry.py`、MediaPipe 适配、`inference/perception/` 新 profile 与导出、`training/dmd/gaze_labels.py`、新版批处理及配置、相应测试与文档。
- 不修改旧默认配置去自动开启 gaze，不强制旧缓存升级；新版从独立输出目录开始。
- 类别固定 `left_mirror,left,front,center_mirror,front_right,right_mirror,right,infotainment,steering_wheel`。23 维为原 19 维后追加 `iris_left_x,iris_left_y,iris_right_x,iris_right_y`；原 3 个头姿槽在新版启用。
- gaze_targets int64 `[N]`，gaze_label_masks uint8 `[N]`；未知 target=0/mask=0。四行为 `[N,4]` 不变。25 Hz、100 步、stride25。

## Task 1：联合训练核心

- [x] 先编写测试：配置23维/46通道；模型输出行为 `[B,4]`、gaze `[B,9]`；gaze loss 对共享 F 编码梯度非零；全缺失不可用；单任务 batch 和未知mask无虚假监督。

```python
output = model(x, quality, eligible)
self.assertEqual(output['behavior_logits'].shape, (2, 4))
self.assertEqual(output['gaze_logits'].shape, (2, 9))
```

- [x] 运行 `.venv-training/Scripts/python.exe -m unittest tests.test_gaze_training -v`，记录因能力缺失失败后实施。
- [x] 配置/数据接口明确校验新旧版本、类别顺序、hash、数组dtype/shape/有限值/未知占位；复用训练集归一化与录制级mmap，不保存重复窗口。
- [x] 实现共同编码 + gaze F头、与推理一致的可用性、masked BCE/CE、全轮任务归一指标、分被试/视角统计；首轮保持基础超参、全五任务。
- [x] CPU/GPU/NPU薄入口、best/last保存、精确续训、重载预测接口。合成fixture跑真实数据加载和训练，续训匹配连续训练；缺硬件不得称NPU通过。

## Task 2：感知与标签

- [x] 先测已知旋转、退化矩阵、468/478拓扑、闭眼、缩放/旋转不变虹膜坐标、多脸同索引关联；运行 `tests.test_gaze_perception` 观察RED。
- [x] 新版模式请求面部矩阵，严格坐标约定与有限值/正交性检查；新头姿无效不影响有效旧特征，虹膜无效逐眼mask。
- [x] 新版profile仅显式选择；新feature schema固定23字段，runtime身份记录新模式；从实际PTS提取，输出record-v3与新数组。
- [x] 标签测试覆盖九类、not_valid、冲突、未知类型、各相机offset、间隙/未来禁止及窗口末槽。复用已有标注完整性校验，独立gaze数组不回填旧四行为目标。

```python
self.assertEqual(result['gaze_label_masks'][invalid_slot], 0)
self.assertEqual(result['gaze_targets'][invalid_slot], 0)
```

## Task 3：全量准备、缓存与发布

- [x] 编写新版init测试：旧preparation只读迁移、Dist/Drows双视角+Gaze face、ROI按精确身份复制、其余false；与旧版身份和划分保持一致。
- [x] 新 `training.dmd.gaze_batch init/run/publish` CLI 复用现有hash/锁/来源和失败恢复思路。新配置/清单schema拒绝旧缓存冒充；记录旧preparation来源hash而不改原文件。
- [x] run输出23维特征、四行为标签、gaze标签及实际PTS；额外数组完整性校验及failure排除；publish调用新版训练数据加载器和覆盖检查。
- [x] 测试源变更/缓存损坏/标签缺失/ROI未确认/路径逃逸/版本混用；模拟感知边界，真实标签→缓存→发布→CPU联合训练。

## Task 4：交付与回归

- [x] 同步模型设计、数据契约、全量指南、旧缓存迁移边界、权重与模型卡要求；明确105是现有清单预测数量而不是硬编码。
- [x] 独立规格审核，修复后独立质量审核；不将自审代替审核。
- [x] 运行 `.venv-training/Scripts/python.exe -m unittest discover -s tests -q` 和 `git diff --check`。
- [x] 记录本机CPU工程验证、真实样例标签验证、服务器NPU待验边界。给用户完整正式流程命令，不要求先运行旧8/12任务试跑。

基线：HEAD 08ad545；开始时仅已批准设计文档未跟踪，保留用户内容。已有依赖环境可用，不执行pip升级。每项测试须记录实际命令与结果；计划勾选以实现和验证为准。

完成记录：2026-09-21本机253项回归（251通过、2因Windows符号链接权限跳过），独立规格/质量审核通过；真实gaze标注帧解析、原preparation生成105任务和本机MediaPipe截图接口已检查。服务器全量与新版NPU训练尚未执行，见[工程验证](../../experiments/20260921-gaze-joint-training.md)。
