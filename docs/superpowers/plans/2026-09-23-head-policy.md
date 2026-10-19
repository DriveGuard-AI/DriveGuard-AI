# Head Policy Implementation Plan

Goal: 在不改变旧模型语义的条件下运行 H 覆盖门槛与短时缺失对照。

Architecture: dependency-free policy 校验、NumPy 掩码统计与 Torch 模型规则一致；新 kind 明确保存 policy，所有调用方沿用 checkpoint 配置。

Tech Stack: Python stdlib, NumPy, PyTorch, unittest.

- [x] 在 tests/test_head_policy.py 写行为测试并运行失败：覆盖 .25 的 25/24 槽边界，gap 2 的 2/3 槽边界，全无有效值，严格 policy 与旧模型逐值相等。
- [x] 新建 training/gaze/head_policy.py 参数校验；扩展 head_evidence.py/head_model.py；在 common/config.py/model_factory.py 注册新 kind；将 policy 传入 data.py/diagnostics.py，预测返回 gap 状态。
- [x] 添加只读 training/gaze/head_coverage.py，按视角/标签记录候选资格与 gap 分布。拒绝覆盖已有目录，不推理、不改缓存、不根据标签修改规则。
- [x] 添加配置 relation_gaze_head_coverage25.json 和 relation_gaze_head_gap80.json；文档给出先 audit 后分别从头训练、诊断的完整命令。
- [x] CPU 合成训练并验证 reload、同 policy resume、不同 policy resume 拒绝、诊断/预测/覆盖一致；运行 gaze 与 relation 回归、无 site-packages 配置检查、git diff --check。

Verification command: `.venv-training/Scripts/python.exe -m unittest discover -s tests -p test_head_policy.py -v`，再运行 test_gaze*.py 与 test_relation*.py。

2026-09-23 验证：7 项 head policy 测试、43 项 gaze 测试、23 项 relation 测试、5 项 config CLI 测试通过。候选配置在 python -S 下通过 npu_train --check-config，head_coverage --help 通过。独立审查发现高精度覆盖阈值的 NumPy/Torch 浮点边界差异，已统一使用十进制阈值换算的整数最少有效槽数；新增回归先失败后通过，复核无剩余实质问题。NPU 实际训练与真实缓存审计需在服务器运行。
