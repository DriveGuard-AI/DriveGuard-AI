# 关系门控模型CPU工程验证

日期：2026-09-19；负责人：shadow；分支：model；范围为合成特征，不是DMD成绩。

## 交付与验证

- 新增四流关系门控模型，默认13,524个可训练参数；冻结感知模型不计入。
- v2配置/manifest/真实观测年龄导出、训练主体归一化、缺失依赖与100槽历史过滤、质量门控、共享CPU/GPU/NPU训练入口、checkpoint重载/续训已实现。
- 原105项测试全部保留，新23项测试包含模型10、数据6、训练5、导出2。
- .venv-perception（torch2.7.1+cpu）：128项测试通过，50.454秒。
- .venv-training（torch2.14.0+cpu）：128项测试通过，45.817秒。都是Windows CPU，不是远程NPU。
- 三入口的python -S --check-config通过，device_verified=false，不导入torch或初始化硬件。

## 可复现合成闭环

```powershell
.\.venv-perception\Scripts\python.exe -B -m training.relation.smoke --output-dir runs/relation-smoke-20260919
```

此目录已存在；再次执行需换新的输出目录，不覆盖原证据。

本机报告位于runs/relation-smoke-20260919/smoke_report.json（Git忽略）：2轮完成6次更新；续训至第3轮共9次更新，参数和history与连续3轮完全一致；best权重重载输出有限值。每轮训练记录2个全无可用输入窗口；head_yaw/pitch/roll保持unsupported。合成loss仅验证优化链路，不作为精度报告；checkpoint不作为公开正式权重。

## 审查与修复

独立规格审查和代码质量审查均通过，修复并保留以下回归：

1. 无效流不仅权重归零，其时序向量必须在门控输入前归零，避免通过其他流权重间接泄漏。独立复验权重/logit差异及该流输入梯度均为0。
2. 新包初始化不提前导入torch，恢复无依赖预检能力。
3. 正负标签预检必须扣除无输入窗口；否则只剩负例也可能启动训练。原始标签、不可用正例与实际损失覆盖分别报告。

独立检查确认整录制与窗口切片/65,536槽分块的历史mask一致。全量旧训练、感知和视频导出测试无回归。

## 边界

未运行DMD或真实驾驶视频，未安装依赖、连接服务器、执行GPU/NPU训练或导出ONNX/TensorRT。可选随机专家缺失增强、DMD标注映射、业务任务支持组合与风险告警适配待后续实施。既有tests/忽略规则保留，测试文件目前只在本机工作区；可分发的smoke命令不依赖tests包。代码尚未提交。

服务器训练方式见[关系门控运行指南](../training/RELATION_RUNBOOK.md)；模型结构见[关系门控设计](../architecture/RELATION_GATED_FUSION.md)。

