# TCN 首轮工程验证（非模型跑分）

日期：2026-09-18 · 负责人：shadow · 分支：model。

结论：首个 TCN 训练工程增量通过本地 CPU 验证与独立规格/质量审查。没有使用 DMD，没有验证目标 GPU/NPU；下述 loss 不能作为驾驶行为识别精度。

## 1. 可追溯环境

| 项目 | 本次记录 |
| --- | --- |
| Git HEAD | `fe3c5b301eeff47b2391b33c86d04fd3aa7efbea`，dirty=true；本增量尚未提交，该提交号不包含新实现 |
| 系统 | Windows 11 / CPU，smoke 显式 1 个 torch 计算线程 |
| Python / PyTorch / NumPy | 3.12.14 / 2.14.0+cpu / 2.3.5；复用已有忽略环境，未安装或升级依赖 |
| 模型 | 12 特征 + 12 mask，100 槽；5 个 32 通道因果残差块，31,204 参数，四个独立 logit |
| 训练 | FP32 / Adam / lr=0.001 / weight_decay=0.0001 / dropout=0.1 / seed=42；完整解析值保存在工件 config.json |
| 数据 | `training.temporal.smoke` 自动生成四个虚拟主体，每录制 125 槽，train/val 各两个主体，无测试集、无真实标签 |
| 输出根目录 | `runs/temporal-smoke-20260918/`，Git 忽略；含数据、initial、resumed 与 smoke_report.json |
| 数据追溯 | manifest/profile/label_mapping 与每个数组的 SHA-256 记录在 initial/resumed 的 data_summary.json；工件包含副本 |
| 耗时 | 完整 unittest 报告 7.956 秒；smoke 工具调用墙钟约 3.69 秒，仅作工程记录，不是模型/设备性能基准 |

## 2. 执行与结果

从仓库根目录、已有本地环境运行：

```powershell
.venv-training/Scripts/python.exe -B -m unittest discover -s tests -v
.venv-training/Scripts/python.exe -B -m training.temporal.smoke --output-dir runs/temporal-smoke-20260918
```

上述 smoke 目录已经存在，复跑请换新名称；程序拒绝覆盖既有目录。

- 29 项测试通过：配置/无依赖双入口 5 项、模型/损失 9 项、数据 7 项、训练恢复 7 项、smoke 1 项。
- 已测因果性、部分标签梯度、训练集唯一时间槽归一化、主体隔离、路径逃逸拒绝、未知批次跳过及覆盖计数、验证聚合与 batch 划分无关。
- 已测可信工件的 checksum / `weights_only=True` 加载、拒绝续训时变更数据和超参数、拒绝覆盖已有目录。
- 连续三轮与“两轮 + 恢复第三轮”的 CPU 模型参数逐位一致，history 相同；不推导跨设备或跨 PyTorch 版本逐位一致。
- 规格审查发现跳过未知 batch 时遗漏覆盖计数，已先补失败回归测试，再修复并复核通过；代码质量独立复跑也为 29/29。
- GPU/NPU 双入口 `python -S ... --check-config` 均返回 config_valid、device_verified=false；git diff --check 通过，数据/环境/runs 忽略，models/releases 不整体忽略。
- 文档本地链接、14 个 Spec 场景与 PPT 14 页映射、内存 SQLite 五表/约束、OpenAPI 10 个操作/62 个静态引用检查通过；不是完整 OpenAPI schema 验证或图示渲染验收。

独立 smoke 工件结果：

| 阶段 | 已完成 epoch | global_step | best epoch | 合成 validation loss |
| --- | --- | --- | --- | --- |
| initial | 2 | 4 | 2 | 0.696955107152462 |
| resumed | 3 | 6 | 3 | 0.692526675760746 |

resumed/best.pt SHA-256：`799fa4bb8fcb839af3a67ba213a17d5e62b12b283ea480bbe6ddc68e4badb249`。

resumed/last.pt SHA-256：`805202f29bb6fa779252974e20e29825c38e9c1d1ed6d910005f45b9b992a81b`。

这些是本次合成工件的哈希；重新运行的时间戳等元数据会变化，不要求哈希完全相同。合成工件不复制到正式权重目录、不宣称 DMD 权重已开源。

## 3. 未验证项与下一步

- 目标 PyTorch 2.7.1、CUDA GPU、昇腾 910B3/CANN 9.0.0/torch_npu 2.7.1.post4 的实际训练及跨设备恢复未验证；配置预检只证明格式可读。
- YOLO、MediaPipe、手物交互特征、完整 profile、实时质量适配、融合、前后端与部署尚未实现；24 通道仅基线。
- DMD 四任务分数、独立测试集、事件误报/延迟、端到端 FPS 全部未测。数据任务继续暂停。
- 正式权重尚未产生或发布；来源许可未核实前保留 `pending_permission`。公开交付要求见[发布目录](../../models/releases/README.md)。
- 下一步按[完整实施计划](../training/IMPLEMENTATION_PLAN.md)推进感知与交互适配；真实训练前按[数据契约](../training/DATA_CONTRACT.md)接入特征，操作见[训练指南](../training/RUNBOOK.md)。
