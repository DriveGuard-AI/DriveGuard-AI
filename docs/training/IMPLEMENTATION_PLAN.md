# shadow 训练工程首批实施计划

日期：2026-09-18。执行：shadow，当前任务内逐步实施；分支 `model`。

目标：在真实数据、服务器软件版本尚未到位时，交付能够独立验证的轻量 TCN 训练闭环，避免先花费完整数据训练成本。

方案：共享 PyTorch 模型/训练循环 + CUDA/torch_npu 双入口 + JSON 配置 + 外置特征清单。首版单卡 FP32；YOLO、原始 DMD 转换、ONNX 与真实训练作为后续独立任务。

- [ ] 文档对齐：Requirements/Spec/Architecture/Testing/Governance 引用 shadow 职责；之后创建 model 分支。
- [ ] 数据契约：JSON manifest 引用每个窗口 `.npy`，形状 `[T,F]`；窗口标签为四类别 multi-hot；主体切分泄漏、非法路径和维度有失败测试。
- [ ] 配置与双入口：公共 JSON + gpu/npu 小配置；CLI `--check-config` 不依赖 torch 或数据；GPU/NPU 不可用时失败。用 unittest 覆盖参数校验与路径优先级。
- [ ] TCN：因果膨胀 Conv1d 残差块，最后时间步分类；测试输出形状、未来输入不影响过去、有限梯度。
- [ ] 训练与 checkpoint：BCEWithLogitsLoss、Adam、验证损失选 best、epoch 级 resume、CPU state_dict；合成数据两 epoch 和恢复到第三 epoch 验证工件与指标。
- [ ] 文档与交接：RUNBOOK、模型结构 Markdown、实验结果页和公开权重目录；不填虚构跑分或下载链接。
- [ ] 完成验证：unittest、双入口预检、SQLite/API 不受本轮影响；git diff --check、忽略规则和 Markdown 链接检查。

校验命令：`python -m unittest discover -s tests -v`，`python -m training.temporal.gpu_train --check-config`，`python -m training.temporal.npu_train --check-config`。

恢复方式：从未勾选项继续；仅修改本计划对应的训练与文档文件。当前无运行服务、训练数据或生产库变更，不需要数据库迁移。实际硬件和权重发布验证与代码 smoke 分开记录。
