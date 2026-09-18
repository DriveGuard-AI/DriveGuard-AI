# shadow 训练工程首批实施计划

日期：2026-09-18。执行：shadow，当前任务内逐步实施；分支 `model`。

目标：数据未就绪时先完成 TCN 设计，再实现训练闭环。NPU 软件版本已确认并通过小型 Linear 测试，不追加远程硬件排查。用户已确认完整四任务方向；Gaze 两个分片、Drows 一个分片、Dist 一个分片在下载，数据任务暂搁置。

方案：共享 PyTorch 模型/训练循环 + CUDA/torch_npu 双入口 + JSON 配置 + 外置特征清单。首版单卡 FP32；YOLO、原始 DMD 转换、ONNX 与真实训练作为后续独立任务。

- [x] 文档对齐：Requirements/Spec/Architecture/Testing/Governance 引用 shadow 职责；之后创建 model 分支。
- [x] 编写 [模型设计](MODEL_DESIGN.md)与结构图：方向已确认，详细设计待用户评审，不代表模型实现完成。
- [ ] 数据契约：manifest 引用录制级特征数组和窗口范围，避免重复落盘重叠窗口；单窗 `[100,24]`，四类别 target 与 label_mask；测试跨子集主体泄漏、非法路径和维度。当前不扫描真实数据。
- [ ] 配置与双入口：公共 JSON + gpu/npu 小配置；CLI `--check-config` 不依赖 torch 或数据；GPU/NPU 不可用时失败。用 unittest 覆盖参数校验与路径优先级。
- [ ] TCN：因果膨胀 Conv1d 残差块，最后时间步分类；测试输出形状、未来输入不影响过去、有限梯度。
- [ ] 训练与 checkpoint：Masked BCEWithLogitsLoss、Adam、逐类聚合验证损失选 best、epoch 级 resume、CPU state_dict；合成数据两 epoch 和恢复到第三 epoch 验证工件与指标。未知标签不参与对应损失，全未知 batch 不更新。
- [ ] 文档与交接：RUNBOOK、模型结构 Markdown、实验结果页和公开权重目录；不填虚构跑分或下载链接。
- [ ] 完成验证：unittest、双入口预检、SQLite/API 不受本轮影响；git diff --check、忽略规则和 Markdown 链接检查。

校验命令：`python -m unittest discover -s tests -v`，`python -m training.temporal.gpu_train --check-config`，`python -m training.temporal.npu_train --check-config`。

恢复方式：从未勾选项继续；仅修改本计划对应的训练与文档文件。当前无运行服务、训练数据或生产库变更，不需要数据库迁移。实际硬件和权重发布验证与代码 smoke 分开记录。
