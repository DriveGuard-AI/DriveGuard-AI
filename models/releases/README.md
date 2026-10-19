# 正式权重交付目录

当前没有正式 DMD 权重，也没有公开下载链接。runs 中的合成 smoke checkpoint 仅用于工程测试，不作为模型成果发布。

正式版本由 shadow 提供：

- best.pt（选定模型权重）与 last.pt（可续训 checkpoint），或满足两者的同一工件；使用 CPU state_dict，不 pickle 完整模型。
- 对应 SHA-256、公共 JSON、特征/标签协议、normalizer、训练代码 revision、数据版本/切分摘要、运行环境与真实评估报告。
- 模型卡：四任务可用性、局限、GPU/NPU 各自验证状态、YOLO/MediaPipe/TCN 组合版本、来源与权重许可。
- 可访问的公开工件地址（Git LFS/Release/模型托管方式待实际交付选择）；不能用代码开源代替 checkpoint 开源。

训练环境路径与原始 subject 清单不直接公开：生成发布用去敏配置/摘要；去敏或重新打包后重新计算校验和并保留内部来源关联。

DMD 或上游权重许可未确定能否满足公开交付时，模型卡标记 pending_permission 并解决来源问题，不宣称已经开源。不自动发布任何文件，也不为示意填造权重或成绩。

当前加载方式与 best/last 区别见[训练使用说明](../../docs/training/RUNBOOK.md)。仓库不按 `.pt/.pth` 扩展名全局忽略；临时 runs 忽略，选定正式工件或可追溯下载索引才进入此目录。
