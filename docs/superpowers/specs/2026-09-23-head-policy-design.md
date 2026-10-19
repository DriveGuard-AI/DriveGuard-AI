# H 分支覆盖与短时缺失对照

用户已同意在逐窗口诊断后继续实施。旧 H 模型恢复了原来 47 个 F 不可用正例中的 38 个；剩余 199 个漏检中，82 个末槽有效但覆盖不足，106 个末槽无效，11 个 H 合格仍漏检。本轮仅验证 H 使用条件，不更改正式感知缓存或分类阈值。

采用新的显式模型种类 `relation_gaze_head_policy_v1`，在 model.head_policy 中保存 min_valid_ratio 与 max_gap_steps。旧 kind、旧配置和 checkpoint 保持严格末槽有效且覆盖 >= .5 的规则。两个独立候选：覆盖 .25 / gap 0；覆盖 .5 / gap 2。25 Hz 下 gap 2 表示距离最近同时三轴有效的采样槽 80 ms，不等于距原始视频测量的精确年龄。

仅依据预处理后的有效掩码计算窗口覆盖和末尾连续缺失槽数；全缺失用窗口长度作内部哨兵，报告输出空时间。H 的因果 TCN 继续接收原值与原掩码，不前向填充、不使用未来数据。末槽有效沿用原 F freshness；允许 gap 时用 1-gap/(max_gap_steps+1) 作为明确的间隔衰减分数。H 不合格时 embedding 和 quality 均置零。

训练监督、模型前向、预测、诊断必须使用同一 policy。新 kind 必须显式携带合法 policy，旧 kind 拒绝 policy。保持相同网络参数布局，使严格 policy 的初始化与前向可和旧模型逐值对比；不同配置间不能 resume，候选从头训练。

诊断新增 gap、current_valid、blocked reasons、policy 元数据以及按视角和 H 阻断条件分组的结果。提供单独只读 coverage audit 命令，读取 checkpoint 缓存，无须 NPU 推理或训练，统计各候选可使用的正负窗口与 gap 分布。覆盖变化不能宣称为召回提升。

验证包括 policy 参数拒绝、NumPy/Torch 一致、边界/全缺失/非同时三轴、缺失值不泄漏、因果性、默认精确兼容、训练 reload/resume 与预测诊断的一致性。CPU 测试不能替代真实 NPU 和验证集实验。
