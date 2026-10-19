# Body 分心：四组自动对照

沿用现有特征缓存，一次命令顺序训练四组，并自动评估每组 best/last、输出同一张比较表。候选参数仅用于实验，尚未证明能提高真实准确率。

| 组 | 唯一改动 | 检验的问题 |
|---|---|---|
| baseline | coverage25，当前头姿有效，标准随机顺序 | 同轮数、同种子的对照 |
| sampling | 可监督的 body 分心正例且 H 不可用时，采样权重为 3 | 加强这类样本的学习能否减少漏检 |
| face_dropout | 有 O/HO 输入的 body 训练窗口，以 0.3 概率移除全部 F/FO 证据 | 缺少人脸信息时能否更好使用其他输入 |
| gap80 | 保持 coverage25，允许最近有效头姿距当前最多 2 槽（80ms） | 短暂缺失是否导致可用头姿被拒绝 |

每组保留相同模型宽度、数据、随机种子、batch size、学习率和总轮数。sampling 有放回抽取原窗口数，3 是相对采样权重，**不表示必定重复三次**。每轮样本数不增加，但实际梯度步数由可监督批次决定并记录在 history 中。没有把三项修改一起打开，以便区分收益来源。

已有 coverage25 最后一轮诊断中，152 个 H 不可用的 body 正例全部漏检：150 个只剩 O，2 个有 O+HO。上述两项训练干预不能补出缺失的头姿或身体运动信息，因此既要观察漏检是否减少，也要观察误报是否增加。这里的 O 是 phone/drink_object confidence，不是完整身体视觉表示。

## 运行

在服务器 `/mnt/workspace/DriveGuard-AI` 更新代码后运行：

```bash
bash scripts/server/train_body_dist_suite.sh
```

使用原已验证的 NPU Python。不会启动 MediaPipe、YOLO 或重新提取缓存。四个训练进程顺序运行；每个训练完成后依次运行 best/last 诊断，避免同时争抢同一块 NPU。训练日志每轮写出进度。

只检查配置、不导入 torch、不创建输出目录：

```bash
bash scripts/server/train_body_dist_suite.sh --check-config
```

若希望先做较短的统一初筛，可明确改变所有组的轮数及目录；短跑结果不能与旧 20 轮直接比较，也不应仅根据早期排序淘汰收敛较慢的组：

```bash
EPOCHS=10 OUTPUT_DIR=runs/body-dist-comparison-screen-001 \
  bash scripts/server/train_body_dist_suite.sh
```

默认仍为每组 20 轮。指定一部分组、切换设备、准备配置：

```bash
bash scripts/server/train_body_dist_suite.sh --variants baseline sampling
bash scripts/server/train_body_dist_suite.sh --prepare-only
```

独立运行入口可通过 `--config` 指定基线，必须是 coverage25/零间断、无已开启干预的 head-policy 模型。`--backend cpu|cuda|npu`、`--device-index` 可覆盖设备；CPU/GPU 主机通过 `PYTHON_BIN` 指定对应环境。

## 输出与判断

默认目录 `/mnt/workspace/DriveGuard-AI/runs/body-dist-comparison-001`：

- `comparison.csv`：各组 best/last 的 body、face、H 可用/不可用分组混淆计数、精确率、召回率、F1、误报率、输入覆盖，以及 gaze/其他行为指标。
- `comparison.json`：同一比较的完整报告；含按 F/O/HO/FO 可用性组合细分的预测、特征覆盖和门控均值。
- `<组>/train-001/body_dist_audit.json`：原始 train/val 输入覆盖及可监督正负样本分布，采样加权数量、增强候选数量。只用训练标签决定采样权重。
- `<组>/train-001/history.json`：完整训练历史；face_dropout 组含每轮实际增强数量。
- `<组>/best-diagnostics-001/`、`last-diagnostics-001/`：已有格式的 `summary.json` 和 `windows.csv`。
- `<组>/train.log`、`best-diagnostics.log`、`last-diagnostics.log`：各阶段日志。

优先比较 body 的 `all_positive_detection_rate`、F1 和误报率，再看 face、gaze、其他行为是否退化。`recall` 分母只包含可预测正例；`all_positive_detection_rate` 分母包含无输入正例，避免掩盖覆盖损失。H 分组随 gap80 政策变化，因此不能把不同组的 H 子集当作完全相同的样本集合。完整 body/face 比较会校验数据 fingerprint、窗口身份、标签和阈值一致。

best 沿用总验证 loss 选择，last 是最后一轮，不会按本次结果自动更改阈值、挑选中间轮权重或部署赢家。诊断覆盖 val/all views/all subsets；与旧仅 distraction subset 的报告比较前须核对样本范围。训练/验证集已经用于开发，最终泛化需要独立受试者测试集。

## 中断后继续

保持原命令的轮数、变体、目录等参数不变，追加：

```bash
bash scripts/server/train_body_dist_suite.sh --continue
```

已完成阶段先校验配置与检查点 checksum 后复用。未完成训练若有有效 last.pt，会在新的 `train-002` 等目录中续训；即便续训进程在写出首个新 checkpoint 前再次失败，也保留原续训来源。未完成的诊断写入新的编号目录。原报告和权重不会被覆盖，顶层比较报告随进度更新。

如果在最后一轮 checkpoint 与 result.json 发布之间中断，入口会明确报错要求检查最终发布状态；不会把可能尚未完整发布的 best/history 冒充完整训练。若无可用 checkpoint，失败阶段从头运行到新的目录。改变 seed/epoch/变体等配置则必须使用新输出目录；需要延长既有训练时，使用原训练入口的 `--resume` 到新目录。

## 实现边界

新增可选 `training.body_dist_experiment`，旧配置无需添加且原逻辑不变。face_dropout 完整清除 F 与 FO 的值、掩码、质量、资格，H 及 gaze 由现有规则自动失效，O/HO 保留；仅作用于有替代输入的 body 训练窗口，不根据标签选择增强对象。缓存、验证输入及标签保持原样。CPU torch RNG 和 loader generator 进入原有 checkpoint，已用合成数据验证精确续训。

若训练集中没有符合条件的加权样本，sampling 保留原随机遍历，审计中的 `sampling_active=false` 明确表示此干预未实际生效；不会仅因空目标集合而改成均匀有放回采样。

本地验证：10 项新增测试通过，涵盖四组完整合成训练及 8 份 best/last 诊断、重复继续不改动权重、重试来源保留、失败报告、掩码语义、标签排除、精确续训和中性配置兼容；另有 45 项既有训练、配置、head-policy 与诊断测试通过。诊断测试使用 `python -m unittest discover -s tests -p test_gaze_diagnostics.py` 运行，以满足其已有 fixture 导入方式。NPU 实机训练尚未运行，合成数据测试不证明真实识别效果改善。
