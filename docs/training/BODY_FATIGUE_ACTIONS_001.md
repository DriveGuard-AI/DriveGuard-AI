# Body 疲劳：动作监督与事件判定

上一轮 D_tune32-seed43-best 为 15/61 检出、3/215 误报。本轮改变监督方式：
学习每个采样帧的眼睛开闭和哈欠，比较直接疲劳输出与按预测动作计算的持续事件。
这些都是原验证集上的探索结果，不是独立测试或全任务模型的部署效果。

## 输入与实验

使用 `runs/body-fatigue-head-adaptation-001/comparison.json` 中的
`D_tune32-seed43-best`，校验检查点、来源、像素缓存、原始标注 SHA256 并复现发布指标。
从检查点定位已完成的 `head-frames`。不下载权重、不重解码视频、不重跑 MediaPipe。
启动时会读取缓存/视频以校验哈希，并打印阶段进度，校验不等于重新提取。

默认运行 **4 次训练**，同一 D43 热启动，每组 seed42/43：

| 组 | 学习目标 | 输出 |
|---|---|---|
| A_direct_control | 原疲劳 BCE | 直接疲劳分数 |
| B_action_supervision | 疲劳 BCE + 0.5×闭眼 BCE + 0.5×哈欠 BCE | 直接分数、事件分数 |

事件输出复用 B 的同一份权重，不是额外训练一个模型。默认各训练 6 轮，耐心值 7，
使对照和辅助组得到相同训练轮数；如修改轮数，也应让耐心值大于轮数保持这一对照。
编码器只微调 layer4 卷积，BN 与原特征归一化冻结。学习率沿用上一轮。
两组在相同的监督窗口并集上采样；A 不计算辅助损失。完全未知窗口不产生梯度。
每个累积组中，各项 BCE 分别除以自身实际有效监督数量，尾批也按实际数量处理。

B 的 `best.pt` 根据眼睛/哈欠验证 BCE 的均值选择，直接/事件两个输出共用此检查点。
另存 `fatigue-best.pt` 按原疲劳验证 BCE 选择，用于和 A 做相同选模指标的对照。
保留初始候选和 `last.pt`；不要把动作选模指标变化造成的差异全部归因于辅助监督。

## 标签和时间约束

- 疲劳目标仍是原窗口末尾标签，所有已知/未知验证窗口和原分母都保留。
- 辅助标签由原 OpenLABEL 动作、相机偏移、真实源帧 PTS 对齐；先重编译并逐项核对
  原疲劳数组，再使用新动作标签。不会把窗口末尾标签复制给窗口内所有帧。
- 明确 open/close 才监督眼睛；冲突、缺失、undefined 均未知。
- 哈欠正例来自原哈欠标注。无哈欠仅在明确睁眼且该时刻没有哈欠标注时作为负例，
  与原疲劳负证据约定一致。单纯缺少哈欠标注、闭眼未标哈欠等情况仍未知。
- 短暂闭眼可以提供闭眼监督，即使尚未满足疲劳时长。distraction/gaze 记录不生成
  新的闭眼/哈欠监督；原有疲劳监督照常使用。未知不被当作负例。
- 原始标签和像素源时间必须匹配；相机偏移、来源改变或旧疲劳标签不能复现即停止。

事件推理只用图像预测、源 PTS 和缺帧信息，不读真实动作标签，也不读取未来帧。
眼睛/哈欠阈值预先固定为 logit0。闭眼累计时长沿用原数据的 `closed_eye_min_ms`。
从录像开头按时间顺序累计，录像间重置；源帧间隔超过原 80ms 规则或像素失效就重置。
短暂预测睁眼也重置，避免把多个普通眨眼拼成持续闭眼。事件判定为预测哈欠或持续闭眼。
在原窗口末尾读取结果；可用性仍遵循原整窗像素有效条件。

8Hz 是采样观察，不保证看到采样间所有眨眼，闭眼起点也有量化误差。
本轮没有阈值拟合；报告里的 FPR03 扫描只作诊断，不能作为独立泛化结果或部署阈值。
低质量侧脸仍可能限制动作识别；这轮用于区分监督问题和输入信息不足，不承诺提升。

## 启动与继续

代码同步后，在服务器项目根目录执行：

```bash
mkdir -p runs
nohup bash scripts/server/train_body_fatigue_actions.sh \
  > runs/body-fatigue-actions-001.log 2>&1 &
echo $!
```

查看进度：

```bash
tail -f runs/body-fatigue-actions-001.log
```

同参数异常中断、确认原进程结束后：

```bash
nohup bash scripts/server/train_body_fatigue_actions.sh --continue-run \
  >> runs/body-fatigue-actions-001.log 2>&1 &
echo $!
```

完成的训练验证后复用；中断训练从校验通过的 last.pt 恢复到新 train-NNN 目录，
保留旧尝试与 epoch 级 RNG/优化器状态。改变路径/参数请使用新 `OUTPUT_DIR`。
可以通过 `ADAPTATION_REPORT`、`VARIANT`、`DATASET_ROOT`、`PYTHON_BIN` 指定原输入。
新检查点格式为 `fatigue_actions_v1`，需 `load_actions` 加载，不可直接替换旧组合模型。

## 查看和回传

输出目录 `runs/body-fatigue-actions-001/`：

- `comparison.json` / `comparison.csv`：原 D43，以及 A 的 best/last、B 的
  best/fatigue-best/last 各自直接/事件输出，默认共 17 行。
- `by-subject.csv`：分人疲劳指标。
- `by-subtype.csv`：整体及分人的闭眼、哈欠、两者重叠、未能归类的正例检出数；
  同一人员范围内子类互斥，分母可加总，`subject_id=all` 为整体。
- `action-metrics.json`：眼睛、哈欠的 TP/FP/FN/TN/召回/F1；按唯一采样帧统计，
  不重复计算重叠窗口。训练 history 的辅助 BCE 是窗口帧统计，两者单位不同。
- `action-label-audit.json`：来源哈希、训练/验证每条录像的动作正负例与未知数量。
- 各候选 `*-windows.csv`：原窗口标签、预测及窗口末尾的正例子类。

优先回传 comparison.json、by-subject.csv、by-subtype.csv、action-metrics.json。
重点看：辅助任务能否识别动作、哪类事件漏检、B 是否超过相同选模指标的 A，
以及 subject7/14 是否同时改善。保留原 61 正例/215 负例（实际以原准备数据为准），
用相同误报预算比较；本轮不评估或替换其他任务。
