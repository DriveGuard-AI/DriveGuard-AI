# 闭眼分支分数诊断

这一步不训练模型。复用 `body-fatigue-actions-001` 中 seed42 的 `best.pt` **事件读出**，先重现原报告，再判断闭眼漏检发生在帧分数还是连续时长判定。对应已运行结果为 19/61 TP、0/215 FP；程序从原报告读取并核对，不硬编码这些计数。

## 运行

在服务器仓库根目录，拉取包含本工具的提交后运行：

```bash
mkdir -p runs
nohup bash scripts/server/diagnose_body_fatigue_eyes.sh \
  > runs/body-fatigue-eye-diagnostics-001.log 2>&1 &
echo $!
tail -f runs/body-fatigue-eye-diagnostics-001.log
```

这是 **1 次诊断、0 次训练、5 组事件读出对照**。需要保留原 action comparison、checkpoint 与 SHA256 sidecar、原 prepared/cache、head-frames 和 Dataset 标注。不会重新下载权重、解码视频或提取 MediaPipe。使用现有 `.venv-visual-npu/bin/python`；可用 `PYTHON_BIN` 环境变量覆盖。

更改路径示例：

```bash
ACTION_REPORT=runs/body-fatigue-actions-001/comparison.json \
OUTPUT_DIR=runs/body-fatigue-eye-diagnostics-002 \
nohup bash scripts/server/diagnose_body_fatigue_eyes.sh \
  > runs/body-fatigue-eye-diagnostics-002.log 2>&1 &
```

输出目录必须是新目录。中断后使用新编号，原缓存可继续复用；没有 `--continue-run`。只有生成 `summary.json` 且 `status=complete` 才代表通过全部检查。校验发现 checkpoint、标注、采样或指标不一致时会报错，不自动放宽。

## 输出

默认目录 `runs/body-fatigue-eye-diagnostics-001/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 原结果重现、正负分数分布、ROC-AUC/AP、按人结果、五组对照及阈值来源 |
| `comparison.csv` | 五组对照的窗口 TP/FP/FN、召回率、误报率 |
| `by-subject.csv` | 每个人的窗口指标 |
| `by-subtype.csv` | 纯闭眼、纯哈欠、重叠及未归类正例的检出 |
| `scores.csv` | train/val 分开的帧级统计、每人 AUC/AP 和固定零阈值混淆计数 |
| `score-curves.csv` | 帧级阈值扫描；相同分数不可拆开选择 |
| `frames.csv` | 每个唯一采样帧的原始 eye/yawn logit、标注、掩码、slot/PTS；不重复累计重叠窗口 |
| `*-windows.csv` | 每组对照的全部原窗口和预测 |

先发回 **summary.json、comparison.csv、by-subtype.csv**。若需定位具体人或时间点，再查其他 CSV。

## 五组对照与解释

1. `original_event`：原始闭眼和哈欠事件读出，必须与已发布结果一致。
2. `yawn_only`：关闭闭眼分支，核实已有检出是否依赖哈欠。
3. `eye_only`：关闭哈欠分支，检查闭眼的实际贡献。
4. `train_eye_fpr03`：只用训练集已知眼部标签选帧级 FPR≤3% 的闭眼阈值，再固定应用到验证集。
5. `train_eye_balanced`：只用训练集选帧级 balanced accuracy 最高的闭眼阈值，再固定应用到验证集。

后两组重新计算闭眼持续时长，哈欠阈值和原始闭眼时长政策不变。它们是**探索性诊断**：模型已经拟合训练集，checkpoint 又由验证集选出，因此不能称为独立校准或独立测试。帧级 3% 误报预算也不保证窗口级误报≤3%；必须检查实际 `comparison.csv`。

unknown 和无效帧不进入正负分数统计，但全部原窗口保留在事件评估中。只有一类标签时 AUC 和阈值为 null；阈值 null 表示不接受任何闭眼阳性。`eye_known` 是原标注掩码，结合 `valid` 才能计入帧级评价。

若验证集正负眼部分数排序良好、只在零阈值下漏检，下一步才考虑独立校准。如果训练分得开而验证混杂，优先处理泛化；如果两边都混杂，再做原视频高分辨率局部眼部输入与独立眼部分支对照。不能仅凭单一 AUC 值自动决定方案，也不能用哈欠改善掩盖纯闭眼仍然零检出。

本工具不会替换任何 pt、修改部署阈值或改变 gaze/dist/phone 等其他任务输出。
