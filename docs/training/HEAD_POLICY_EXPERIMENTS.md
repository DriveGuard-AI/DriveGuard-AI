# H 分支使用条件对照

本轮只改变 H 使用条件。原缓存、四个原分组、gaze 条件、分类阈值、训练数据划分保持原定义。实验参数未在真实验证集上证明有效。

| 方案 | 模型 kind | 最低有效覆盖 | 允许末尾连续缺失 |
|---|---|---:|---:|
| 已训练的 H 基线 | relation_gaze_head_v1 | 50% | 0 槽 |
| coverage25 | relation_gaze_head_policy_v1 | 25% | 0 槽 |
| gap80 | relation_gaze_head_policy_v1 | 50% | 2 槽 / 80 ms |

每个候选只改变一项使用条件。末槽无效但允许短 gap 时，H gate 第二项质量按 `1-gap/(max_gap_steps+1)` 衰减；末槽有效沿用原 freshness。未补齐缺失值或掩码。三轴必须在同一槽有效，过期与 normalizer 不支持的数据仍然无效。模型继续读取带掩码的因果历史，`recent_head_supported` 明确表示来自最近历史而非当前头姿。

根据用户提供的 H 基线第 18 轮诊断 CSV，coverage25 将新增 98 个 body 可用窗口，其中 42 个是基线漏检正例、56 个是基线正确判负窗口；face 新增 12 个（3 个基线漏检、9 个基线正确判负）。这只是输入资格变化，不是新模型精度结果。gap80 的增量需要原逐槽掩码，旧 CSV 无法给出。

## 1. 先运行覆盖审计

在服务器仓库 `/mnt/workspace/DriveGuard-AI` 中执行：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.head_coverage \
  --checkpoint runs/relation-gaze-head-npu-001/best.pt \
  --view all \
  --output-dir runs/body-head-coverage-001
```

使用 CPU 读取 checkpoint、原特征和原 normalizer，不执行神经网络推理、训练或视频提取。不要传 `--backend`。目录必须不存在，已有报告不覆盖。

输出 `summary.json` 和 `windows.csv`：

- `variants.strict/coverage25/gap80` 分视角统计可用正例/负例/未知窗口；`newly_eligible_*` 是相对严格基线新增的输入资格，不是恢复检出的数量。
- `gap_bins` 分正负标签统计距离最近有效头姿槽的间隔，`none` 表示整窗口没有有效头姿。
- CSV 的 `h_gap_steps/h_gap_ms` 可按 `record_id + start_slot + end_slot` 与已有预测报告匹配。它是采样网格间隔，不是源视频测量精确年龄。
- 看新增负例数量及其分布，不能仅以新增正例多为依据选方案。两个候选单独评估，不在本轮直接组合。

## 2. 候选分别从头训练

保留已有 `relation-gaze-head-npu-001` 作为基线。使用相同种子、缓存和 20 轮预算；不从旧 checkpoint resume。

仅降低覆盖门槛：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze_head_coverage25.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --epochs 20 \
  --output-dir runs/relation-gaze-head-coverage25-npu-001
```

仅允许末尾缺失两槽：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze_head_gap80.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --epochs 20 \
  --output-dir runs/relation-gaze-head-gap80-npu-001
```

实际 manifest 路径须沿用原训练路径。单张 NPU 上先后运行两组。可先给训练命令追加 `--check-config`；改变 policy 或模型 kind 后续训会被拒绝，同一实验自己的 last.pt 可按原规则续训。

## 3. 用各自 best.pt 诊断

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.diagnostics \
  --checkpoint runs/relation-gaze-head-coverage25-npu-001/best.pt \
  --backend npu --view all \
  --output-dir runs/body-dist-head-coverage25-001
```

gap80 组将 checkpoint 替换为 `runs/relation-gaze-head-gap80-npu-001/best.pt`，输出使用 `runs/body-dist-head-gap80-001`。

诊断报告新增 `head_policy`、CSV 中的 `h_current_valid/h_gap_steps/h_gap_ms/h_block_reasons`，以及 `groups.view_h_evidence`。旧 checkpoint 也可以重新运行诊断，得到严格规则下的 gap 统计。不会将旧 checkpoint 的使用条件静默改为候选值。

评估仍使用最低验证损失选择 best.pt，并在同一批窗口、同一分类阈值下比较 body 分心召回、F1、误报数量以及 face 分心和 gaze 指标。剩余 82 个“末槽有效、覆盖不足”的旧漏检是否恢复，必须匹配预测 CSV 后才能判断。验证集已用于开发，最终泛化需使用未参与这些调整的数据。
