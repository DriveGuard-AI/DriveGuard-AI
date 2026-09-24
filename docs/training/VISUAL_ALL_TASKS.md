# C_fusion 组合系统全任务评估

## 评估的是什么

将原 gap80 多任务模型与 C_fusion 组合：仅 body 视角的 distraction 输出
使用 C_fusion，其余行为、face distraction、gaze 仍使用原模型。
这不是重新训练，也不是把单输出 C_fusion 当作一个五任务模型。

同一次评估包含：

- 原 gap80 基线。
- C_fusion seed42 best，logit 阈值 0（此前默认阈值 F1 最好的候选）。
- C_fusion seed43 best，logit 阈值 0，作为同阈值对照。
- C_fusion seed43 best，使用原稳定性报告中的 validation_fpr03 阈值。

最后一种阈值只对 body distraction 生效，并绑定原报告的权重哈希和窗口。
body 误报率约束不等同于全视角分心误报率约束；face 仍采用基线阈值。
模型选择及低误报阈值都用过验证集，这份评估不是独立测试集的泛化成绩。

## 服务器运行

同步新增的 Python 文件和服务器脚本后，在项目根目录运行：

```bash
bash scripts/server/evaluate_visual_all_tasks.sh
```

该命令前台执行一次评估，不会训练、下载权重或重新提取视频特征。
需要现有 prepared 缓存、原 gap80 权重、seed42/43 权重及其 SHA256 sidecar、
原始结构化特征缓存，以及稳定性实验的 comparison.json。路径来自已有
prepared.json 和 comparison.json，默认适用于原服务器目录。

如需后台执行：

```bash
mkdir -p runs
nohup bash scripts/server/evaluate_visual_all_tasks.sh \
  > runs/visual-all-tasks-001.log 2>&1 &
echo $!
```

已有输出目录不覆盖。再次运行时指定新目录：

```bash
OUTPUT_DIR=runs/visual-all-tasks-002 bash scripts/server/evaluate_visual_all_tasks.sh
```

`PYTHON_BIN`、`PREPARED_DIR`、`STABILITY_REPORT`、`BACKEND`、`DEVICE_INDEX`
和 `BATCH_SIZE` 均可通过环境变量覆盖。

## 如何看结果

默认输出目录 `runs/visual-all-tasks-001/` 中：

- `report.json`：完整指标、来源信息、一致性审计和成对错误统计。
- `metrics.csv`：便于对比的各任务及分组指标。
- `predictions.jsonl`：逐窗口预测，供定位新增检出、漏检与误报。

先检查完整状态、窗口匹配和未替换输出的一致性审计，再比较：

1. 五任务整体，以及 body/face 分视角指标。
2. 原始已标注数量、有效预测数量与覆盖率，避免把不可用样本藏在分母外。
3. 分心的全正例检出率、F1、误报率，并对齐阈值口径。
4. 每个受试者的结果，避免总体增长掩盖个别受试者退化。
5. body 有/无有效头姿两组中，新增检出、丢失检出、新增误报和消除误报。

RGB 输入不可用时明确回退到 gap80，并计数；不会把无效 RGB 输出当作正常分数。
gaze 没有有效标签的分组保留空指标，不填成 0% 准确率。

本地只验证合成数据上的执行和一致性；真实服务器结果以完成后的报告为准。
历史汇总计数的合成预览见 `VISUAL_ALL_TASKS_HISTORICAL.md`。

2026-09-24 本地验证：33 项相关 unittest 通过（其中新评估器8项），CLI help、
Git Bash 语法检查通过；未在真实NPU或真实验证集上重跑。复现测试：

```powershell
.venv-training/Scripts/python.exe -m unittest tests.test_visual_all_tasks tests.test_visual_data tests.test_visual_training tests.test_visual_stability tests.test_visual_early_stop
```
