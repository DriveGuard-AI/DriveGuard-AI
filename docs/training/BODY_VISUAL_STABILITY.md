# Body 融合分支：归因与稳定性对照

第一轮验证集结果中，C best 在误报率不超过 3% 的诊断阈值下得到 105 TP / 34 FP，A 为 89 TP / 31 FP。但 A 是原联合模型，C 使用新二分类头，因此尚不能把全部收益归因于 RGB；C 的第 20 轮也明显差于第 3 轮。本轮补原特征输入对照、重复随机种子和容量对照。

## 实验配置

| 组别 | 输入 | 分类头宽度 | 每组种子 |
| --- | --- | --- | --- |
| D_context_control | 原模型 context；归一化 RGB 输入置零 | 64 | 42、43、44 |
| C_fusion | 原模型 context + RGB | 64 | 42、43、44 |
| E_fusion_small | 原模型 context + RGB | 32 | 42、43、44 |

共 **9 次新训练**，按种子逐组顺序运行；A 只评估一次。新训练均采用学习率 0.001、dropout 0.1、weight decay 0.0001、batch 64，最多 20 轮。E 只改变宽度，暂不同时改变正则化参数，便于判断效果来源。

D 与 C 的参数结构、初始化方式、样本及训练设置完全一致。D 在投影前把 RGB 输入置零，常量分支仍可学习偏置，但不包含图像信息。这比直接移除 RGB 分支更能控制结构差异。D 保留和 C 相同的 RGB 有效性条件，是输入消融对照，不是专门为 RGB 不可用时设计的部署模型。

统一以验证 BCE 早停：连续 3 轮相对早停参考值没有超过 0.0001 的下降则停止。严格最低 BCE 仍用于保存 best.pt；微小下降可以更新 best，但不一定重置早停计数。每个 epoch 保存计数器、参考值、随机状态及停止状态，断点恢复会沿用这些状态。

本轮仅加载上一轮的 RGB 缓存、冻结 context 和训练集归一化，不重新下载权重、不解码视频、不重新提取特征。测试集不用于训练、早停或阈值选择。

## 服务器启动

同步更新代码后，在 `/mnt/workspace/DriveGuard-AI` 运行。不需要再次执行安装脚本。

```bash
cd /mnt/workspace/DriveGuard-AI
mkdir -p runs
nohup bash scripts/server/train_body_visual_stability.sh \
  > runs/body-visual-stability-001.log 2>&1 < /dev/null &
echo $!
```

默认读取：

```text
runs/body-visual-comparison-001/prepared-001/prepared.json
```

如果上一轮恢复后实际使用 `prepared-002`，设置 `PREPARED_DIR=runs/body-visual-comparison-001/prepared-002` 再启动。该路径可在上一轮 `comparison/plan.json` 的 `provenance.preparation` 中核对对应缓存/来源，或查看各 prepared 目录的完成清单；不能使用只有部分文件的目录。

可通过 `PYTHON_BIN`、`PREPARED_DIR`、`OUTPUT_DIR`、`DEVICE_INDEX`、`EPOCHS`、`BATCH_SIZE`、`PATIENCE`、`MIN_DELTA` 环境变量改变配置。默认 Python 是 `.venv-visual-npu/bin/python`，训练后端 NPU。命令行 `--seeds 42 43 44` 可覆盖默认种子列表；变更配置须使用新输出目录。

```bash
tail -f runs/body-visual-stability-001.log
```

只有确认前一进程退出后才续跑同一目录：

```bash
nohup bash scripts/server/train_body_visual_stability.sh --continue-run \
  >> runs/body-visual-stability-001.log 2>&1 < /dev/null &
echo $!
```

已完成或已早停的组不会重新训练。配置或缓存身份变化会拒绝续跑；未完成训练从有效 epoch 检查点恢复。

## 结果与判读

输出目录：`runs/body-visual-stability-001/`。

- `result.json`：最终完成标志，`status=complete`。
- `progress.json`：已完成组数；完成时是 9/9。
- `comparison.json` / `comparison.csv`：A 一行加 9 次训练各 best/last，共 19 行。
- `aggregate.csv`：各组 best 的均值、标准差、最小/最大值，以及配对差值。
- 各 `*-windows.csv` / `*-summary.json`：逐窗口及各被试、录像的细分数据。
- `D_context_control-seed42/train-*/history.json` 等：各次训练曲线和早停结果。

优先判断 C 相对 D 是否跨种子稳定提高 `fpr03_tp` / `fpr03_recall`，而不只是某个种子恰好更好。`paired_best.C_minus_D` 给出同一种子的差值，`E_minus_C` 判断缩小容量是否有益。也需检查实际误报数和默认阈值表现；同为 3% 预算并不意味着最终误报数完全相等。

`validation_fpr03_groups` 在各被试、录像及固定头姿可用/不可用组上使用同一个全局阈值，**不单独为每个组选择阈值**。因此可以直接回答：低误报条件下，134 个原头姿缺失正窗口究竟找回多少。

均值/标准差只描述不同随机种子的训练波动，标准差采用总体公式（ddof=0），不是跨人群置信区间。当前验证只有两名被试，且已多次用于选型；本轮结果仍是开发指标。方案和阈值冻结后，再用未参与选型的数据验证，不能把验证集阈值直接作为已验证部署阈值。
