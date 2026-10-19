# Body 疲劳视觉训练 001

这次训练疲劳专用的 RGB 时序分类器，直接复用已经提取并校验过的 RGB 缓存。
不重新下载 ResNet18、不重新解码视频、不修改原始特征缓存或现有全任务模型。

`0/61` 在感知回放报告里表示正例窗口的 F 组可用性，不等同于疲劳分类器 TP。
新分支是否能预测只取决于窗口的 8 个 RGB 样本是否有效，不依赖 F、头姿或双眼关键点。
训练和评估都使用 fatigue 列及其 known mask，未知标签不作为负例。

## 服务器启动

在 `/mnt/workspace/DriveGuard-AI` 仓库根目录、model 分支执行。先确认拉取成功：

```bash
git pull --ff-only origin model
```

再启动后台训练：

```bash
nohup bash scripts/server/train_body_fatigue_visual.sh \
  > runs/body-fatigue-visual-001.log 2>&1 &
echo $!
```

默认读取 `runs/body-visual-comparison-001/prepared-001` 的元数据，定位原始 RGB 缓存、
特征 manifest 和 gap80 checkpoint。需要保留这些输入。若原准备目录是其他编号，
用环境变量 `PREPARED_DIR` 指向该目录。Python 默认 `.venv-visual-npu/bin/python`。

共 **3 个方向 × 2 个种子（42、43）= 6 次顺序训练**，在 NPU 0 上逐个执行：

| 方向 | 输入 | 损失 |
| --- | --- | --- |
| B_rgb | RGB 时序 | 普通 BCE，作为对照 |
| C_rgb_balanced | RGB 时序 | 正例权重 = 训练集可用负例数 / 可用正例数 |
| D_fusion_balanced | RGB 时序 + 冻结的原模型特征 | 同上 |

每次最多 20 轮，验证损失连续 4 轮不改善即停止；best 按未加权验证 BCE 选择。
权重只用训练集已知且 RGB 可用的窗口计算，所有种子共用相同样本和归一化。
RGB 为现有冻结编码器输出：4 秒内 8 个样本，训练轻量时序网络，不是高帧率眨眼检测器。

## 查看结果

```bash
tail -f runs/body-fatigue-visual-001.log
```

- `runs/body-fatigue-visual-001/comparison.csv`：原模型 + 6 次训练的 best/last，共 13 行。
- `runs/body-fatigue-visual-001/comparison.json`：同一验证集的总体、按人、按录像、F 缺失分组，及跨种子汇总。
- `runs/body-fatigue-visual-001/training-labels.json`：真正进入训练的正负例数与权重。
- 每个实验目录的 `train-001/history.json`、`best.pt`、`last.pt`：训练曲线及权重。
- 每个 `*-windows.csv`：逐窗口预测，可定位实际检出与误报录像。

主表固定概率阈值为 0.5（logit 0）。首先看 `tp`、`fn_total`、`fp`、
`false_positive_rate`、`all_positive_detection_rate`，以及 `f_missing_tp`。
`fn_total` 把无法预测的正例也计入漏检；旧 `fn` 只统计可预测正例中的漏检。
`fixed_threshold_pass` 仅表示固定阈值下 TP > 0 且 FPR ≤ 3%，是最小筛选条件，
不表示达到部署标准。`diagnostic_fpr03_*` 是验证集阈值扫描，不能当独立测试效果。
完整 body 验证集始终保留，正例分母按实际标签计算，不强写成 61。

异常中断后，在确认旧进程已结束的情况下续跑：

```bash
nohup bash scripts/server/train_body_fatigue_visual.sh --continue-run \
  >> runs/body-fatigue-visual-001.log 2>&1 &
echo $!
```

续跑校验配置与来源，完成的实验复用；失败阶段从已校验的 epoch 检查点接续。
输入不兼容直接报错，不能将旧分心模型当疲劳模型使用。
本地合成测试只验证标签、缺失处理和训练流程；真实提升以此次服务器结果为准。
