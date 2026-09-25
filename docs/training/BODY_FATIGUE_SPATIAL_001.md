# Body 疲劳：窗口核对与头部 RGB 对照

上一轮普通 RGB 在固定阈值下恢复了部分 subject14 正例，但 subject7 的 35 个正例
在所有实验中仍全部漏检。本轮只改变图像区域，检验整体 RGB 是否丢失了局部细节。

## 一条命令启动完整流程

服务器 `/mnt/workspace/DriveGuard-AI` 仓库根目录，先拉取成功：

```bash
git pull --ff-only origin model
```

然后后台运行：

```bash
nohup bash scripts/server/train_body_fatigue_spatial.sh \
  > runs/body-fatigue-spatial-001.log 2>&1 &
echo $!
```

脚本使用现有 `.venv-visual-npu/bin/python`，输入为上一轮的
`runs/body-fatigue-visual-001/comparison.json`、其引用的准备数据及检查点、原特征缓存、
`Dataset` 视频、`models/local/resnet18-f37072fd.pth`。不下载模型、不重跑 MediaPipe，
不覆盖这些输入或现有全任务模型。

完整流程顺序执行三步：

1. **窗口核对。** 重算 `B_rgb-seed43-best`，核对检查点、准备数据和已发布指标。
   导出所有已知验证窗口的标签/分数、分人的正负例分数分布、排序 AUC 和归一化截断比例。
   每个正例窗口导出三帧接触表；每人另取最多 8 个负例，包含高分误报候选。
   校验原始视频摘要、完整时间戳、采样对应关系及原始窗口末端的疲劳标签。
   检查失败立即停止，不继续提特征或训练。
2. **头部 RGB 提取。** 使用同一份本地冻结 ResNet18，在新的 `head-rgb-cache`
   目录提取裁剪特征。采样仍是每 4 秒 8 帧，必须与整体 RGB 的视频身份、PTS 和有效性逐项一致。
   这一步需要重新读取视频；原有整体 RGB 缓存直接复用。默认用 NPU 0，支持
   `RGB_BACKEND=cpu`（这是运行方式选择，不改变实验采样）。
3. **六次顺序训练。** 整体 / 头部 / 整体＋头部，分别跑 seed42、seed43。
   普通 BCE、相同模型容量、相同早停规则；最多 20 轮，耐心值 4。

## 裁剪及实验边界

所有记录使用同一个驾驶员 ROI 内归一化矩形 `[0.05, 0, 0.65, 0.65]`。
它是本轮固定的**候选头部区域**，不是逐帧人脸检测，也不按 subject 或验证分数调框。
它是否完整覆盖实际头部运动，需要看导出的预览；机器检查只能验证来源、时间和
标签值的一致性，不能证明标签语义正确或覆盖全程。因此结果保留 `manual_review=pending`，
完整流程可继续产生实验结果，但不自动认定裁剪合格或选择部署模型。

新模型都使用 1024 维输入：

| 实验 | 输入的两半 | 目的 |
| --- | --- | --- |
| A_whole | 整体 512 维＋零 512 维 | 同容量整体对照 |
| B_head | 零 512 维＋头部 512 维 | 局部细节对照 |
| C_whole_head | 整体 512 维＋头部 512 维 | 两类画面信息结合 |

这使三个实验的参数量相同，区别是画面信息。上一轮 512 维普通 RGB 的 best 另外复现、
列为历史参考，不能把网络输入维度改变造成的差异全部归因于裁剪。
归一化只拟合训练记录；未知疲劳标签不作为负例；RGB 可用性独立于 F 组。

## 输出和验收

```bash
tail -f runs/body-fatigue-spatial-001.log
```

- `audit-001/index.html`：正例及负例预览，整体图上标出原驾驶员 ROI 和新裁剪框。
- `audit-001/summary.json`：机械检查、分人分数、AUC、截断比例、人工核对状态。
- `audit-001/windows.csv`：全部已知验证窗口，不仅是选出的图片。
- `comparison/comparison.json`：3 个旧参考（原疲劳＋两种子普通 RGB）与 12 个新 best/last，默认共 15 行。
- `comparison/comparison.csv`：检出、总漏检、误报和 F 缺失分组。
- JSON 的 `aggregate_best`：按实验和按人的跨种子汇总；`paired_best_deltas`：同一种子下 B−A、C−A 的变化。
- 每次训练的 `train-001/history.json`、`best.pt`、`last.pt`：训练过程与检查点。

窗口标签位于末端，预览中前两帧不能当作独立正例标签；重叠窗口也不是独立疲劳事件。
验收要同时看 subject7 是否检出、subject14 是否退步，以及两人的 FP/FPR，不能只看总体 TP。
主指标固定 logit 0；FPR03 是验证集扫描诊断，不是独立测试集效果。
本轮不改变凝视、分心、电话、饮水等任务的输出。

需要先看诊断再跑训练，可以在同一脚本后加 `--audit-only`；看完后用
`--continue-run` 继续完整流程。异常中断也在确认旧进程已结束后续跑：

```bash
nohup bash scripts/server/train_body_fatigue_spatial.sh --continue-run \
  >> runs/body-fatigue-spatial-001.log 2>&1 &
echo $!
```

已完成的诊断逐文件校验后复用；头部缓存按完整记录恢复；训练按已校验的 epoch 检查点恢复。
参数或输入改变时拒绝混用，请用新的 `OUTPUT_DIR`。其他路径可通过 `FATIGUE_REPORT`、
`DATASET_ROOT`、`WEIGHTS`、`PYTHON_BIN` 环境变量设置。

本地测试只验证实现与协议，不能替代服务器真实视频上的 TP/FN/FP 结果。
