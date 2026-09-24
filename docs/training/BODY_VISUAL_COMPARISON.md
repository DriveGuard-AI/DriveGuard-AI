# Body distraction：冻结 RGB 特征对照

这轮验证「原有结构化特征缺少可用的视觉信息」这一假设。不会保证提高准确率，也不修改原有联合模型。训练目标限定为 body 视角的 distraction 二分类。

| 组别 | 输入 | 新训练内容 |
| --- | --- | --- |
| A_gap80 | 原结构化特征 | 无，复用 gap80/best.pt |
| B_rgb | 冻结 ResNet18 的 RGB 特征 | 小型因果时序分类头 |
| C_rgb_context | RGB + 冻结 gap80 的分组嵌入、质量、有效性和原 dist logit | 小型融合时序分类头 |

一共 **3 组对照、2 次新训练**。B、C 顺序训练，共用一次 RGB 提取和原特征编码；每组训练默认 20 轮。比较表中 A 一行，B/C 各 best、last 两行，共 5 行。

## 在 Linux NPU 服务器运行

先把本次新增的 `training/visual` 和服务器脚本同步到服务器项目。以下命令在 `/mnt/workspace/DriveGuard-AI` 执行。

第一次安装独立视觉环境，并明确下载官方预训练权重（约 45 MB）：

```bash
cd /mnt/workspace/DriveGuard-AI
bash scripts/server/setup_body_visual.sh
```

脚本用原 NPU Python 创建 `.venv-visual-npu`，继承已有 torch、torch_npu 和 NumPy，只安装 PyAV 12.3.0 / Pillow 10.4.0，不安装 torchvision。权重来自 `https://download.pytorch.org/models/resnet18-f37072fd.pth`，完整 SHA256 必须是 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。已有文件摘要不符会报错，不覆盖。离线机器可手动放置同一文件到 `models/local/resnet18-f37072fd.pth`，仍会校验。

若已看到 `Visual dependencies ready` 后出现下载超时，依赖安装已完成，只需重试权重下载，无需重新安装环境。下载器默认每次网络操作超时 300 秒、最多尝试 3 次；失败的临时文件会清理，下次尝试从头下载。只有验证通过的完整权重才会发布。

```bash
.venv-visual-npu/bin/python -m training.visual.weights --download \
  --timeout 300 --attempts 3
```

该命令成功后再启动训练。自编 `test.sh` 应加 `set -e`，或用 `if`/`&&` 连接准备与启动步骤，避免安装/下载失败后仍启动后台进程。后台 PID 不代表训练成功。如果服务器持续无法访问官方下载地址，可在能联网的机器下载同一权重并传到上述位置，再运行不带 `--download` 的校验命令。

启动后台流水线：

```bash
mkdir -p runs
nohup bash scripts/server/train_body_visual_suite.sh \
  > runs/body-visual-comparison-001.log 2>&1 < /dev/null &
echo $!
```

这个命令是 `nohup`；安装命令不是。先提取 RGB，再准备冻结特征，最后依次训练 B/C。默认 **CPU 提取 RGB、NPU 训练分类头**，视频只按原始时间戳解码并以 2 Hz 编码一次。可以在新任务开始前通过 `RGB_BACKEND=npu` 指定 NPU 编码，例如 `RGB_BACKEND=npu nohup bash ...`；没有可用 NPU 会报错，不会静默降级。暂未在真实 Ascend 环境验证此编码路径。

默认输入与上一轮一致：

```text
MANIFEST=runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json
DATA_ROOT=runs/dmd-full-yolo11-features-001
DATASET_ROOT=Dataset
CHECKPOINT=runs/body-dist-comparison-001/gap80/train-001/best.pt
OUTPUT_DIR=runs/body-visual-comparison-001
```

如果 gap80 的 best.pt 在另一个目录，通过 `CHECKPOINT=/实际路径/best.pt` 指定；同时必须保留 `.pt.sha256.json`。可设置 `EPOCHS`、`BATCH_SIZE`、`SEED`、`RGB_BATCH_SIZE`、`DEVICE_INDEX`，所有配置会写入运行身份。`--check-config` 只检查命令格式，不代表依赖、权重或数据通过验证。

查看进度：

```bash
tail -f runs/body-visual-comparison-001.log
```

只有确认旧进程已退出后，才对同一目录续跑：

```bash
nohup bash scripts/server/train_body_visual_suite.sh --continue-run \
  >> runs/body-visual-comparison-001.log 2>&1 < /dev/null &
echo $!
```

续跑会校验配置、输入及缓存摘要。已完成的视频缓存复用；中断的视频用新 attempt 重跑；准备阶段用新目录重建未完成结果；分类头从完整 epoch 检查点恢复。不要同时启动两个进程写同一个输出目录。改变配置需设置新的 `OUTPUT_DIR`。

## 查看结果

完成标志：`runs/body-visual-comparison-001/result.json` 中 `status=complete`。

主要结果：

```text
runs/body-visual-comparison-001/comparison/comparison.json
runs/body-visual-comparison-001/comparison/comparison.csv
```

同目录还有各组 `*-windows.csv`，以及 B/C 的 `*-summary.json`。B/C 训练过程在 `comparison/B_rgb/train-*/history.json` 和 `comparison/C_rgb_context/train-*/history.json`。

优先判断：

1. `overall.all_positive_detection_rate`：TP / **全部已知正样本**，不可用的正样本也在分母。
2. `overall.false_positive_rate`：FP / 全部已知负样本，同时看 `coverage` 和 `available_false_positive_rate`，防止缺失造成假低误报。
3. `baseline_h_ineligible`：固定按 A 的头姿不可用窗口分组，观察 B/C 是否找回过去缺失的正样本。
4. `validation_fpr03`：验证集上误报率不超过 3% 时可达到的 TP。相同分数的窗口不能拆开选取。此阈值由验证集选出，仅用于比较；不是独立测试成绩或最终部署阈值。
5. `by_subject` / `by_record`：确认收益是否仅来自一个人或一段录像。

默认分类阈值为 p≥0.5（logit≥0）。`fn/tn` 只针对可用预测，另列不可用正/负样本；`all_positive_f1` 将不可用正样本计入漏检。A 的 checkpoint 由原联合验证损失选出；B/C 的 best 由 body distraction 验证 BCE 选出，二者选择目标不同。不要把 B/C 当作 gaze、fatigue、phone、drink 的完整替代模型。

先用验证集决定是否值得进一步微调视觉编码器，再冻结方案并在未用于选型的测试集验证。若 B/C 都不能在可接受误报率下超过 A，这轮应否定当前冻结 RGB 方案，而不是继续增加训练轮数。

## 对齐与数据约束

- 每个 100 槽（25 Hz、4 秒）窗口读取 `[12,24,37,49,62,74,87,99]`，最后一帧与标签时刻对齐，不取未来帧。
- 每个视频使用原缓存中记录的 `driver_roi`，整块 ROI 按比例缩放并填黑边至 224×224，不额外猜测人脸裁剪框。
- 原视频字节摘要、解码 PTS、原特征缓存、帧数、分辨率和样本身份必须匹配。来源帧老于 80 ms 时，该帧特征为零；8 帧全部有效才有 RGB 预测。
- 提取所有 body train/val/test 记录，但归一化只用 train；训练只用 train，选型/比较只用 val。test 标签不参与训练或阈值选择。
- B/C 独立初始化，使用相同随机种子和训练日程。编码器全程冻结，不训练 MediaPipe；不会写回原缓存和原 checkpoint。

本地验证覆盖真实官方权重的 CPU 编码、合成数据端到端、时间对齐、摘要/来源校验、固定评估分母和中断恢复。真实数据上的提升幅度、服务器总耗时以及 NPU 兼容性，需以此次服务器实际结果为准。

第一轮融合结果后的后续对照见 [BODY_VISUAL_STABILITY.md](BODY_VISUAL_STABILITY.md)：复用 prepared 缓存，加入关闭 RGB 输入的配对对照、三个随机种子和早停，不必重新安装或提取 RGB。
