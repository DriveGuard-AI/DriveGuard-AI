# 从最佳头部模型继续做视觉与时间采样对照

默认共同起点为 `body-fatigue-head-stability-001` 报告中的
`C_regularized-seed42-best`（当前验证结果 13/61 检出、2/215 误报）。
原 pt 文件不改写。它仅保存时序分类器，所以新流程还要加载当时提特征使用的
本地 ImageNet ResNet18 权重；哈希必须一致，不自动下载。

## 实验

| 组 | 图像编码器 | 每 4 秒帧数 |
| --- | --- | --- |
| A_frozen8 | 冻结，继续训练时序分类器 | 8 |
| B_tune8 | 解冻 layer4 卷积，联合训练 | 8 |
| C_frozen32 | 冻结，继续训练时序分类器 | 32 |
| D_tune32 | 解冻 layer4 卷积，联合训练 | 32 |

每组 seed42、43，共 **8 次顺序训练**，适合现有单张 NPU。
“同时尝试多个方向”指统一对照流程，不是向同一张卡同时启动八个训练进程。
所有组加载同一份最佳分类器权重；两种子只改变顺序、dropout 等随机过程，
不代表两个独立预训练模型。新阶段重置优化器；异常中断后的续跑恢复该阶段的优化器和 RNG。

原模型只有三层时序卷积。四组都增加同一层带零初始门控的因果时序残差层，
使可学习的感受野覆盖 32 帧。门控初始为零，因此 8 帧模型初始化时应复现旧预测。
这个公共扩展让四组时序参数形状一致；32 帧组不是仅换一个采样参数的旧模型。
图像 BN 统计量及其参数保持冻结，特征归一化沿用旧训练集统计；本轮不额外加入图像增强。

## 执行

代码同步到服务器后，在 `/mnt/workspace/DriveGuard-AI` 根目录运行：

```bash
mkdir -p runs
nohup bash scripts/server/train_body_fatigue_head_adaptation.sh \
  > runs/body-fatigue-head-adaptation-001.log 2>&1 &
echo $!
```

```bash
tail -f runs/body-fatigue-head-adaptation-001.log
```

首次需要重新读取训练/验证视频，生成共享的 224×224 RGB 像素缓存；
原来的 512 维特征不能用来微调编码器。32 帧含有原先 8 帧的所有采样时刻，
新增帧使用原 25Hz 时间轴上的因果采样。会核对视频哈希、PTS、尺寸、原始缓存和裁剪。
只读取 train/val 的像素；不训练或评估 test。像素缓存比原特征缓存大，日志会给出空间估算。

之后先用新图像路径复现旧检查点：标签、可用性、固定阈值判断必须一致，logit 允许
1e-4 绝对/相对浮点误差。失败就停止，不忽略错误继续比较。
训练保留第 0 轮初始候选；如果后续 BCE 没改善，best.pt 会仍是第 0 轮。
这应解读为继续训练没有带来 BCE 改善，不能把它算作新增收益。

默认微批 2、累积 8 次（满批约 16 窗口一次更新），编码分块 16 帧；
分类器学习率 1e-4、编码器 1e-5，最多 10 轮、早停耐心 3。
累积梯度按实际已知且可用的监督样本数归一化；未知标签不会当成负例。
微调的时间/显存成本高于冻结特征训练，实际吞吐需 NPU 实测。

显存不足时，使用新输出目录统一调整四组，例如：

```bash
BATCH_SIZE=1 ACCUMULATION_STEPS=16 FRAME_CHUNK=8 \
OUTPUT_DIR=runs/body-fatigue-head-adaptation-small-001 \
nohup bash scripts/server/train_body_fatigue_head_adaptation.sh \
  > runs/body-fatigue-head-adaptation-small-001.log 2>&1 &
```

若只是异常中断而不改参数，确认原进程结束后加 `--continue-run`，并使用原目录和日志。
完成的像素记录/训练复用，未完成记录重新生成到新尝试目录，训练从有效 epoch 恢复。
可通过 `STABILITY_REPORT`、`VARIANT`、`DATASET_ROOT`、`WEIGHTS`、`PYTHON_BIN`、`DEVICE_INDEX`
修改路径/设备。不要将输出设在输入目录内，不要同时启动两个写同一输出目录的任务。

## 结果

输出根目录默认 `runs/body-fatigue-head-adaptation-001`：

- `comparison.json`：原最佳参考，加八次训练各自 initial/best/last，默认 25 行。
- `comparison.csv`、`by-subject.csv`：整体和分人 TP/FP/FN/召回/F1/误报率。
- `reference-replay.json`：原模型从像素到预测的复现检查。
- 每行 JSON 的 `changes_from_original`：找回正例、丢失原检出、新误报、移除误报。
- `paired_best_deltas`：同种子各组相对继续训练对照 A 的差异。
- 各实验 `train-001/history.json`、`initial.pt`、`best.pt`、`last.pt`。

优先回传三份 comparison/by-subject 文件。固定阈值为 logit0；FPR03 扫描仅供验证诊断。
同时看是否优于原 13/2、是否优于共同扩展后的 A，以及 subject7/14 各自退步多少。
继续保留所有验证窗口和未知标签，不把重叠窗口当独立疲劳事件。
新检查点包含编码器和时序模块，格式不同于旧 visual_probe；使用新的 load_adaptation 加载，
不能直接塞进旧全任务组合。候选确认后才做组合接入与全任务回归。
