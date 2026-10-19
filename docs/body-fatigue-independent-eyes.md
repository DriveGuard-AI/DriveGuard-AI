# 独立闭眼分支对照

本轮处理眼部分支跨被试失效：现有事件模型检出 19/61，全部来自哈欠；单独眼部分支检出 0/61。降低闭眼阈值未增加正例检出，反而在 subject14 增加了 14 个误报。新实验从同一个动作模型初始化独立眼部分支，保持原模型的哈欠分数不变。

## 三组实验

| 组 | 输入 | 训练采样 |
| --- | --- | --- |
| A_head_independent | 现有头部 RGB | 帧均匀采样 |
| B_eye_native | 原视频直接裁出的眼部 RGB | 帧均匀采样 |
| C_eye_subject_balanced | 同一眼部 RGB | 按训练被试均衡采样 |

默认每组种子 42、43，共 **6 次训练，顺序执行**。每次最多 6 轮；默认 patience=7，让各组完成相同训练预算。独立眼分支只微调 ResNet18 的 layer4 卷积和眼分类头，冻结 BatchNorm。三组共享初始化、标签、有效帧集合、训练轮数与类别权重；C 只改变被试采样。所有类别权重和采样频率仅来自训练集。

best 按验证集各被试的正负类平衡眼部 BCE 选取，不包含哈欠损失。若训练未改善该指标，epoch 0 可以保持 best，报告仍包含最后一轮。保留原有 1000ms 连续闭眼规则和固定 logit=0 阈值。

眼部区域从原视频、原时间戳提取。检测在既有头部区域内进行，选投影跨度较大的单眼；眼角原始间距至少 6 像素，带边距的裁剪必须完整位于头部区域内。此规则只生成候选眼区，不代表人眼确认可见。定位失败、过小或越界都标记为缺失，不当作开眼样本。原始小眼区放大到 224×224 不会创造细节，因此报告记录原始眼角间距。

## 服务器运行

同步代码后，在项目根目录执行。沿用已有环境和本地模型，无新权重下载：

```bash
nohup bash scripts/server/train_body_fatigue_eyes.sh > body-fatigue-eyes-001.log 2>&1 < /dev/null &
echo $!
```

脚本本身前台执行；上面的 `nohup` 命令使其后台运行。先用 `.venv-perception-arm64/bin/python` 构建原视频眼部缓存，再用 `.venv-visual-npu/bin/python` 顺序训练 6 次。原视频解码和 MediaPipe 提取在 CPU 环境进行。眼部 uint8 缓存体积与现有头部像素缓存接近，需要额外磁盘空间。

默认输入是 `runs/body-fatigue-actions-001/comparison.json` 中的 `B_action_supervision-seed42-best-event`，自动读取该报告绑定的头部缓存。默认数据目录为 `Dataset`，默认新输出目录为 `runs/body-fatigue-eyes-001`。

可通过环境变量指定 `ACTION_REPORT`、`VARIANT`、`DATASET_ROOT`、`HEAD_CACHE`、`OUTPUT_DIR`、`PERCEPTION_PYTHON`、`PYTHON_BIN`、`DEVICE_INDEX`、`BATCH_SIZE`、`ACCUMULATION_STEPS`、`FRAME_CHUNK`、`EPOCHS`、`PATIENCE`。更换数据路径必须与原缓存的来源信息一致。

```bash
# 非默认环境路径的例子
PERCEPTION_PYTHON=/path/to/perception/bin/python \
PYTHON_BIN=/path/to/visual/bin/python \
bash scripts/server/train_body_fatigue_eyes.sh
```

中断后使用相同环境变量和设置续跑，复用完整缓存记录和已完成训练：

```bash
nohup bash scripts/server/train_body_fatigue_eyes.sh --continue-run > body-fatigue-eyes-001-resume.log 2>&1 < /dev/null &
echo $!
```

要单独运行两个阶段，可调用 `python -m training.visual.eye_frames --help` 和 `python -m training.visual.fatigue_eyes --help` 查看参数。前者使用感知环境，后者使用训练环境。

## 查看结果

`runs/body-fatigue-eyes-001/comparison/` 包含：

- `comparison.json`、`comparison.csv`：原模型加 6 次训练各自 best/last 的组合、眼分支独立读出，共 25 行对照；以及正例恢复、原有检出丢失和新增误报。
- `by-subject.csv`、`by-subtype.csv`：subject7、subject14，以及纯闭眼、纯哈欠、重叠类别，保留原分母。
- `frame-metrics.json`：有效且标签明确的独立眼帧分类指标，用来区分帧分类失败与连续闭眼规则未触发。
- `coverage.csv`、`eye-window-coverage.csv`：正负例眼区覆盖率、原始眼部尺寸、各窗口是否具备持续观察条件；覆盖率不是识别准确率。
- `input-audit/index.html`：每条记录正负例及缺失案例，左为现有头部图及候选眼区位置，右为直接从原视频裁出的眼部图。优先检查 subject14。
- 各方案的 `*-windows.csv`、训练子目录中的 `history.json`、`initial.pt`、`best.pt`、`last.pt` 和完整来源信息。

组合读出固定原哈欠分数。眼部缺失只重置闭眼计时；原有哈欠检出不能因此丢失，程序对此强制校验。评估仍包含原有全部窗口（当前 61 个已知正例、215 个已知负例及未知标签窗口），不会因眼部缺失删除正例。

主要验收：纯闭眼 0/13 是否改善、215 个负例新增多少误报、两个被试是否都改善。报告是验证集选模结果，不是独立测试集结论。本轮不替换全任务路由，也不自动发布模型；真实效果要以服务器运行结果为准。
