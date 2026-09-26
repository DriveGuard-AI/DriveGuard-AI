# 眼区连续性恢复对照

前一轮原视频眼区模型能在有效验证帧中检出 46/49 个闭眼帧，但纯闭眼事件仍为 0/13。输入审计确认：这 13 个窗口均没有满足原 1000ms 规则的连续眼区输入。当前任务先恢复输入并复用已训练权重，**不启动新训练**。

## 实验范围

固定 `B_eye_native-seed42-best-combined` 的 best 眼分支及其原动作模型的哈欠分支。先复现原有模型和缓存结果，再对比三个提取方案：

| 缓存 | 检测频率 | 短时跟踪 |
| --- | --- | --- |
| sampled8 | 原有每秒 8 个采样位置 | 无；必须复现原结果 |
| dense | 每个原视频解码帧 | 无 |
| dense_track | 每个原视频解码帧 | 检测暂时无脸时，最多距上次确认检测 250ms |

三组均保存原有 8Hz 采样时刻的原视频眼区，输出时钟及窗口不变。dense 增加的是定位更新频率，不修改闭眼判定时长。

跟踪只从已确认的眼区开始，使用当前帧图像做前后向光流检查；要求足够纹理、运动一致、位移受限、眼区位于头部范围内。源帧间隔超过 80ms 或距上次确认检测超过 250ms 时停止。检测出现多脸歧义、无效关键点、区域筛选拒绝时不能靠跟踪绕过。新检测成功优先于跟踪。跟踪结果仍只是候选眼区，需要检查预览，不能把跟踪通过当成闭眼或真实可见的证明。

仅重新提取原验证集所有录像，保留负例和未知标签窗口；不按正例挑选视频，也不改变原 61/215 分母。三份新增像素缓存约占原完整眼缓存验证部分的三倍；检测在 CPU 上进行，实际耗时取决于视频总长。

## 启动与续跑

代码同步至服务器后执行：

```bash
cd /mnt/workspace/DriveGuard-AI
nohup bash scripts/server/compare_eye_continuity.sh \
  > body-fatigue-eye-recovery-001.log 2>&1 < /dev/null &
echo $!
```

这是一个后台任务，内部顺序生成三组缓存、再执行固定模型推理。**训练次数为 0**。使用已有 `.venv-perception-arm64`（PyAV、MediaPipe、OpenCV、NumPy、Pillow）和 `.venv-visual-npu` 环境，不自动安装或下载模型。

默认：

- `EYE_REPORT=runs/body-fatigue-eyes-001/comparison/comparison.json`
- `VARIANT=B_eye_native-seed42-best-combined`
- `DATASET_ROOT=Dataset`
- `OUTPUT_DIR=runs/body-fatigue-eye-recovery-001`

支持环境变量 `PERCEPTION_PYTHON`、`PYTHON_BIN`、`DEVICE_INDEX` 及以上路径覆盖。头部缓存从原报告绑定的来源读取。中断后保持设置不变：

本轮 `VARIANT` 必须保持 B seed42 best，启动脚本会在解码前拒绝更换模型。

```bash
nohup bash scripts/server/compare_eye_continuity.sh --continue-run \
  > body-fatigue-eye-recovery-001-resume.log 2>&1 < /dev/null &
echo $!
```

完整缓存记录经校验后复用；原缓存保持 v1 兼容，恢复策略、OpenCV 版本（跟踪组）、来源及采样信息另行绑定到新缓存。输出目录不得覆盖输入。

## 查看结果

在 `runs/body-fatigue-eye-recovery-001/comparison/` 查看：

1. `input-support.json`：每组、每被试、每类疲劳正例有多少窗口具备连续观察条件。这是“把有效眼区假设为闭眼”的输入上限，不是预测准确率。
2. `rejection-reasons.csv`：区分检测不到脸、关键点无效、区域筛选、多脸歧义等。`sample_detector` 是保存位置的检测原因；`source_detector_calls` 是所有检测调用，不能混用分母；`origin` 区分检测和跟踪来源。
3. `comparison.json`、`comparison.csv`、`by-subject.csv`、`by-subtype.csv`：原眼缓存及三个新缓存，各有 combined/eye_only 读出，共 8 行。含相对原模型的恢复、丢失、新增误报和移除误报；哈欠分数固定。
4. 每组目录的 `eye-window-coverage.csv`、`coverage.csv`、`input-audit/index.html`：按窗口检查时长、按样本核查裁剪。预览中 origin=0/1/2 分别表示缺失、检测、跟踪，age 为距最近确认检测的毫秒数。

优先比较纯闭眼支持数是否从 0/13 增加、实际检出是否增加以及负例误报。恢复后帧级评价集合会扩大，不能只比较不同集合的 AUC。所有结果均是验证集上的探索性对照，程序不自动替换模型或全任务路由。

独立眼训练入口现增加 `eye-support.json` 检查：纯闭眼正例连续观察支持数为零时，先写诊断报告，再在启动新训练或恢复未完成训练前报错停止。已完成的训练仍可复核和复用。固定模型恢复评估即使支持数为零仍会完成，便于核查负例误报和失败原因。
