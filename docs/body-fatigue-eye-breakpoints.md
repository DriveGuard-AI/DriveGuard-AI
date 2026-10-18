# 纯闭眼漏检断点分析

当前已验证的 dense_track 结果为疲劳 20/61、误报 0/215，纯闭眼 1/13。纯闭眼中两条窗口具备连续输入条件，其中一条仍未检出。此工具自动分析这些窗口，同时保留成功检出作为对照。窗口数量由报告决定，不写死被试或记录 ID。

本轮训练次数为 **0**。固定原 B seed42 best、原哈欠分支、logit≥0 和 1000ms 闭眼事件规则；不改跟踪策略、模型、数据标签或旧缓存。所有验证窗口先完整推理并复现既有报告，随后按标签选择诊断页面。报告不是新模型成绩，也不自动改变全任务路由。

## 启动

代码同步到服务器后：

```bash
cd /mnt/workspace/DriveGuard-AI
nohup bash scripts/server/diagnose_eye_breakpoints.sh \
  > body-fatigue-eye-breakpoints-001.log 2>&1 < /dev/null &
echo $!
```

一个后台任务，顺序执行两步：

1. NPU 使用已有缓存复现完整验证集预测，输出逐帧状态、窗口分析及眼区回放。
2. CPU 仅解码纯闭眼正例涉及的录像，从录像开头重放定位，记录停止原因。每个新鲜采样位置的坐标、有效性、来源、跟踪年龄和眼区像素都必须复现旧缓存；否则报错，不生成成功报告。

不重新生成三套眼区缓存，不安装依赖或下载模型。默认使用 `.venv-visual-npu`、`.venv-perception-arm64`。

默认 `RECOVERY_REPORT=runs/body-fatigue-eye-recovery-001/comparison/comparison.json`，`OUTPUT_DIR=runs/body-fatigue-eye-breakpoints-001`，`DATASET_ROOT=Dataset`。可通过这些环境变量和 `PYTHON_BIN`、`PERCEPTION_PYTHON`、`DEVICE_INDEX` 覆盖。CPU MediaPipe/OpenCV 版本必须与原 dense_track 缓存一致。

中断后保持路径、输入和环境不变：

```bash
nohup bash scripts/server/diagnose_eye_breakpoints.sh --continue-run \
  > body-fatigue-eye-breakpoints-001-resume.log 2>&1 < /dev/null &
echo $!
```

分析阶段复用已完成且所有输出文件校验通过的结果；未完成时重新做固定推理。原视频重放阶段复用已完成且校验通过的录像；未完成的录像必须从头重放以保留跟踪状态。

## 优先发回的结果

位于 `runs/body-fatigue-eye-breakpoints-001/`：

- `analysis/summary.json`：漏检窗口数量、按原因/被试统计、全部纯闭眼窗口明细。
- `source-replay/summary.json`：定位重放是否复现、每条录像与各窗口的终止原因计数。
- `source-replay/stop-reasons.csv`：便于比较 250ms 到期、无初始种子、光流失败、运动不一致、越界等原因。

直接查看 `analysis/index.html`，点击窗口进入可拖动、播放的头部/眼部对照页面；每个窗口也有 `replay.gif`。左右分别是模型缓存头部图和实际眼部输入，黄色框为候选眼区，缺失眼区显示黑色占位。回放包含窗口前 1 秒上下文，显示当前 logit、来源、距上次检测时间、持续闭眼计时及输入连续时长。

## 明细口径

- `analysis/windows.csv` 保留全部正例、负例、未知窗口；`pure-eye-windows.csv` 是纯闭眼正例子集，包含成功检出对照。
- `analysis/frames.csv` 保留整段录像的 8Hz 采样序列。计时从录像开始计算，窗口边界不清零。`source_reset` 为原缓存绑定的源视频重置标记，不因 8Hz 相邻采样间隔重新推断。
- `input_duration_eligible` 把可用眼区假设为闭眼，是连续输入上限；`eye_event` 使用真实预测。事件仍在窗口终点读取，较早曾触发不代表终点检出。
- `longest_*_run_seen_ms` 可包含窗口开始前的连续历史。`longest_missing_sample_span_ms` 是同一串缺失样本首尾的采样网格时间差，单个缺失样本为 0ms；同时查看缺失样本数。不能据此断言两次采样之间所有原帧也缺失。
- `reason` 是窗口终点的主要阻断条件，详细信息还包括低分数量、分数范围及最后一次中断位置；多个问题可能同时存在。
- `source-replay/*-source.csv` 是原解码帧粒度，与 8Hz 分母不同。`trace_json` 包含实际执行到的光流检查数值。无数据的检查不会猜测。
- `tracking_status=stopped` 表示这帧首次终止现有跟踪；之后连续 `no_seed` 不是反复发生的终止事件。`last_stop_*` 可追溯长缺失前的终止点。检测成功但纹理不足记为 `detected/seed_insufficient_texture`，此帧仍可提供检测眼区，后续无法靠它初始化跟踪。
- `stop-reasons.csv` 的 `record_all_frames` 为整段录像，`requested_intervals_unique_frames` 为诊断片段的去重并集，`window_with_preroll` 为各窗口含前置上下文。重叠窗口计数不能直接相加。

只在 `source-replay/summary.json` 完成且 `cache_reproduced=true` 时，才把新记录的定位原因用于解释旧结果。此前缓存阶段能解释输入/分数/计时，但不能提供历史光流终止原因。
