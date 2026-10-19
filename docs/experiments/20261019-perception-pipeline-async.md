# 感知管线耗时剖析与检测器异步化

2026-10-19 · `model` 分支 · 复现脚本：`scripts/local/profile_perception.py`、
`scripts/local/compare_windows.py`

范围：定位单帧 250 ms 的真实构成，把最慢的协作者移出关键路径，并量化这样做的
精度代价。不涉及训练、阈值调整、NPU 或应用层。

## 1. 为什么先剖析再改

上一轮只掌握端到端数字（`processing_ms` p50 250 ms），而曾记录的分阶段表
（MediaPipe 17 ms、eye 4 ms）与它不自洽。在不知道 250 ms 由谁构成的情况下做优化
等于碰运气，所以先测量。

## 2. 方法

`CameraPerception` 是真实产品类。剖析脚本不修改产品代码，而是把它的四个协作者
包一层计时器后直接调用 `frame()`：

| 被计时对象 | 说明 |
|---|---|
| `perception_worker.decode_jpeg` | JPEG 解码（1280×720） |
| `detector.predict` | 到检测器进程的完整 HTTP 往返 |
| `landmarks.predict` | MediaPipe face + hands |
| `locator.step` | 第二个 face 图（头部裁剪）上的眼位 |
| `extractor.step` / `tracker.update` | 特征抽取与跟踪 |

素材为 `runs/frames/electr1` 的 60 张真实帧，检测器与感知进程分别运行，与生产
部署一致。

## 3. 结果

```
full frame()  n=59  median=  242.7ms  min=  216.3  max=  327.4
  decode     n= 60  median=  13.94ms  p90= 14.96  max= 23.55
  detector   n= 60  median= 133.63ms  p90=155.88  max=163.18
  landmarks  n= 60  median=  57.50ms  p90= 78.83  max= 94.86
  eye        n= 60  median=  22.63ms  p90= 45.27  max= 78.23
  features   n= 59  median=   0.37ms  p90=  0.58  max=  1.97
  tracker    n= 60  median=   0.02ms  p90=  0.06  max=  0.11
  accounted       median= 228.10ms   unaccounted= 14.58ms
```

判定：

1. **检测器占 55%**（133.6 / 242.7），是唯一值得动的单项；
2. MediaPipe 实测 57.5 ms，是旧记录的 3.4 倍；eye 实测 22.6 ms，是 5.7 倍。
   旧数字是单阶段空跑值，与并发时的真实成本无关；
3. `features` 与 `tracker` 合计 0.4 ms，不构成瓶颈；
4. 未归因的 14.6 ms 落在 base64 编解码、JSON 与 GIL 争用上。

## 4. 改动

`inference/live/perception_worker.py` 新增 `objects_for()` 与 `collect_detection()`：

- 检测器跑在 `ThreadPoolExecutor(max_workers=1)` 上；
- **忙时直接复用上次结果，不排队**——队列深了在检测器慢于帧率时永远排不空，
  而且每条排队条目只会让证据更旧；
- 响应的 `timestamp_ms` 保持它自己的捕获时刻，因此特征行的 `ages` 反映真实滞后，
  而不是假装这些框属于当前帧；
- 跟踪器只在收到新检测时推进一次，避免同一批框被反复关联到不同时刻；
- 首帧阻塞一次，使 slot 0 与原严格串行实现逐位一致；
- `close()` 先 `shutdown(wait=True)` 再关传输层，避免请求写进已关闭的连接；
- 30 s 死锁保护：请求卡住时立即失败，而不是让证据静默冻结到传输超时（120 s）。

## 5. 验证

### 5.1 等价性

同一段 `electr1.mp4`（1114 帧，25 Hz 网格）分别跑串行与异步，按 `window_end_ms`
去重后逐窗比对。注意服务会把最近一次窗口结果在每帧重发，所以 jsonl 的 1014 行
其实只有 **41 个不同窗口**。

| 任务 | 可用率 B→C | 检出率 B→C | 逐窗一致 | score Δ中位 | score Δ最大 |
|---|---|---|---|---|---|
| fatigue | 100% → 100% | 2.4% → 2.4% | **41/41** | 0.0000 | 0.0000 |
| phone_use | 100% → **97.6%** | 97.6% → 95.1% | 40/41 | 0.0001 | 0.0177 |
| drinking | 100% → **97.6%** | 100% → 97.6% | 40/41 | 0.0000 | 0.0007 |
| distraction | 100% → 100% | 65.9% → 68.3% | 40/41 | 0.0312 | 0.1132 |
| gaze | 63.4% → 63.4% | — | **41/41（100%）** | — | — |

差异窗口只有 2 个：

```
 5960 ms  phone_use 1.0->None, drinking 1.0->None
16960 ms  distraction 0.33->0.392
```

第一个是异步的直接代价：该窗口有一帧证据缺失，两个任务的 `available` 变 false。
第二个是阈值边界抖动。

### 5.2 性能

| 指标 | 串行 | 异步 |
|---|---|---|
| 墙钟（1114 帧） | 366.8 s | 236.5 s / 237.6 s |
| p50 | 282 ms | **172 ms** |
| 实时倍率 | 0.12× | **0.19×** |

### 5.3 中断

两次异步运行中，一次在第 979 帧因 `[WinError 10060]`（连接检测器超时）中断，
另一次 0 中断。该错误在异步化之前就出现过（见 `44c373b` 的修复背景），不是本次
引入；但异步把它的后果从“立即失败”放大成“证据静止直到传输超时”，因此补了 30 s
死锁保护。

## 6. 结论与边界

1. 异步化把单帧从 242.7 ms 降到 172 ms，**仍未达到 8 Hz 所需的 125 ms**；
2. 下一段瓶颈是 MediaPipe（57.5 ms）+ eye（22.6 ms）+ 解码（13.9 ms），全在 CPU；
3. 代价已量化：`phone_use`/`drinking` 有 2.4% 窗口证据不可用，1 个窗口的
   `distraction` 判定翻转；
4. 未验证：真实 webcam 下的长时间稳定性、`drinking` 误报场景（KL-01）以外的素材、
   以及异步对 `ages` 长期贴近 81 ms 上限的影响。

## 7. 顺带确认的两条契约

- 捕获间隔 ≤120 ms（8.33 Hz）时，25 Hz 槽位与头采样均保持 100% `fresh`
  （`CausalClock` 会用同一帧展开多个槽）；160 ms 起开始掉。
- `smoke_live_video.py` 新增 `--stride`，与 `--step-ms` 配合可模拟低捕获率而不把
  片段变成慢放。12.5 Hz 采样（`--stride 2 --step-ms 80`）与 25 Hz 的 41 个窗口
  比对：fatigue/phone_use/drinking 全部 41/41 一致，distraction 39/41。
