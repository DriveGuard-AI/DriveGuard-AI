# 本机 CPU 联调（非发布）

本页描述**没有 NPU 的开发机**如何在本机跑通 live 推理联调，以及由此测得的性能边界。
官方部署、模型组合与接口契约见 [LIVE_COMPOSITE.md](LIVE_COMPOSITE.md)。

> **范围声明**：本页所有性能数字与"实时摄像头不可用"的结论**只对该 CPU 环境成立**，
> 不能外推为模型能力或 NPU/GPU 部署结论。是否满足实时性必须在目标加速器上重新实测
> （源帧率、端到端延迟仍属未验证项）。
>
> 本页生成的 bundle 是**开发副本**，沿用 `webcam_integration_candidate_not_live_validated`
> 资格，不得当作已验证准确率的结果对外使用，也不得替代 `scripts/server/export_live_composite.sh`
> 导出的正式发布包。

## 1. 为什么不能直接加载发布包

`inference/live/bundle.py` 的 `load_bundle()` 会校验两件事，本机都过不去：

```powershell
.\.venv\Scripts\python.exe -m inference.live.bundle --check models/releases/live-composite-v1/bundle.json
# error: Bundle code identity differs; use matching source or export a new bundle
```

1. **`code_identity`**：`inference/` 与 `training/**` 下所有 `.py` 的 sha256 都被钉在
   `bundle.json` 里。只要源码有任何提交，校验就失败。
2. **加速器**：`runtime.json` 里 `objects.backend` 是 `npu`。CPU 上
   `select_device('npu')` 会直接报错（无 CPU 回退）。

正确做法不是改发布目录，而是用 `scripts/local/prepare_dev_bundle.py` 生成一份开发副本。

## 2. 三个解释器

与服务器上的拆分一致：主服务、感知子进程、检测器服务分别用独立解释器。

| 环境 | 用途 | 关键依赖 |
|---|---|---|
| `.venv` | 模型装配、测试、live 主服务 | torch 2.7.1+cpu、numpy、Pillow |
| `.venv-perception` | 感知子进程（MediaPipe） | mediapipe 0.10.21、numpy 1.26.4、opencv-contrib-python 4.11.0.86 |
| `.venv-detector` | YOLO11 检测器服务 | torch 2.7.1+cpu、torchvision 0.22.1+cpu、ultralytics 8.3.199 |

```powershell
py -3.11 -m venv .venv-perception
.\.venv-perception\Scripts\python.exe -m pip install numpy==1.26.4 Pillow opencv-contrib-python==4.11.0.86
.\.venv-perception\Scripts\python.exe -m pip install mediapipe==0.10.21

py -3.11 -m venv .venv-detector
.\.venv-detector\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv-detector\Scripts\python.exe -m pip install ultralytics==8.3.199
```

`ultralytics` 版本必须与 `inference/experts/protocol.py` 里的 `ULTRALYTICS_VERSION`
一致，否则 `settings_identity` 与检测器服务对不上。

仓库另有一份更完整的感知依赖快照 `requirements-perception-win-cpu.lock.txt`
（Python 3.12.14，含 torchvision/av/scikit-image 等），用于**完整离线视频抽取**。
上表是本页 live 联调实际使用的 Python 3.11 组合。两者用途不同、版本不同，
**不要混装进同一个环境**：live 只需要 numpy / Pillow / opencv / mediapipe，
装进 torch、scikit-image 会让感知子进程变重、启动变慢，也会模糊"感知环境不含
模型"的隔离；需要离线抽取能力时另建环境并按该锁文件安装。

## 3. 生成开发副本

令牌文件先就位（两个用途不同，都不能少于 24 字符）：

```powershell
.\.venv\Scripts\python.exe -c "import secrets;from pathlib import Path;d=Path('runs/live-local');d.mkdir(parents=True,exist_ok=True);(d/'api.token').write_text(secrets.token_urlsafe(32),encoding='ascii');(d/'detector.token').write_text(secrets.token_urlsafe(32),encoding='ascii')"
```

```powershell
.\.venv\Scripts\python.exe scripts\local\prepare_dev_bundle.py `
  --output runs\live-local\bundle --detector-token-file runs\live-local\detector.token
```

脚本行为与边界：

- 权重与资产用**硬链接**复制，不占额外空间，发布目录保持只读；
- 重写 `code_identity`、把 `objects.backend` 换成 `cpu`、把检测器令牌指向本机绝对路径；
- **绝不硬链接会被改写的文件**（`runtime.json` 单独写入），并在写后断言发布文件未被修改；
- 拒绝覆盖已存在的输出目录，最后用 `load_bundle` 自校验。

源码或权重变化后必须换一个新 `--output` 重新生成，否则校验失败。

## 4. 启动顺序

检测器先就绪，再加载组合模型：

```powershell
.\.venv-detector\Scripts\python.exe -u -m inference.experts.detector_service `
  --config runs\live-local\bundle\runtime.json --ready-file runs\live-local\detector-ready.json
```

等日志出现 `Detector ready: ...` 后：

```powershell
.\.venv\Scripts\python.exe -u -m inference.live.server --bundle runs\live-local\bundle\bundle.json `
  --perception-python "$PWD\.venv-perception\Scripts\python.exe" `
  --token-file runs\live-local\api.token --port 8765 --backend cpu --device-index 0
```

浏览器打开 `http://localhost:8765/#token=<runs/live-local/api.token 的内容>`，
在预览画面上拖拽框选驾驶员区域后开始推理。

停止：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like "*inference.live.server*" -or $_.CommandLine -like "*detector_service*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 5. 无摄像头冒烟

用合成 JPEG 走同一套 HTTP 契约，证明传输、感知子进程、检测器服务和时序窗口都在工作，
**不证明识别效果**（合成帧不含人脸）：

```powershell
.\.venv\Scripts\python.exe scripts\local\smoke_live.py --frames 170 --interval-ms 40
```

有视频文件时按 25 Hz 网格喂入（离线分析，比实时慢约 7 倍）。`--retries` 会在会话
因推理错误关闭后重开会话并从失败帧继续，长视频不会因为一次抖动前功尽弃：

```powershell
.\.venv\Scripts\python.exe scripts\local\smoke_live_video.py `
  --video "D:\path\to\clip.mp4" --output runs\live-local\video.jsonl --retries 2
```

2026-10-18 实测（1280×720 会话、1114 帧 / 44.6 秒时间线）：0 次中断、1014 个结果，
四任务始终 `available`，处理耗时 p50 250ms，墙钟 322 秒（0.14× 实时）。

## 6. 实测性能（8 核 CPU，yolo11s @640）

服务端 `processing_ms` 口径（`server.py` 的 `frame()` 包围 `bridge.call` + `stream.consume`）：

| 指标 | 值 |
|---|---|
| 最小 | 109 ms |
| 中位数 | **141 ms** |
| 最大 | 234 ms（25 槽边界帧） |

每帧构成（`scripts/local/profile_perception.py`，1280×720，检测器在独立进程内并发）：

| 阶段 | 中位 | p90 |
|---|---|---|
| JPEG 解码 | 13.9 ms | 15.0 ms |
| 检测器 HTTP 往返（含 2.76 MB 上传与结果校验） | 133.6 ms | 155.9 ms |
| MediaPipe face + hands | 57.5 ms | 78.8 ms |
| eye locator（第二个 face 图，头部裁剪） | 22.6 ms | 45.3 ms |
| 特征抽取 + 跟踪 | 0.4 ms | — |
| 未归因（base64、JSON 编解码、GIL 争用） | 14.6 ms | — |
| 合计（`CameraPerception.frame`） | **242.7 ms** | max 327.4 ms |

早期版本记的 MediaPipe 17 ms / eye 4 ms / JPEG 6 ms 是单阶段空跑值。与检测器进程
并发时不可复现，差值分别是 3.4×／5.7×／2.3×。以本表为准，因为它测量的是同一函数、
同一时刻的真实竞争。

单帧 242.7 ms 对应 4.1 Hz。检测器本身占 55%，所以把它移出关键路径是收益最大的单项改动：

| 方案 | 关键路径 | 帧率 |
|---|---|---|
| 严格串行（原实现） | 242.7 ms | 4.1 Hz |
| 检测器异步（滞后一帧复用） | **172 ms**（实测全视频 p50） | **5.8 Hz** |

已验证的免费优化：

| 方案 | 前向 | 相对 | 输出一致性 |
|---|---|---|---|
| eager NCHW | 108 ms | 1.00× | — |
| `channels_last` | 92 ms | 1.17× | max abs diff 0.0 |
| `torch.jit.trace` | 99 ms | 1.10× | max abs diff 0.0 |
| `jit.trace` + `channels_last` | 83 ms | 1.31× | max abs diff 0.0 |
| `torch.compile` | 不可用 | — | 本机缺少 MSVC `cl.exe` |

没有采用 `jit.trace`：它会把 ultralytics 检测头的 `self.shape != shape` 分支固化成常量
（`head.py` 抛 `TracerWarning`），一旦以后调整 `input_size` 就静默失效，收益不值这个风险。
线程数保持默认 8（实测 4/6/16 都更慢）。

### 检测器异步化的边界

检测器不再阻塞单帧：请求在后台单线程中执行，忙时直接复用上一次结果而不排队。
由此产生的证据年龄由 `ages` 如实上报（上限仍为 81 ms），带来两个必须记住的代价：

| 指标 | 串行 | 异步 |
|---|---|---|
| 墙钟（1114 帧） | 366.8 s | 236.5 / 237.6 s |
| p50 | 282 ms | 172 ms |
| 41 个窗口逐窗一致 | — | 39/41（另 2 个见下） |
| `phone_use` / `drinking` 可用率 | 100% | **97.6%** |

- 窗口 5960 ms：`phone_use`、`drinking` 因该帧证据缺失变为不可用；
- 窗口 16960 ms：`distraction` 0.33→0.392 跨过阈值，多一次检出；
- `fatigue` 与 `gaze` 的 41 个窗口完全一致（gaze 类别一致率 100%）。
- 偶发 `[WinError 10060]`：两次全视频运行中出现一次（第 979 帧），中断后可重试续跑。
  异步会把一次挂起放大成“证据静止”，因此加了 30 s 死锁保护使其快速失败而不是拖满
  120 s 传输超时。

## 7. 为什么本机不能用实时摄像头

浏览器一次只提交一帧，下一帧在上一帧返回后才采集，所以上传速率就是 1/单帧耗时。
单帧 141 ms 时，25 Hz 网格（40 ms 一格）只有约 28% 的槽位拿到新像素，其余槽位
`fresh=false`、`ages` 达到上限。实测两种喂帧方式的差别：

| 喂帧方式 | 槽位覆盖 | 每结果可用任务数 |
|---|---|---|
| 密集（视频文件按 40ms 步进） | 100% | 4 / 4 |
| 实时（浏览器一帧一发） | 19–23% | 0 / 4 |

这不是调参能解决的。疲劳分支要求 `len(head_valid)==32 and all(head_valid)`
（`inference/live/models.py`），即连续 32 个 head 采样槽都必须是新鲜帧：

| 槽位覆盖率 | 32 连续全有效的概率 |
|---|---|
| 28% | ≈ 0 |
| 50% | 约 2×10⁻¹⁰ |
| 90% | 约 3% |
| 99% | 约 73% |

即使把 YOLO 前向压到 0，非检测开销（MediaPipe 21 ms + JPEG/base64 6 ms + IPC 10 ms）
已经接近 40 ms 预算。所以本机的定位是**离线视频分析**（密集喂帧、慢于实时），
不是实时摄像头；实时性需要目标加速器实测确认。

## 8. 排障要点

- **`code_identity` 不匹配**：源码已超前于发布包，重新生成开发副本（换新 `--output`）。
- **`core.autocrlf=true` 破坏发布包**：Git 会把 `models/releases/` 下的文本文件改成 CRLF，
  导致 SHA-256 校验失败。`.gitattributes` 已用 `models/releases/** -text` 固定字节内容，
  旧检出需要重新 checkout 才能恢复。
- **几乎每个请求都返回 409**：服务端每 250 ms 的会话过期巡检会与请求抢同一把锁。
  已改为最近 1 秒内有新连接时跳过巡检（`/api/live/status` 本身也做过期检查），
  回归测试见 `tests/test_live_http.py`。
- **发布目录被改写**：`prepare_dev_bundle.py` 不会硬链接会被改写的文件；若手工复制
  `runtime.json` 再就地编辑，硬链接会把发布文件一起改坏。
- 本机排障不影响服务器部署路径；服务器侧见 [LIVE_COMPOSITE.md](LIVE_COMPOSITE.md)。
