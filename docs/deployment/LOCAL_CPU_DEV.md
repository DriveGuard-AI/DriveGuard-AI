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
（Python 3.12.14，含 torchvision/av/scikit-image 等）。上表是本次实测实际使用的
Python 3.11 组合，两者未做交叉验证；需要完整离线抽取能力时以锁文件为准。

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

## 6. 实测性能（8 核 CPU，yolo11s @640）

服务端 `processing_ms` 口径（`server.py` 的 `frame()` 包围 `bridge.call` + `stream.consume`）：

| 指标 | 值 |
|---|---|
| 最小 | 109 ms |
| 中位数 | **141 ms** |
| 最大 | 234 ms（25 槽边界帧） |

每帧构成：

| 阶段 | 耗时 |
|---|---|
| YOLO11s 前向 + NMS | 92 ms（channels_last）／114 ms（NCHW） |
| MediaPipe face + hands | 17 ms |
| eye locator（第二个 face 图，头部裁剪） | 4 ms |
| JPEG 解码 + 裁剪 + base64 | 6 ms |
| 跨进程 JSON 传输（RGB 采样帧约 200 KB） | 约 10 ms |
| 服务端模型窗口（每 25 槽一次） | +78 ms，仅边界帧 |

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
