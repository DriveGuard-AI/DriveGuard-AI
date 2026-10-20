# 🛡️ DriveGuard AI

### 驾安智护：基于多模型感知与时序分析的危险驾驶检测系统

DriveGuard AI 是一套面向真实车内环境的驾驶员监测系统（DMS）。系统结合计算机视觉、面部状态分析与时序行为建模，识别疲劳驾驶、手机使用、饮水、低头和视觉分心等风险。

它关注的不只是某一帧出现了什么，而是：

> **驾驶员过去几秒钟的行为是否正在形成驾驶风险。**

> 当前状态（2026-10-20）：
> - **应用侧**：MVP 前后端可运行。离线分析（上传/抽帧/回放/与实时互斥）、实时摄像头预览（MJPEG + 帧序号/摘要对齐）、四任务六参数判决表均已实现；后端 **114 项测试通过**。
> - **模型侧**：关系门控/TCN 训练与实时组合推理可用。已在 **Ascend 910B** 上跑通「摄像头 → 检测器 → 感知 → 组合模型 → 风险快照」全链路；真实 DMD 精度验收、NVIDIA GPU/Jetson 实机与正式发布仍未完成。
> - **已知问题**：body 视角 `drinking` 存在系统性误报（KL-01），不能直接当告警使用，详见[已知限制](docs/testing/KNOWN_LIMITATIONS.md)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `backend/`、`frontend/` | 应用侧：FastAPI 服务与 Dashboard |
| `inference/`、`training/`、`configs/`、`models/`、`scripts/`、`tests/` | 模型侧：感知/时序推理、训练入口、配置与权重 |

## 快速开始（应用侧）

> **解释器锁定 Python 3.11**：版本写在 `backend/.python-version`，`run.ps1` 会读取它并校验实际解释器。
> **依赖锁定 `backend/requirements-dev.lock.txt`**（运行时用 `requirements.lock.txt`）：锁文件记录的是后端测试通过时的精确版本；`requirements.txt` 只保留可读的直接依赖范围。
> 若 `python --version` 显示 3.7（例如旧版 Anaconda），**必须另建虚拟环境**，否则依赖无法安装。

### 后端（FastAPI + SQLite）

最省事的方式（自动建虚拟环境 + 装依赖 + 启动）：

```powershell
cd backend
.\run.ps1                    # Windows；macOS/Linux 用 ./run.sh
.\stop.ps1                   # 停止（等价于在运行终端按 Ctrl+C）
```

手动方式，Windows PowerShell：

```powershell
cd backend
py -3.11 -m venv .venv                       # 版本见 .python-version
.\.venv\Scripts\Activate.ps1                 # 必须带路径，Activate.ps1 不在 PATH 中
python -m pip install --upgrade pip          # 旧版 pip 请先升级
pip install -r requirements-dev.lock.txt     # 锁定版本；改依赖后重新生成锁文件

$env:DGB_ADMIN_TOKEN = "dev-token"           # PowerShell 环境变量写法
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

macOS / Linux / Git Bash：

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.lock.txt     # 与 Windows 使用同一份锁文件

export DGB_ADMIN_TOKEN=dev-token             # `VAR=value cmd` 仅 Bash 支持
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 前端（React + TypeScript + Vite）

```bash
cd frontend
npm install
npm run dev      # 开发模式，Vite 代理 /api 与 /ws 到 127.0.0.1:8000
npm run build    # 构建产物 frontend/dist，由 FastAPI 托管
```

### 验证

```bash
cd backend
python -m pytest            # 当前 114 项测试
alembic upgrade head        # 版本化迁移
```

关键环境变量：

| 变量 | 用途 |
| --- | --- |
| `DGB_DATABASE_URL`、`DGB_ADMIN_TOKEN` | 数据库与管理 Token |
| `DGB_INFERENCE_HZ` | 风险快照产生频率（默认 5 Hz，页面实际刷新率） |
| `DGB_INFERENCE_MODE` | `fake`（脚本回放）/ `live`（连接模型侧 live 服务） |
| `DGB_LIVE_SOURCE` | `frames`（JPEG 序列）/ `camera`（本地摄像头） |
| `DGB_LIVE_URL`、`DGB_LIVE_TOKEN_FILE` | live 服务地址与 Token 文件 |
| `DGB_LIVE_VIEW` / `DGB_LIVE_ROI` / `DGB_LIVE_IMAGE_SIZE` | 机位（`body`/`face`）与裁剪区域 |
| `DGB_CAMERA_*`、`DGB_PREVIEW_MAX_FPS`、`DGB_PREVIEW_MAX_CLIENTS` | 采集与预览上限 |

更多排障说明见 [backend/README.md](backend/README.md)。
## 快速开始（模型侧）

模型侧在独立环境中运行，不与应用侧共用解释器。三种已验证的部署形态：

| 形态 | 环境 | 入口 |
| --- | --- | --- |
| 本机 CPU 联调 | `.venv` + `.venv-perception` + `.venv-detector` | [LOCAL_CPU_DEV.md](docs/deployment/LOCAL_CPU_DEV.md) |
| Ascend NPU 服务器 | 系统 Python 3.12（torch 2.10 + torch_npu 2.10）+ `.venv-yolo-npu` + `.venv-perception-arm64` | [SERVER_SETUP.md](docs/training/SERVER_SETUP.md) |
| 训练 | 同上，`torch_npu` 已预装，**不要**执行会降级 torch 的 `requirements-npu.txt` | [RELATION_RUNBOOK.md](docs/training/RELATION_RUNBOOK.md) |

NPU 上启动组合推理（检测器 + 感知 + 模型）：

```bash
# 每次使用新的 SESSION_DIR；bundle 需按当前源码重新导出
python3 scripts/local/prepare_dev_bundle.py \
  --output runs/live-local/bundle-npu --backend npu --device-index 0 \
  --detector-token-file runs/live-local/detector.token

PYTHON_BIN="$(command -v python3)" \
DETECTOR_PYTHON="$PWD/.venv-yolo-npu/bin/python" \
PERCEPTION_PYTHON="$PWD/.venv-perception-arm64/bin/python" \
BUNDLE_DIR="$PWD/runs/live-local/bundle-npu" \
SESSION_DIR="$PWD/runs/live-session-001" \
BACKEND=npu DEVICE_INDEX=0 \
bash scripts/server/start_live_composite.sh
```

训练入口（设备不可用时会明确失败，不回退 CPU）：

```bash
python3 -m training.temporal.npu_train --check-config   # 只校验配置
python3 -m training.relation.smoke --output-dir runs/relation-smoke-001   # CPU 合成闭环
```

## 实时监控链路

```
CameraSource（25 Hz 网格，丢时间片不追帧）
  ├─ PreviewTap ──► MJPEG 预览（X-Seq / X-Sha256，与推理同字节）
  └─ LiveInference ──HTTP 帧──► live 服务（检测器 + 感知 + 组合模型 / NPU）
                                     │
                          RiskSnapshot（含 task_details 六参数）
                                     ▼
                      WebSocket ──► Dashboard（风险 / 预览 / 判决依据）
```

- 预览默认关闭，需在设置页打开 `preview.enabled`（需要管理员 Token）；离线回放期间不提供实时预览。
- 快照产生率由 `DGB_INFERENCE_HZ` 决定（默认 5 Hz），模型侧实际结果率取决于推理耗时（NPU 上约 3.8 Hz，CPU 更低）。
- `task_details` 暴露 `available / logit / score / threshold_logit / detected / source` 与派生 `margin = logit - threshold_logit`，用于回答“为什么是/不是 DANGER”。
- live 服务只监听 `127.0.0.1`；应用在另一台机器时需要通过 SSH 隧道（`ssh -N -L 8765:127.0.0.1:8765 <host>`）。

## 核心能力

- 疲劳、长时间闭眼、异常眨眼与打哈欠检测
- 手机、饮水等操作分心识别
- 头部姿态与视觉注意力分析
- TCN 时序行为识别与多模型风险融合
- `SAFE`、`WARNING`、`DANGER` 三级风险评估
- 本地实时告警、危险事件记录与 Web 可视化
- 实时摄像头预览，预览帧与推理输入逐帧同源（序号 + sha256）
- 逐任务判决参数（logit / 阈值 / margin / score / 来源），可解释“为什么触发”

> 边缘加速：已在 **Ascend 910B** 上完成实时组合推理验证；ONNX / TensorRT / Jetson 属于目标方案，尚未验证。
> 识别精度尚未验收，`drinking` 等分支存在已知误报，参见[已知限制](docs/testing/KNOWN_LIMITATIONS.md)。

## 工作流程

```text
Camera
  → Object / Face / Head Pose Perception
  → EAR / MAR / PERCLOS / Detection Features
  → Sliding Window + TCN
  → Risk Fusion
  → Local Alarm + FastAPI + Web Dashboard
```

单帧检测结果不会直接作为最终行为结论。例如，检测到手机只表示手机出现在画面中；是否正在使用手机，需要结合连续帧特征和时序模型判断。

## 技术栈

| 领域      | 技术                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------- |
| AI 与训练 | Python 3.11、[PyTorch](https://pytorch.org/)                                                       |
| 视频处理  | [OpenCV](https://opencv.org/)、[GStreamer](https://gstreamer.freedesktop.org/)                      |
| 目标检测  | YOLO 系列轻量实时检测模型                                                                         |
| 人脸分析  | [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) |
| 时序建模  | [Temporal Convolutional Network（TCN）](https://arxiv.org/abs/1608.08242)                          |
| 模型部署  | Ascend CANN / torch_npu（已验证）；ONNX Runtime / TensorRT（目标）                                              |
| 后端      | [FastAPI](https://fastapi.tiangolo.com/)、REST API、WebSocket                                      |
| 前端      | [React](https://react.dev/)、TypeScript、Vite、ECharts                                             |
| 数据库    | SQLite；多设备/云端场景预留 PostgreSQL 迁移路径                                                   |
| 边缘平台  | Ascend 910B（已验证）、NVIDIA GPU / Jetson Orin（未验证）                                         |

## 数据集与模型

第一阶段使用 [DMD（Driver Monitoring Dataset）](https://dmd.vicomtech.org/) 进行驾驶员状态和行为建模。原始数据包含 **Face / Body 两个机位**（Hands 是感知专家输入，不是独立机位）及疲劳、分心、视线和动作标注，详见 [DMD 论文](https://arxiv.org/abs/2008.12085)；实际可用范围以[当前公开版本](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset)和已下载分片为准。

实时推理只支持这两个机位：`view=body`（驾驶位上半身，跑 gap80 + C_fusion + 眼动分支）与 `view=face`（面部特写，只有 coverage25）。一次会话只能选一个，不能并行。

数据按照驾驶员主体划分训练集、验证集和测试集，避免同一驾驶员跨集合造成数据泄漏。

主要模型与特征包括：

- YOLO：手机、瓶、杯等交互物体检测
- MediaPipe Face Landmarker：面部关键点提取
- EAR / MAR / PERCLOS：眼睛、嘴部和疲劳特征
- Head Pose：Yaw、Pitch、Roll 姿态估计
- TCN：3–5 秒滑动窗口的时序行为识别
- Risk Fusion：融合行为概率、持续时间与输入可靠性

## Edge-First 与隐私

核心推理和告警默认在车辆边缘设备本地完成：

- 断网时仍可继续检测和告警
- 原始车内视频默认不上传云端
- 默认只保存危险事件元数据
- 视频片段保存属于可选配置

## 文档

- [服务器初始化：文件同步、NPU 依赖与最小测试](docs/training/SERVER_SETUP.md)
- [完整文档索引](docs/README.md)
- [PPT 全方案对齐](docs/governance/PPT_ALIGNMENT.md)（多专家模型、前后端选型与部署）
- [详细需求](docs/requirements/REQUIREMENTS.md)
- [架构设计](docs/architecture/ARCHITECTURE.md)
- [项目 Spec](docs/specs/PROJECT_SPEC.md)
- [数据库设计](docs/database/DATABASE_DESIGN.md)
- [API 设计](docs/api/API_DESIGN.md) / [OpenAPI](docs/api/openapi.yaml)
- [测试策略](docs/testing/TEST_STRATEGY.md)
- [文档协作规范](docs/governance/DOCUMENTATION_GOVERNANCE.md)
- [实现状态](docs/implementation/MVP_STATUS.md)

## 目标

DriveGuard AI 的目标是构建一套实时、连续、可解释、低延迟、可边缘部署的驾驶员安全监测系统。

**Drive safely. Guard every journey.**

## License

[MIT License](LICENSE)
