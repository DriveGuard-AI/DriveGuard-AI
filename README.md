# 🛡️ DriveGuard AI

### 驾安智护：基于多模型感知与时序分析的危险驾驶检测系统

DriveGuard AI 是一套面向真实车内环境的驾驶员监测系统（DMS）。系统结合计算机视觉、面部状态分析与时序行为建模，识别疲劳驾驶、手机使用、饮水、低头和视觉分心等风险。

它关注的不只是某一帧出现了什么，而是：

> **驾驶员过去几秒钟的行为是否正在形成驾驶风险。**

> 当前状态：**MVP 前后端骨架已可运行**；**模型识别与判断尚未实现**，推理由脚本化回放提供（`inference_mode = fake`），详见[实现状态](docs/implementation/MVP_STATUS.md)。

## 快速开始

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
python -m pytest -q          # 40 项测试
alembic upgrade head         # 版本化迁移
```

关键环境变量：`DGB_DATABASE_URL`、`DGB_ADMIN_TOKEN`、`DGB_INFERENCE_HZ`、`DGB_INFERENCE_SCENARIO`（`default` / `safe` / `fatigue` / `degraded`）。

更多排障说明见 [backend/README.md](backend/README.md)。

## 核心能力

- 疲劳、长时间闭眼、异常眨眼与打哈欠检测
- 手机、饮水等操作分心识别
- 头部姿态与视觉注意力分析
- TCN 时序行为识别与多模型风险融合
- `SAFE`、`WARNING`、`DANGER` 三级风险评估
- 本地实时告警、危险事件记录与 Web 可视化
- ONNX / TensorRT 加速和 NVIDIA Jetson 边缘部署

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
| 模型部署  | [ONNX Runtime](https://onnxruntime.ai/)、[TensorRT](https://developer.nvidia.com/tensorrt)          |
| 后端      | [FastAPI](https://fastapi.tiangolo.com/)、REST API、WebSocket                                      |
| 前端      | [React](https://react.dev/)、TypeScript、Vite、ECharts                                             |
| 数据库    | SQLite；多设备/云端场景预留 PostgreSQL 迁移路径                                                   |
| 边缘平台  | NVIDIA GPU、Jetson Orin                                                                           |

## 数据集与模型

第一阶段使用 [DMD（Driver Monitoring Dataset）](https://dmd.vicomtech.org/) 进行驾驶员状态和行为建模。DMD 提供 Face、Body、Hands 等车内视角及疲劳、分心、视线和驾驶动作相关标注，详见 [DMD 论文](https://arxiv.org/abs/2008.12085)。

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

- [完整文档索引](docs/README.md)
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
