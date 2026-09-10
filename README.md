# 🛡️ DriveGuard AI

### 驾安智护：基于多模型感知与时序分析的危险驾驶检测系统

DriveGuard AI 是一套面向真实车内环境的驾驶员监测系统（DMS）。系统结合计算机视觉、面部状态分析与时序行为建模，识别疲劳驾驶、手机使用、饮水、低头和视觉分心等风险。

它关注的不只是某一帧出现了什么，而是：

> **驾驶员过去几秒钟的行为是否正在形成驾驶风险。**

> 当前状态：**Planning / Early Development**。仓库正在建立需求、架构和 MVP 基础能力，暂未提供可运行版本。

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

## 目标

DriveGuard AI 的目标是构建一套实时、连续、可解释、低延迟、可边缘部署的驾驶员安全监测系统。

**Drive safely. Guard every journey.**

## License

[MIT License](LICENSE)
