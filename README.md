# 🛡️ DriveGuard AI

### 驾安智护 —— 基于多模型感知与时序分析的危险驾驶安全检测系统

DriveGuard AI 是一套面向真实车内环境的 **Driver Monitoring System（DMS）**，通过计算机视觉、驾驶员面部状态分析和时序行为建模，对疲劳驾驶、视觉分心和操作分心等危险状态进行实时识别。

项目不仅关注“画面中出现了什么”，还关注：

> **驾驶员过去几秒钟正在做什么，以及这种行为是否已经形成驾驶风险。**

因此，系统由传统的：

`Single-frame Detection + Rule-based Decision`

升级为：

`Multi-model Perception + Temporal Understanding + Risk Fusion`

---

## ✨ Project Overview

传统危险驾驶检测项目通常采用目标检测网络识别闭眼、张嘴、手机或香烟等目标，再通过固定阈值直接判断驾驶员是否存在危险行为。

这种方案存在几个明显问题：

* 单帧检测无法准确理解持续性驾驶行为；
* 正常眨眼、短暂低头等行为容易造成误报警；
* PERCLOS 等固定阈值难以适应不同驾驶员；
* 面部、头部、手机和手部信息彼此独立；
* 检测到手机并不等于驾驶员正在使用手机；
* 算法、GUI 和业务逻辑往往高度耦合，难以部署和扩展。

DriveGuard AI 在此基础上重新设计整个系统架构，通过不同专家模块分别获取驾驶员状态，再利用时序模型进行统一行为理解。

---

# 🎯 Project Goals

DriveGuard AI 的目标是构建一套完整的实时驾驶员安全监测系统，实现：

* 疲劳驾驶检测
* 长时间闭眼检测
* 异常眨眼检测
* 打哈欠检测
* 低头与异常头部姿态检测
* 手机使用检测
* 喝水等操作分心检测
* 视觉注意力偏移检测
* 时序危险行为识别
* 综合风险等级评估
* 实时报警
* 历史危险事件记录
* Web 可视化监控
* GPU / Jetson 边缘部署

最终系统输出三级风险状态：

```text
Safe
Warning
Danger
```

并同时给出主要风险来源和模型置信度。

---

# 🧠 Core Idea

DriveGuard AI 将驾驶员安全检测划分为三个不同层次：

```text
Perception
    ↓
Temporal Understanding
    ↓
Risk Decision
```

也就是：

### 1. Perception

观察驾驶员当前状态：

* 眼睛是否闭合？
* 嘴部是否张开？
* 头部朝向哪里？
* 是否出现手机？
* 是否出现水瓶？
* 是否存在相关交互物体？

### 2. Temporal Understanding

分析过去连续数秒的状态变化：

* 是普通眨眼还是长时间闭眼？
* 是普通张嘴还是完整打哈欠过程？
* 手机只是短暂出现，还是持续使用？
* 驾驶员是否持续低头或偏离道路？

### 3. Risk Decision

融合多个模型结果，判断：

```text
Fatigue
Phone Use
Drinking
Distraction
Eyes Off Road
...
```

并进一步计算综合驾驶风险。

---

# 🏗️ System Architecture

整体系统由五个主要层级组成：

```text
                     In-Vehicle Camera
                            │
                            ▼
                  ┌──────────────────┐
                  │  Video Capture   │
                  │ OpenCV/GStreamer │
                  └────────┬─────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │     AI Perception       │
              │                         │
              │  Object Detector        │
              │  Face Landmarker        │
              │  Head Pose              │
              │  EAR / MAR / PERCLOS    │
              └────────────┬────────────┘
                           │
                           ▼
                  Frame-Level Features
                           │
                           ▼
                  ┌────────────────┐
                  │      TCN       │
                  │ Temporal Model │
                  └───────┬────────┘
                          │
                          ▼
                    Risk Fusion
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Local Warning             Backend API
                                  FastAPI
                                      │
                         ┌────────────┼────────────┐
                         ▼            ▼            ▼
                    WebSocket       REST API     Database
                         │
                         ▼
                React Web Dashboard
```

---

# 🤖 AI Pipeline

## Object Detection

目标检测模块用于识别与危险驾驶行为相关的物体，例如：

```text
Phone
Bottle
Cup
Cigarette
Other Interactive Objects
```

第一版计划采用轻量级 **YOLO 系列实时目标检测模型**。

目标检测模块只负责回答：

> **驾驶员附近出现了什么物体？**

而不会直接将：

```text
Phone detected
```

判断为：

```text
Phone use
```

行为判断由后续时序模型完成。

---

## Face Analysis

系统通过 **MediaPipe Face Landmarker** 提取驾驶员面部关键点。

基于 landmarks 计算：

### Eye Aspect Ratio

```text
EAR
```

用于描述眼睛闭合程度。

### Mouth Aspect Ratio

```text
MAR
```

用于描述嘴部张开程度。

### PERCLOS

统计一定时间窗口内驾驶员眼睛闭合比例：

```text
PERCLOS
```

同时提取：

```text
Blink Frequency
Blink Duration
Eye Closure Duration
Yawn Duration
```

---

## Head Pose

利用人脸关键点估计：

```text
Yaw
Pitch
Roll
```

用于识别：

* 长时间低头
* 转头
* 偏离前方道路
* 可能的视觉分心行为

后续版本将进一步加入 Gaze / Eyes-Off-Road 模块。

---

# ⏱️ Temporal Behavior Recognition

危险驾驶行为通常不是单帧事件。

例如：

```text
Closed Eye
```

可能只是一次正常眨眼。

因此系统不会直接根据某一帧做最终风险判断。

每一帧首先生成统一特征：

```text
x(t) = [
    EAR,
    MAR,
    PERCLOS,
    Blink,
    Yaw,
    Pitch,
    Roll,
    Phone Confidence,
    Bottle Confidence,
    ...
]
```

连续若干帧组成：

```text
X = [x(t-N+1), ..., x(t)]
```

第一版计划采用：

```text
3–5 second sliding window
```

随后送入：

**Temporal Convolutional Network（TCN）**

进行驾驶行为建模。

TCN 输出多个危险状态概率：

```text
Fatigue
Phone Use
Drinking
Distraction
...
```

---

# ⚠️ Risk Fusion

系统不会仅输出一个简单的：

```text
Danger = True
```

而是首先获得多个风险概率：

```text
Fatigue       0.84
Phone Use     0.12
Drinking      0.03
Distraction   0.67
```

随后通过 Risk Fusion 模块得到综合风险等级：

```text
SAFE
WARNING
DANGER
```

第一阶段使用基于概率、置信度和持续时间的规则融合。

后续计划升级为：

```text
MLP Fusion
Gated Fusion
Reliability-Aware Fusion
```

实现多专家自适应决策。

---

# 📊 Dataset

项目第一阶段主要采用：

## DMD — Driver Monitoring Dataset

DMD 是面向 Driver Monitoring System 的驾驶员状态监测数据集。

数据包含多个车内视角，例如：

```text
Face
Body
Hands
```

并提供与以下任务相关的标注：

* Drowsiness
* Distraction
* Gaze
* Hands
* Driver Actions
* Temporal Events

不同视角在系统中的作用：

| View                | Usage                                   |
| ------------------- | --------------------------------------- |
| Face                | EAR / MAR / Head Pose / Fatigue         |
| Body                | Driver action analysis                  |
| Hands               | Object interaction / manual distraction |
| Temporal Annotation | TCN training                            |

训练集、验证集和测试集将按照 **Driver Subject** 进行划分，而不是随机按照视频帧划分。

即：

```text
Train Drivers
≠
Validation Drivers
≠
Test Drivers
```

以避免模型记住驾驶员身份和车内背景造成测试指标虚高。

---

# 🖥️ Frontend

系统使用：

```text
React
TypeScript
Vite
ECharts
```

构建 Web Dashboard。

计划包含：

### Real-Time Monitoring

显示：

* 实时驾驶员视频
* Detection Bounding Boxes
* Fatigue Score
* Distraction Score
* PERCLOS
* EAR
* MAR
* Head Pose
* 当前风险等级

### Event History

记录：

```text
Timestamp
Event Type
Confidence
Duration
Risk Level
```

### Statistics

统计：

* 危险事件数量
* 疲劳事件次数
* 分心事件次数
* 行为持续时间
* 风险趋势

### System Settings

配置：

* Camera
* Alarm Delay
* Threshold
* Model Path
* Model Version

---

# ⚙️ Backend

后端采用：

```text
FastAPI
```

提供两种通信方式。

## REST API

用于：

* 历史事件查询
* 系统配置
* 统计信息
* 模型信息
* 系统状态

## WebSocket

用于实时推送：

```text
Fatigue Score
Distraction Score
Risk Level
EAR
MAR
PERCLOS
Head Pose
Detection Results
```

---

# 💾 Database

MVP 阶段采用：

```text
SQLite
```

主要记录危险驾驶事件：

```text
event_id
timestamp
event_type
confidence
duration
risk_level
```

后续如果扩展到多车辆或云端管理，可以迁移至：

```text
PostgreSQL
```

---

# 🧰 Technology Stack

| Layer                   | Technology                |
| ----------------------- | ------------------------- |
| Language                | Python 3.11               |
| Deep Learning           | PyTorch 2.x               |
| Dataset                 | DMD                       |
| Video Processing        | OpenCV / GStreamer        |
| Object Detection        | YOLO Series               |
| Face Analysis           | MediaPipe Face Landmarker |
| Fatigue Features        | EAR / MAR / PERCLOS       |
| Temporal Model          | TCN                       |
| Backend                 | FastAPI                   |
| Real-Time Communication | WebSocket                 |
| Frontend                | React + TypeScript + Vite |
| Visualization           | ECharts                   |
| Database                | SQLite / PostgreSQL       |
| Model Runtime           | ONNX Runtime              |
| GPU Acceleration        | TensorRT                  |
| Container               | Docker / Docker Compose   |
| Edge Platform           | NVIDIA Jetson Orin        |
| Version Control         | Git / GitHub              |

---

# 📦 Recommended Repository Structure

```text
DriveGuard-AI/
│
├── inference/
│   ├── detector/
│   ├── face/
│   ├── temporal/
│   └── fusion/
│
├── backend/
│   ├── api/
│   ├── websocket/
│   ├── database/
│   └── services/
│
├── frontend/
│   └── src/
│
├── data/
│   ├── dmd/
│   ├── annotations/
│   └── features/
│
├── training/
│   ├── detector/
│   └── temporal/
│
├── models/
│
├── configs/
│
├── deployment/
│   ├── docker/
│   ├── onnx/
│   └── tensorrt/
│
├── tests/
│
├── docs/
│
└── README.md
```

---

# 🚀 Deployment

DriveGuard AI 采用 **Edge-First** 架构。

核心驾驶风险判断在车辆本地完成：

```text
Camera
   ↓
Edge AI Device
   ↓
AI Inference
   ↓
Risk Detection
   ↓
Local Warning
```

不会依赖：

```text
Camera
   ↓
Internet
   ↓
Cloud AI
   ↓
Internet
   ↓
Warning
```

这样可以：

* 降低报警延迟
* 在断网环境继续工作
* 减少视频上传
* 降低带宽消耗
* 提高车内隐私保护

---

# ⚡ Model Deployment Pipeline

训练阶段：

```text
PyTorch
```

部署阶段：

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime / TensorRT
   ↓
NVIDIA GPU / Jetson
```

开发环境优先采用 NVIDIA RTX GPU。

边缘原型计划部署在：

```text
NVIDIA Jetson Orin Series
```

---

# 📈 Evaluation

项目不会只采用 Accuracy 评价系统。

主要指标包括：

```text
Precision
Recall
F1 Score
False Alarm / Hour
Detection Delay
FPS
```

尤其关注：

### Dangerous Behavior Recall

减少危险事件漏检。

### False Alarm Rate

避免系统频繁误报警。

### Detection Delay

评价危险行为出现后系统需要多久触发报警。

### Real-Time Performance

系统目标：

```text
≥ 25 FPS
```

---

# 🔒 Privacy

DriveGuard AI 默认采用本地边缘推理。

设计原则包括：

* 原始车内视频默认不上传云端；
* 风险判断在本地完成；
* 默认仅保存事件元数据；
* 视频片段存储作为可配置功能；
* 网络不可用时核心检测功能仍应正常工作。

---

# 🗺️ Roadmap

## Phase 1 — MVP

* [ ] DMD 数据解析
* [ ] Subject-independent 数据切分
* [ ] Face Landmarker
* [ ] EAR / MAR
* [ ] PERCLOS
* [ ] Head Pose
* [ ] Phone / Bottle Detection
* [ ] Frame Feature Extraction
* [ ] TCN Training
* [ ] Fatigue Detection
* [ ] Phone Use Detection
* [ ] Drinking Detection
* [ ] Distraction Detection
* [ ] Risk Fusion
* [ ] FastAPI Backend
* [ ] WebSocket
* [ ] React Dashboard
* [ ] SQLite Event Logging
* [ ] Real-Time Alarm

## Phase 2 — Optimization

* [ ] Gaze Estimation
* [ ] Eyes-Off-Road Detection
* [ ] Hand Landmark / Pose Model
* [ ] Hands-Off-Wheel Detection
* [ ] Gated Risk Fusion
* [ ] ONNX Optimization
* [ ] TensorRT FP16
* [ ] Jetson Deployment
* [ ] Low-Light Robustness
* [ ] Occlusion Robustness

## Phase 3 — Research

* [ ] Multi-camera Fusion
* [ ] Cross-driver Generalization
* [ ] Cross-camera Generalization
* [ ] Video Transformer
* [ ] Reliability-Aware Expert Fusion
* [ ] Driver Risk Prediction

---

# 📌 Current Status

> **Project Status: Planning / Early Development**

当前阶段主要工作：

1. 完成系统需求与技术架构设计；
2. 整理 DMD 数据集；
3. 建立 Face / Object / Temporal 三条基础模型链；
4. 建立第一版可实时运行的 MVP。

---

# 🎯 Vision

DriveGuard AI 希望解决的不只是：

> **“这一帧驾驶员是否闭眼？”**

而是：

> **“驾驶员过去几秒钟的行为是否正在形成驾驶风险？”**

最终目标是建立一套：

**实时、连续、可解释、低延迟、可边缘部署的驾驶员安全监测系统。**

---

## DriveGuard AI

**Drive safely. Guard every journey.**

驾安智护，让 AI 守护每一段旅程。
