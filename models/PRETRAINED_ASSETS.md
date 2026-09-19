# 本地预训练工件清单

2026-09-19 · shadow。用户批准准备模型和独立感知环境后，从以下官方地址下载。文件位于 Git 忽略的 `models/local/`，用于开发验证；不是本项目自训权重或正式发布。本表 SHA-256 为下载后本地计算的固定值，**不是上游发布者提供的签名/校验声明**。

| 文件 | 固定来源 | 字节数 | 许可记录 |
| --- | --- | ---: | --- |
| yolox_nano.pth | [官方0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) | 7694953 | 权重再发布许可待确认；源码Apache-2.0不自动等于权重许可 |
| face_landmarker.task | [Google float16/version 1](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | 3758596 | 官方组件模型卡声明Apache-2.0，证据见下 |
| hand_landmarker.task | [Google float16/version 1](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) | 7819105 | 官方Hand Tracking模型卡声明Apache-2.0 |

```text
cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b  yolox_nano.pth
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  face_landmarker.task
fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  hand_landmarker.task
```

三个文件合计19,272,654字节（约18.38 MiB）。下载时Google对象generation：Face `1683136941916318`、Hand `1682480004222387`；本地receipt保存在忽略的 `runs/perception-setup-20260919/`。

## YOLOX版本与发布边界

适配的源码版本为 [YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)，固定提交 [419778480ab6ec0590e5d3831b3afb3b46ab2aa3](https://github.com/Megvii-BaseDetection/YOLOX/commit/419778480ab6ec0590e5d3831b3afb3b46ab2aa3)。该版本README非legacy Light Models表指向上述Nano权重，不使用legacy预处理。

源码归档SHA-256：`68e5d3abea0337c92e051816edd9182d78aa67424ac3cfea32664ea8f32e7e81`。下载地址为[官方固定提交归档](https://codeload.github.com/Megvii-BaseDetection/YOLOX/zip/419778480ab6ec0590e5d3831b3afb3b46ab2aa3)。本机安装来源目录 `models/local/yolox-0.3.0-src/`；Windows解压时仅排除 `docs/demo/` 中的文档符号链接，模型Python源码没有修改。

[源码LICENSE](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/LICENSE)明确为Apache-2.0；尚未找到明确覆盖预训练权重再发布的维护者声明，相关[上游许可问题](https://github.com/Megvii-BaseDetection/YOLOX/issues/1865)也未提供本次所需的明确答复。因此配置记录 `pending_permission`，**不把该权重复制进本项目正式发布包**。这不取消项目正式权重必须公开的要求；正式交付前要解决许可或更换来源清楚的工件。

## MediaPipe模型卡证据与使用限制

Face bundle由[官方概览](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)链接以下组件卡：[BlazeFace](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf)、[Face Mesh V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)、[Blendshape V2](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf)。模型卡各自声明Apache-2.0，不能混同于网页文字的CC-BY许可。模型卡不是version1二进制的签名证明，二进制身份以固定URL、generation和实际hash共同记录。

手部管线见[官方Hand Tracking模型卡](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20%28Lite_Full%29%20with%20Fairness%20Oct%202021.pdf)，覆盖检测和关键点模型并声明Apache-2.0。

Face Mesh模型卡明确不用于生命攸关的决策，且对视角、遮挡、距离和照明存在限制。本项目当前为研发原型，模型加载通过不等于车载安全认证、驾驶员识别或危险行为检测有效；后续必须做真实车内场景评估，不能依赖它承担车辆安全控制。

## 本地配置

`configs/perception/runtime.local.json` 已填入这三个工件的相对路径/hash/来源/许可。其640×480 ROI、`setup-smoke-not-a-real-recording`标识仅用于空白帧加载检查；**真实视频运行前必须填写录制标识和实际原图ROI**。驾驶员归属默认false，不自动确认身份。使用方式见[感知指南](../docs/training/EXPERT_RUNTIME.md)。
