# YOLOX / MediaPipe 本地视频特征入口

2026-09-19更新 · shadow · model。**三个真实工件与独立Windows CPU环境已准备，真实模型加载/空白帧前向和105项测试通过；真实视频尚未验证。** 本轮没有修改原训练环境、处理DMD或启动训练。位置、版本快照、已知pip依赖警告见[本机准备记录](PERCEPTION_SETUP.md)，工件来源/hash/许可见[预训练清单](../../models/PRETRAINED_ASSETS.md)。

## 1. 已接入的模块

| 模块 | 当前实现 | 尚未证明的能力 |
| --- | --- | --- |
| YOLOX | 0.3.0 内置 yolox-nano / yolox-tiny；COCO80 中 cell phone/bottle/cup 映射；FP32 前向，原图框、分数 | 真实精度、DMD 微调、CUDA/NPU 算子兼容性 |
| Face Tasks | CPU VIDEO，最多2张脸，478/468关键点转原图 xy；输出供 EAR/MAR 等使用 | 自动驾驶员身份、头姿、视线方向、疲劳结论 |
| Hand Tasks | CPU VIDEO，最多4只手，21关键点转原图 xy；人工归属限定 | 自动手部归属、遮挡恢复、物理接触证明 |
| 跟踪 | 同类别、IoU≥0.3且双向唯一才沿用 ID；空检测、歧义或间隔>80 ms分配新身份 | 多目标跟踪指标；交叉或替换物体仍可能误关联 |
| 视频入口 | 本地单视频流、原始PTS、25Hz因果选帧、流式特征导出 | 摄像头/网络流/实时性能、真实解码器兼容性 |

代码入口：[YOLOX](../../inference/experts/yolox_adapter.py)、[MediaPipe](../../inference/experts/mediapipe_adapter.py)、[视频 CLI](../../inference/perception/video.py)。输出仍遵循[特征桥接](PERCEPTION_BRIDGE.md)：19个值+19个mask，TCN 输入 `[B,100,38]`。

YOLOX 不采用 legacy RGB/归一化变换：BGR uint8 按比例缩放后放在114填充图左上，转换 CHW FP32；分数为 objectness×最高类别概率，类别内 NMS 后恢复原图坐标。前向只在指定 cpu/cuda/npu 上运行，不可用就报错；**CPU NumPy 后处理是显式设计边界，不是将失败的 NPU 前向悄悄回退。** 实现口径见 [YOLOX 0.3.0 官方预处理](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/yolox/data/data_augment.py)及[推理示例](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/tools/demo.py)。

MediaPipe 将 BGR 转成 RGB SRGB，调用 `detect_for_video`；Face/Hand 是独立任务。当前不输出变换矩阵推导的头姿，三个头姿特征 mask=0。它的 CPU delegate 与 PyTorch/torch_npu 无关，不能因为 NPU 可用就认为 MediaPipe 在 NPU 上运行。[Face Tasks](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)、[Hand Tasks](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)

## 2. 工件和配置

复制并编辑[配置示例](../../configs/perception/runtime.example.json)为同目录 `runtime.local.json`（Git 忽略）：

- `recording_id`、`view`：真实录制标识和单视角，不把三个不同相机画面拼成同一录制。
- `driver_roi`：**原始解码图**上的 `[x1,y1,x2,y2]`；示例1280×720不是自动标定值，必须按实际视频调整。当前不裁剪、镜像或自动旋转；若视频带旋转标记，按原始解码栅格设置 ROI。
- `driver_region_confirmed_exclusive`：默认false。只有逐录制人工确认区域内脸/手专属于驾驶员时才能设true；乘客可能伸手进入时不能作此声明。检测到多张 ROI 内人脸或无效面部结果时撤销该帧手部归属。它不是自动身份识别，漏检乘客仍是风险。
- 每个工件必须填写 `path`、小写64位 `sha256`、精确 `source` 与 `license`。相对路径相对于**配置文件目录**，与当前 shell 所在目录无关。全零hash/REPLACE字段只是模板，不能用于实际运行。
- `objects` 选择 nano/tiny、cpu/cuda/npu、device_index、input_size及阈值；不支持自定义可执行 exp.py、不同类别数量或 legacy 权重。

本地临时下载可放 `models/local/`（忽略），也可放服务器外置目录。**正式开源权重仍放 models/releases 或其公开发布地址，不能以忽略本地副本代替公开交付。** 代码许可不自动涵盖权重和传递依赖，来源与再发布权限仍需逐工件核对；校验hash只能证明文件一致，不能证明许可已审核。

Linux 计算工件校验和示例：

```bash
sha256sum /absolute/path/yolox_nano.pth /absolute/path/face_landmarker.task /absolute/path/hand_landmarker.task
```

程序只加载显式提供的本地文件，绝不自动下载。YOLOX checkpoint 必须能被 `torch.load(weights_only=True)` 安全读取且含严格匹配的 `model` tensor state_dict；不兼容时失败，不回退到任意 pickle 执行。

## 3. 依赖：与 NPU 训练分开

[requirements-perception.txt](../../requirements-perception.txt) 已在独立Windows x64/Python3.12 CPU环境完成安装及模型加载检查，但**不是完整YOLOX环境或跨平台lockfile**；上游ONNX和OpenCV包名的4项pip元数据警告见[准备记录](PERCEPTION_SETUP.md)。精确版本另存[Windows CPU快照](../../requirements-perception-win-cpu.lock.txt)。保留原 requirements.txt / requirements-npu.txt；不要直接安装进已工作的昇腾环境。先独立提取，再把特征数组传给训练服务器，不要求同一机器完成全部模块。

候选包：MediaPipe0.10.21、OpenCV-contrib4.11.0.86、PyAV12.3.0、NumPy1.26.4，以及现有 torch2.7.1 配套 torchvision0.22.1。选择单个 OpenCV distribution，避免多个包覆盖同一个 `cv2`。版本依据为[MediaPipe发布元数据](https://pypi.org/pypi/mediapipe/0.10.21/json)、[OpenCV包说明](https://pypi.org/project/opencv-contrib-python/4.11.0.86/)、[PyAV包](https://pypi.org/project/av/12.3.0/)与[PyTorch版本配对](https://pytorch.org/get-started/previous-versions/)；不同系统/架构的 wheel、ABI 与全部传递依赖仍需验证。

以下是安装步骤参考，本机已按此方式完成；具体已安装版本和验证范围见准备记录。在**独立 CPU 环境**中先选择 CPU wheels，避免默认拉取大体积 CUDA 包：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-perception.txt
```

YOLOX 本体需另备[官方0.3.0源码](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0)，审核来源后从本地目录安装，不由代码下载。上游 requirements 固定了较老的 ONNX 导出依赖，本入口不使用它们，不要求照搬整份上游训练/导出环境；参考[上游依赖](https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/requirements.txt)。本清单补充其常见 Python 导入依赖，实际安装/编译是否成功需单独记录，例如本地源码安装：

```bash
python -m pip install --no-deps --no-build-isolation -e /absolute/path/to/reviewed-YOLOX-0.3.0
python -m pip check
python -c "import torch, torchvision, yolox, mediapipe, cv2, av; print(torch.__version__, torchvision.__version__, yolox.__version__, mediapipe.__version__, cv2.__version__, av.__version__)"
```

`pip check` 可能报告 YOLOX 上游已声明但本入口未使用的导出依赖或 `opencv-python` 包名依赖；不要自动忽略全部错误。分别核对运行时必需导入与未启用组件、记录环境差异，真实小视频通过后再锁定环境。NPU 上 torchvision 导入/ABI、YOLOX前向算子仍必须单独验证，当前不承诺这套候选可直接用于昇腾。

## 4. 运行方式

从仓库根目录执行。第一条只校验配置结构，即使没有任何第三方依赖、权重或视频也可以运行：

```bash
python -S -m inference.perception.video --config configs/perception/runtime.example.json --check-config
```

返回 `config_structure_valid` 与 `assets_verified:false`，**不表示文件存在、hash匹配或环境可运行**。

填好真实工件、安装并验证独立环境后：

```bash
python -m inference.perception.video --config configs/perception/runtime.local.json --input /absolute/path/recording.mp4 --output-dir data/features/recording-001
```

输入限本地、独立存储的 MOV/MP4、Matroska/WebM 或 AVI 容器，且恰好一个视频流，至少产生100个40ms采样槽。打开前限定 demuxer 和 file 协议，并关闭 MOV 外部轨道引用；不接受 ffconcat/HLS 等播放列表，避免只校验列表文件却实际读取其他素材。约束依据见 [FFmpeg format_whitelist 与外部轨道选项](https://ffmpeg.org/ffmpeg-formats.html)。输出目录必须不存在，失败目录保留供排查，换新目录重跑，不自动清空。真实模型初始化/推理、解码、资源关闭或索引发布任一步失败均不发布成功 `record.json`。

时钟使用 [PyAV Frame.pts/time_base](https://pyav.basswood.io/docs/stable/api/frame.html)，首帧作为0点；相对PTS向上取整到整数毫秒，避免向下取整引入不足1ms的未来观测。缺失/重复/倒退PTS、整数毫秒碰撞、损坏帧或变尺寸直接失败，不按平均FPS猜时间。每槽只取最近过去帧；重复选中同一帧只推理一次，保持原时间戳，超过80ms由特征桥接判未知。只输出到最后解码帧时间，不推测尾帧持续时间。当前为阻塞离线执行，不代表25FPS实测吞吐。

## 5. 输出、验证与下一步

输出4个数组与成功索引，`record.json.provenance` 附源视频hash/大小、原始PTS起点、配置、工件来源/hash/许可声明及实际包版本。model_id绑定工件、设置与运行时签名。工件保持只读，提取期间不要替换；源视频大小/mtime变化会中止发布。长视频只保留少量源帧和流式特征，不展开所有帧到磁盘。

`targets` 与 `label_masks` 全零表示**未知**，不是负例；这不是可直接训练的完整 manifest。还需 DMD 标签映射、主体划分、有效特征覆盖核对以及真实模型回放；头姿mask始终0，手动归属关闭时眼/嘴/手相关特征也缺少有效依据。

```bash
python -m unittest discover -s tests -p "test_expert_*.py" -v
python -m unittest discover -s tests -v
```

unittest替代外部 YOLOX/MediaPipe/PyAV/OpenCV 调用，运行真实安全checkpoint加载、坐标/NMS/跟踪/时间逻辑、适配器、特征计算与导出。2026-09-19另用三个实际工件执行了CPU加载/空白帧前向，但不得将二者计为真实感知精度、GPU/NPU兼容或性能跑分。后续仍需小视频验收、自动驾驶员归属/头姿、检测器GPU/NPU训练、DMD四任务训练、融合及正式权重公开。
