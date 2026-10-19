# 本地感知运行时实施计划

2026-09-18 · shadow · model。用户已同意 YOLOX Nano/Tiny + MediaPipe Tasks 接入方案；沿用当前分支和未提交改动，不下载数据/模型，不安装依赖，不启动训练。

**目标：** 显式提供本地工件后，将单视角、未镜像本地视频转换为现有 19 特征/38 通道缓存。工程测试不等于真实模型或 GPU/NPU 验证。

**技术边界：** YOLOX 0.3.0 内置 Nano/Tiny、COCO 80 类；MediaPipe Face/Hand Tasks CPU VIDEO；PyAV 原始 PTS 解码；25 Hz 因果重采样；NumPy CPU NMS 与保守 IoU 跟踪。YOLO 前向可显式选择 cpu/cuda/npu，失败不回退，NPU 算子兼容性仍待实测。头姿保持未知，不从矩阵猜角度。ROI 不证明驾驶员身份；只有人工确认该录制的区域专属于驾驶员时才启用归属，多脸歧义仍失效。

## T1 — 本地工件与 YOLOX

- 新增 `inference/experts/assets.py`：本地路径、SHA-256、来源和许可声明严格校验；不自动下载，禁止任意 pickle 回退；模型标识绑定工件与调用配置。
- 新增 `inference/experts/yolox_adapter.py`：惰性加载依赖，仅支持内置 Nano/Tiny；非 legacy BGR/114 padding/CHW FP32；恢复原图坐标；显式 COCO 类名映射 phone/bottle/cup；分数为 objectness×class confidence；类别内 CPU NMS，输出未跟踪的规范对象。
- `tests/test_expert_assets.py`、`tests/test_expert_yolox.py`：先写失败测试，再实现与回归；覆盖坐标、类别、空预测、非法预测、权重校验和缺失依赖。真实权重测试另记。

## T2 — MediaPipe 与跟踪

- 新增 `inference/experts/mediapipe_adapter.py`：本地 Face/Hand .task、显式 CPU delegate、VIDEO 时间戳、BGR→RGB；归一化 xy 恢复全帧像素；非法/越界关键点不裁剪成假观测；多脸及未知归属保守处理；资源关闭。
- 新增 `inference/perception/tracking.py`：同类别且唯一双向 IoU 匹配才续用 ID；空检测、歧义、超过 80 ms 重置身份。不是经过验证的多目标跟踪器。
- 对应 unittest 先红后绿；测试真实转换逻辑，仅替代外部模型调用边界。

## T3 — 视频与缓存入口

- 新增 `inference/perception/video.py`、`video_source.py`、`runtime.py` 和 `configs/perception/runtime.example.json`。
- 本地视频必须有严格递增 PTS；用首帧归零，不用猜测 FPS 合成时钟；转换为整数毫秒后仍须严格递增。采样槽每 40 ms 选最近过去帧，不取未来帧；同一源帧只推理一次，保留源时间，过期由特征桥接失效；只输出到最后已解码帧，不延长尾部。
- 审查补充：仅允许独立存储 MOV/MP4、Matroska/WebM、AVI，打开前限制 demuxer/file 协议、关闭 MOV 外部轨道，拒绝播放列表带来的多文件溯源缺口。
- 输入尺寸固定，不做镜像/裁剪；手动原图 ROI，明确人工归属声明。不可恢复推理/解码错误中止导出，不伪造空检测。
- 复用原子发布 `export_recording`，成功索引附加来源视频 hash、运行时版本、模型工件与配置溯源；全未知标签不变；旧输出不可覆盖。
- CLI `--check-config` 仅验证结构，不加载依赖/读取权重；运行时先验证工件，再加载实际模型。
- 新增独立感知依赖说明/清单，不改 NPU 训练环境。合成视频帧通过真实适配/特征/导出边界测试，不宣称真实解码器和模型已验证。

## T4 — 审查、验证与文档同步

- 规格审查通过后进行代码质量审查；修复重要发现并回归。
- 全量 unittest、CLI 结构预检、文档静态检查；同步 RUNBOOK、感知文档、总体进度、实验记录和交接。
- 未交付：驾驶员自动归属、头姿/视线求解、真实视频/工件验收、检测器微调及双平台训练、DMD 标注对齐、真实跑分/正式开源权重、融合和应用服务。

## 本轮执行结果

T1–T4工程增量已完成，105项unittest通过（新增43项）；配置无依赖预检、规格/质量复核及文档静态检查通过。详细证据与明确未验证范围见[实验记录](../experiments/20260918-expert-runtime.md)。保持model分支和未提交状态；不把本轮完成等同于完整项目、真实工件或平台验收完成。
