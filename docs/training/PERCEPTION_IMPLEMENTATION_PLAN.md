# 感知特征桥接 Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development；每项先写失败测试再实现，独立规格复核后再进行质量复核。沿用用户指定 model 分支，保留已有未提交修改，不提交、不安装依赖、不恢复真实数据任务。

**Goal:** 把已确认的 YOLO / MediaPipe 专家输出协议落到可测试的特征桥接代码，接入已有 TCN 数组契约；不宣称已完成检测器调用或训练。

**Architecture:** 同一原图坐标系的专家观测 → 无状态几何/归属过滤 → 25 Hz 因果统计 → 19 特征/38 通道的独立候选 profile。输入是每个采样槽的观测，不是原始视频；上游推理、跟踪、主体确认和重采样仍为独立适配工作。

**Tech Stack:** 标准库、NumPy、现有 PyTorch 测试环境；不依赖已安装 YOLO / MediaPipe。不生成图示或下载权重。

## 范围与取舍

沿用完整多专家设计，不把方案缩减为 TCN。选择先实现模型无关的特征桥接：相比立即绑定 YOLO 发行版，能先验证时间、缺失与手物交互语义；相比直接端到端视频训练，不增加数据和设备成本。当前 profile 是工程候选，不是经过数据标定的最终方案。

输入 packet 包含 schema_version=driveguard-experts-v1、recording_id、view、frame_id、timestamp_ms（非负整数）、width/height（正整数）、driver_roi（原图像素 xyxy）、face/objects/hands。专家字段含 status（ok/missing/error）、timestamp_ms、model_id；禁止未来观测。face 含 driver_confirmed、landmarks（468 或 478 个 xy）、head_pose（null 或 yaw_deg/pitch_deg/roll_deg）；objects 含 items（category、box、score、track_id）；hands 含 items（21 个 xy、driver_confirmed、track_id）。失败专家无有效载荷。所有数值有限，未知键拒绝；框/点必须在图内。

固定 profile driveguard-interaction-v1：19 特征，前 12 个保留基线名称，追加 blink_rate_per_min、phone_hand_distance、phone_hand_iou、phone_near_duration_s、drink_hand_distance、drink_hand_iou、drink_near_duration_s。新 profile 的物体候选只取中心在 driver_roi 内的合格框；不可混用旧缓存或 24 通道权重。

EAR 左 [362,385,387,263,373,380]、右 [33,160,158,133,153,144]，(d(p1,p5)+d(p2,p4))/(2*d(p0,p3))；MAR=d(13,14)/d(78,308)。索引采用 MediaPipe 解剖左右；以像素坐标计算，禁止直接在归一化 xy 上算纵横比。头姿只接受已转换为约定坐标的上游结果，不从变换矩阵猜测 gaze。

阈值初值：score>=0.25；专家年龄<=80ms；双眼 EAR 平均<0.2 为闭眼候选；PERCLOS 最近100槽、至少80有效；眨眼使用同一4秒窗口且全部眼部槽有效，只统计完整 open→closed→open，闭眼80–500ms；近邻=手物框最短间隙/ROI对角线<=0.03。所有数值是未标定工程候选。

手必须明确 driver_confirmed 且全部关键点在 driver_roi 内；ROI 不能独立证明手属于驾驶员。最高分合格物体按分数、框坐标、类别、track_id 确定性选取；最近合格手与该同一物体计算距离和 IoU。没有物体时 conf=0/mask=1，距离未知；手缺失时不能推导“未交互”。连续邻近时长需稳定非空物体 track_id，切换目标清零重新累计，缺失后起点未知，直到确定远离再重新邻近。所有连续时长遇缺失重置，不能跨录制/视角继承。

## T1 无状态观测/几何

- [x] 先新增 tests/test_perception_geometry.py：像素纵横比、几何退化、成功无检测与失败区别、ROI 外高分物体、副驾驶/未知手、同分候选、NaN/坏框/未来时间拒绝。
- [x] 运行 `.venv-training/Scripts/python.exe -B -m unittest tests.test_perception_geometry -v`，确认因 inference/perception 尚未实现失败。
- [x] 新建 inference/__init__.py、inference/perception/__init__.py、geometry.py；`spatial_features(packet, profile)` 返回 values、eye_closed、phone_near/drink_near 与选中 track；无效特征以 None 表示。
- [x] 同命令验证通过；只操作该模块和测试，独立规格/质量复核。

## T2 Profile 与因果状态

- [x] tests/test_perception_sequence.py 先验证新 profile/input_channels=38、窗口预热、缺失重置、时间格严格40ms、录制/视角/尺寸/ROI/模型版本固定、无效输入不污染历史。
- [x] 新建 configs/perception/interaction_v1.json、configs/training/temporal_interaction.json、inference/perception/profile.py、sequence.py。
- [x] `FeatureExtractor(profile).step(packet)` 调用几何层，输出 float32 values[19]、uint8 masks[19]；每录制独立实例，必须连续40ms槽，掉帧显式 missing，不插值未来。
- [x] 运行相应 unittest；通过后与 T1 合并复核，不修改既有基线权重或训练行为。

## T3 数组导出与训练接线

- [x] tests/test_perception_export.py 先测试分块流式处理、未知标签全0、不覆盖目录、非法 packet 不产出成功索引；合成四录制接入 FeatureCorpus/TCN 前向。真实峰值内存未测。
- [x] 新建 inference/perception/export.py：`export_recording(packets, output_dir, profile)` 两阶段临时行缓存到新目录，生成 values/masks/targets/label_masks.npy 和 record.json；标签全部未知，不发明 DMD 映射；成功索引最后写入，失败保留不完整工件不删除用户数据；若文件系统拒绝写marker，索引缺失仍表示失败。
- [x] CLI `python -m inference.perception.export --input observations.jsonl --output-dir data/features/new-recording`，流式读取规范 packet，不直接读取视频；输出摘要和完整 profile/hash。
- [x] 集成测试通过，全量 unittest 与现有无依赖训练入口预检；文档同步使用说明、数据契约、设计、计划、工程记录与交接。正式训练前仍需真实标注、版本化专家与硬件验证。

## 证据要求

仅记录实际执行的 CPU/合成验证，不把适配器数据转换称为已运行 YOLO/MediaPipe。新 profile 要与现有基线同时保留，既有29项测试不回退。改动不涉及 API/数据库公开契约，检查后无需改字段。

执行结果：62项全量测试通过（旧29+新33），独立规格/质量复核通过；记录见[感知桥接验证](../experiments/20260918-perception-bridge.md)。本计划范围完成，不等于完整P1/P3或model分支已完成。
