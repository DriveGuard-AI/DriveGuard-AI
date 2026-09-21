# DMD 真实训练准备实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. 按任务测试并审核。

**Goal:** 把审计通过的 DMD 标注转换为可追溯、按被试隔离的训练准备任务与四任务监督；不把合成验证称为真实训练。

**Architecture:** 先输出不解码的准备清单，再以真实解码 PTS 对齐标签。冻结感知模型，沿用现有特征导出和 NPU 训练器。未确认的 ROI 与闭眼时长不自动填写。

**Tech Stack:** Python 标准库、unittest；后续感知执行使用现有 PyAV / NumPy / YOLOX / MediaPipe。

## 本批范围与约束

- 本批交付：标签转换核心、准备清单 CLI、划分/标签配置模板和运行文档。全量感知执行与最终 manifest 发布另一个批次接入，避免未确认 ROI 时启动长时间计算。
- train 被试 `1,5,6,9,10,13,23,29`；留出评估 val 被试 `7,14`。同人所有子集/录制/视角同一划分，无单独 test；val 选择 best.pt，不称独立测试成绩。
- 用户已确认 fatigue = 哈欠 + 持续闭眼相关事件；不是疲劳诊断。闭眼阈值必须显式配置，不默认医学阈值。
- 手机/喝水：显式行为为正，safe_drive 为保守负例；其他、冲突、未标注为未知。
- fatigue：哈欠或已观察达到阈值的连续闭眼为正；明确睁眼且无哈欠、处于 drowsiness 标注覆盖内为代理负例。闭眼尚未达到阈值、开闭过渡、undefined 为未知。
- distraction：仅采用 distraction 子集中 gaze_on_road 的明确 road/nonroad 标签；gaze_zone 暂不强行折算危险分心，gaze 文件保留准备任务但不生成监督。
- 不拼接跨摄像头像素关系、不自动确认驾驶员身份、不把缺失手部视频当整个数据集失败。
- 时间映射按官方 DEx 源码：先把三个 frame_shift 减去最小 shift，得到各摄像头相对 mosaic 起点的 offset；`annotation_frame = decoded_frame_index + offset`。25Hz 槽取最新过去帧，不使用名义 FPS。

## Task 1：纯标签与时间映射

Files: `training/dmd/labels.py`, `tests/test_dmd_labels.py`。

- [x] 先写缺失模块断言与功能测试并验证失败。
- [x] 提供 `compile_labels(document, view, frame_times_ms, *, closed_eye_min_ms, subset)`，返回 slots/targets/label_masks/summary；targets 顺序沿用 TASKS。参数为原始 OpenLABEL dict、视角、完整解码源帧相对 PTS 毫秒列表。
- [x] 检验 inclusive 区间、shift 正负、因果25Hz抽样、未知/冲突/越界掩码、持续闭眼阈值、未知眼状态中断、声明帧数与实际帧数差异、非法输入。审核后补充 signed-int64 与 100,000 槽容量检查，超限报错而不截断。
- [x] 运行 `python -m unittest tests.test_dmd_labels -v`，独立规格与质量审核。28 项标签测试通过；审核问题修复后复审通过。

## Task 2：可搬迁准备清单与配置

Files: `training/dmd/prepare.py`, `configs/dmd/split.subjects.json`, `configs/dmd/labels.v1.json`, `tests/test_dmd_prepare.py`。

- [x] 测试先行：同人跨子集隔离、未分配/重复被试拒绝、审计错误拒绝、缺失视角跳过、源文件修改拒绝、输出禁止覆盖及禁止位于 Dataset 内。
- [x] `python -S -m training.dmd.prepare --audit <audit.json> --dataset-root <Dataset> --output-dir <new directory>` 生成 `preparation.json`、`README.md`；不加载 ML、不解码、不生成假 manifest。
- [x] jobs 用相对路径、稳定安全 job ID、标注 SHA256、各摄像头 offset、默认 face/body 视角与同被试划分；gaze 作为暂不监督的来源记录。ROI 与闭眼阈值保持需要配置的显式状态。
- [x] 运行 `python -m unittest tests.test_dmd_prepare -v`，11 项准备清单测试通过，审核通过。

## Task 3：文档与交付核验

Files: `docs/training/DMD_PREPARATION.md`、文档索引、实验记录。

- [x] 说明第一批不等于全量特征提取或正式训练；给出服务器一个正式 CLI 命令，不再追加临时 test.sh。
- [x] 记录仍需用户决定的闭眼毫秒数、驾驶员 ROI/排他性与后续 manifest 接入。
- [x] 运行全套 unittest：199 项，198 通过、1 项平台链接测试跳过；`git diff --check` 通过。不触碰服务器环境，不推送、不合并、不自动提交用户未要求的改动。
