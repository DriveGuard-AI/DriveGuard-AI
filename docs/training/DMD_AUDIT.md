# DMD 全量标注审计

2026-09-21 · shadow。此工具是正式数据准备入口，不是环境测试或模型训练。

## 服务器运行

先将本轮新增的 `training/dmd/`、测试和文档同步到服务器。在仓库根目录执行：

```bash
cd /mnt/workspace/DriveGuard-AI
/opt/buildtools/Python-3.11.4/bin/python3 -S -m training.dmd.audit \
  --dataset-root /mnt/workspace/DriveGuard-AI/Dataset \
  --output-dir /mnt/workspace/DriveGuard-AI/runs/dmd-audit-001
```

仅使用 Python 标准库；不需要 pip install，不加载感知虚拟环境或 NPU。`-S` 不加载 site-packages。运行中逐个显示标注进度，末尾 stdout 输出 JSON 摘要。

输出目录必须是新目录，且不能位于 Dataset 内。再次执行请换成 `dmd-audit-002` 等新目录；不会覆盖旧报告。

产物：

- `audit.json`：被试、录制、文件配对、每种原始标签的覆盖、流同步声明、标注 SHA-256 和详细诊断。
- `audit.md`：可阅读的标签覆盖和问题摘要。

请将这两个报告发回用于确定标签映射与被试划分，无需上传视频或全部 JSON。

## 如何看结果

- `status=ok` / 退出码 0：完成扫描且未发现解析/结构错误；可能仍有警告。
- `status=errors` / 退出码 2：扫描发现错误。通常仍会生成两份报告；先看报告，不要重装环境。
- 输入根目录/输出目录不可用等启动错误：退出码 2，stderr 说明原因，可能没有报告。
- `training_ready=false`、`video_decoded=false` 是本工具的固定边界，不表示 NPU 验证失败。

| 诊断 | 含义与处理 |
| --- | --- |
| MISSING_VIDEO / MISSING_ANNOTATION | 缺少视角或标注，先记录；drowsiness 缺 hands 不自动阻止扫描 |
| FRAME_COUNT_DECLARATION | 零偏移流声明帧数与标注末帧不一致，后续解码核对，不自动补帧 |
| FRAME_ACTION_MISMATCH / FRAME_TIMELINE_MISMATCH | 区间与逐帧引用存在差异，报告保留两者计数，不擅自选择训练真值 |
| UNREVIEWED_LABEL_TYPE | 四份样例中没有见过的类型；仍统计原始覆盖，不表示官方标签非法 |
| STREAM_DECLARATION_MISMATCH | 同录制的多份标注声明了不同同步信息，转换前需要核实 |
| ANNOTATION_READ_ERROR / INVALID_INTERVALS | JSON 或区间异常；其他文件继续扫描 |
| SUBJECT_MISMATCH / SUBJECT_GROUP_MISMATCH | 标注、目录或分组的主体身份不一致，划分前需要解决 |
| UNSAFE_URI / STREAM_RECORDING_MISMATCH / STREAM_VIEW_MISMATCH | 路径不安全、引用错录制或错摄像头；不会访问外部 URI |
| SYMLINK_SKIPPED / UNRECOGNIZED_LAYOUT | 跳过链接或未识别的布局；需要看具体路径，不能假定已覆盖该部分 |

工具按完整录制前缀配对，允许同 session 多段视频。gaze/hands 标签文件归入同一录制，来源分别统计。每种标签的覆盖帧数是该标注中闭区间并集，不重复计算重叠帧；不同标签轴的帧数不能相加理解为录制总长。同步偏移只报告原值，不猜测加减方向或使用固定 FPS 换算。

## 数据与隐私边界

不读取视频内容、不解压、不复制视频、不修改原始标注；不输出驾驶员年龄、性别等无关属性。报告包含本机根路径、录制名、主体编号和原始标签，请按数据集授权范围分享。所有链接文件均跳过；不会从标注 URI 下载内容。

`runs/` 继续由 Git 忽略；审计源码、测试、规则与文档进入版本管理。不要将原始大 JSON 提交到仓库。

## 下一步不是直接开训

审计结果用于确认四任务的正/负/未知语义与时间对齐。`not_valid`、`unclassified` 不自动当负例，手机出现不等于使用手机，闭眼或哈欠不直接等同真实疲劳。样例词表只用于发现新类型，不构成训练映射。

用户选择两份数据：目标 8 人训练、2 人留出评估；同一人跨子集和视角保持同组。当前训练器用 `val` 选 `best.pt`，因此留出集内部名为 val，不另划 test，也不声称独立最终测试。待全量覆盖确认后固定名单，不按模型跑分挑人。

此工具没有生成五类特征数组、真实标签或 v2 manifest。标签转换、批量感知提取和正式训练配置是后续工作；入口见 [关系门控训练指南](RELATION_RUNBOOK.md)。

## 验证来源

四份用户提供的真实标注已在本机用解析器读取：distraction 6748 帧/15 类，drowsiness 5481 帧/7 类，gaze 5340 帧/11 类，hands 5340 帧/8 类。这里的“类”是原始 action type，不是项目的四个输出头。drowsiness 的一帧声明差异被保留为警告；没有读取配套视频。

2026-09-21 用户回传的 NPU 合成训练已完成 10 轮、30 次更新，并通过续训和权重重载推理。它证明该短程训练链路，不是本机实测、长程稳定性或 DMD 精度。现有 NPU 环境不因本工具发生任何变更。
