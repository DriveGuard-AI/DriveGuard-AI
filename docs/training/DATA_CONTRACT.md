# 录制级特征数据契约 v1

新增联合训练v3：显式23维输入及独立gaze分类数组，见本文末尾第7节。v1/v2不改变含义。

2026-09-18 · 实现：training/common/data.py、normalization.py。只定义提取后的特征接口，不代表 DMD 原始标注已扫描、转换或具备完整四任务覆盖。

## 1. 文件布局与输入

data-root 可位于仓库外。清单为 UTF-8 JSON，schema_version=`driveguard-feature-manifest-v1`；不允许重复键或 NaN/Infinity。每条录制分别引用四个 `.npy`（非 npz、非 pickle）：

| 数组键 | dtype/shape | 含义 |
| --- | --- | --- |
| values | float32 `[N,F]` | 原始观测值，尚未归一化 |
| masks | bool/uint8/float32 `[N,F]`，仅 0/1 | 每槽每特征是否有效 |
| targets | float32 `[N,4]`，仅 0/1 | 同一槽时刻四类状态 |
| label_masks | bool/uint8/float32 `[N,4]`，仅 0/1 | 标签是否已知，与输入有效性独立 |

所有数组必须有限；无效观测建议填 0，最终归一化会强制置零。未知标签必须 target=0、label_mask=0。四任务顺序固定 fatigue/phone_use/drinking/distraction。N 至少覆盖一个窗口；不预先复制重叠窗口。

## 2. 清单字段

| 顶层字段 | 要求 |
| --- | --- |
| schema_version | driveguard-feature-manifest-v1 |
| dataset_version | 非空来源版本，不以分片数量代替版本 |
| feature_schema | version、names（有序 F 个名称）、sample_rate_hz；与公共配置逐项一致 |
| extractor_profile | name、feature_definitions（必须覆盖所有 names）；记录模型/几何定义等来源；synthetic=true 仅用于合成数据 |
| label_mapping | version=driveguard-behavior-v1、tasks 固定四类顺序、非空 rules，说明原始标签到正/负/未知的映射 |
| records | 非空录制清单，字段见下表 |

真实 extractor_profile 应在 feature_definitions/附加元数据中记录单位、公式/关键点索引、模型版本、坐标系/旋转、闭眼判据、物体类别映射、采样和最大观测年龄。代码能检查结构与哈希，不能证明这些文字对应正确提取过程；正式训练前仍需核对提取器和数据来源。

| record 字段 | 要求 |
| --- | --- |
| record_id | 唯一非空标识 |
| subject_id | 全局驾驶员标识，跨子集/分片统一 |
| recording_id | 全局录制标识，同一录制跨视角必须同主体同 split |
| view | 单个视角标识，不把多个摄像头拼成一条特征录制 |
| subset | Gaze/Drows/Dist 等来源；不得成为输入特征或直接充当标签 |
| split | train / val / test |
| values / masks / targets / label_masks | 相对 data-root 的 `.npy` 路径，使用 `/` |

拒绝绝对路径、盘符、反斜线、`..` 和符号链接逃逸。recording_id+view 不重复，values 文件不重复引用；同主体不能跨 split。代码无法从视觉内容识别被错误重命名的主体或复制的视频，清单生成阶段仍要保证标识真实。

## 3. 时间与标签

数组每行代表从同一录制起点开始的均匀采样槽；掉帧保留时间槽并用 mask 表示，不能删除行后压缩时间。首版不从数组推断真实 FPS，不做未来插值；采样器负责遵守 profile。

默认 25 Hz、window_steps=100、stride_steps=25。窗口 `[start,start+100)` 的 target/label_mask 来自 `start+99`，不是窗口内任意一帧出现即算正例。尾部不足一个步长的残余槽不另造短窗口。同一窗口不跨录制/摄像头。

每个启用任务在 train 与 val 的实际末槽窗口标签上必须同时有正负例。子集缺失标签记未知，不用 JSON 中“未出现”推导负例。独立 test 数据可登记但当前训练循环不消费，也不参与模型选择。

## 4. 归一化与内存预算

按 train 录制的每个唯一槽统计有效值，不按重复滑动窗口统计；包含录制中没有成为标签末槽的观测。使用 float64 分块合并人口均值/方差，std 下限1e-6。val/test 复用；标准化裁剪[-5,5]，失效值最终填0，掩码不归一化。

训练从未有效的通道 mean=0、std=1、supported=false；任何数据集使用该版本时该通道 mask 永远为0，不能在部署时擅自开启。输出按 `[F 个值, F 个有效性掩码]` 拼为 float32 `[T,2F]`。

数组使用只读 mmap，最多缓存四条录制；窗口按前缀计数定位，只复制当前 batch 所需窗口。运行前分块检查全数组并计算内容 SHA-256，需读取全部所选特征文件，但不读取原始视频、不占用第二份窗口磁盘空间。运行期间数据须保持不可变。

## 5. 追溯与变更

数据摘要含规范化 manifest、全部数组、extractor_profile、label_mapping 的 hash 和主体列表。跨机器根目录不同不改变内容摘要；相对路径、内容、标签、主体划分改变会导致续训拒绝。

已有[24通道基线 JSON](../../configs/training/temporal_baseline.json)及[38通道交互候选 JSON](../../configs/training/temporal_interaction.json)。后者使用 driveguard-interaction-v1，与[感知桥接](PERCEPTION_BRIDGE.md)的19个特征对应；不可沿用24通道权重或旧定义缓存。原始数据/派生特征/机器路径继续按 .gitignore 排除。

桥接导出的 record.json 仅描述单录制特征工件，不是本契约的完整 manifest。其 targets/label_masks 初始全部未知；还需真实标注映射、全局主体身份与切分，再生成完整清单。不能将几何信号直接当行为真值。

## 6. 关系门控扩展

2026-09-19新增[部署优先关系门控设计](../architecture/RELATION_GATED_FUSION.md)。第1–5节仍适用于v1；以下v2已在校验器、窗口预处理、训练与显式导出中实现。CPU验证及服务器命令见[关系门控指南](RELATION_RUNBOOK.md)。

- 保留19字段和四个数组的含义，另定义`driveguard-feature-manifest-v2`及配套record版本，记录`group_profile`与hash；模型类型/profile/normalizer/标签映射必须匹配。旧v1读写路径保持不变，新旧格式显式选择，拒绝静默补字段。
- 新增`expert_age_ms.npy`：float32 `[N,3]`，列顺序objects/face/hands；为采样槽时刻减该专家实际观测时间，不是程序处理耗时。有限且非负；以profile的`max_age_ms+1`截断，缺失/未知也用该值。负年龄/未来观测拒绝；数组及列定义必须加入内容hash。
- 年龄不能替代有效性mask。Objects成功空检测仍可有效；无脸/手或归属未知使相关mask失效。F/HO/FO所需专家年龄和逐特征依赖由group_profile确定。
- `group_profile`固定F/O/HO/FO顺序、组内特征名、字段依赖、年龄列、统计窗口、资格阈值和质量公式，训练/推理共用。只用训练supported字段计算有效比例；无supported字段的组始终不可用。
- 在采样窗口时计算四流的`valid_ratio/freshness`及资格，不落盘重复窗口；质量不按训练均值方差归一。值归一化仍沿用第4节，不把mask、来源子集名或label_mask当质量特征。
- 可选随机专家缺失增强尚未实现；未来增强必须先修改基础有效性/年龄，再传播到关系字段及历史统计，最后计算质量。没有原始状态可重算时保守屏蔽受影响统计到完整统计窗口恢复；训练增强不得保留完整输入派生的持续时长。
- 旧v1缺少年龄，不能用0填充伪造新鲜观测。只能从原始可追溯观测/视频重新提取新侧文件并记录来源，或继续用旧基线；不强制重新下载/解压全部数据，不覆盖旧缓存。
- 标签未知性、主体独立、外置根目录与只读mmap约束不变。NPU训练只消费这些特征及标签，不要求在训练环境导入感知包。

实现使用独立v2训练配置；manifest的group_profile及group_profile_sha256必须与配置相符。导出显式传--group-profile才产生v2 record；默认仍为v1。所有stat字段在父观测中断后保守屏蔽到该缺口离开100槽历史，窗口切片包含前99槽历史，避免切片边界重置。train/val的正负监督预检使用与训练相同的supported/quality/eligible规则，并另报raw标签与不可用窗口。测试场景见Spec REQ-011/012/013及[测试策略](../testing/TEST_STRATEGY.md)。

## 7. 联合 Gaze v3

`driveguard-feature-manifest-v3` / record-v3配套`driveguard-training-v3`和`driveguard-interaction-v2`，组profile为groups-v2。前19字段保留顺序，新增iris_left_x/iris_left_y/iris_right_x/iris_right_y，数值和mask均`[N,23]`；拼接为46通道。新版启用经过矩阵校验的头姿估计，完整定义由extractor_profile记录。

原四行为数组和age不变，另加`gaze_targets.npy` int64 `[N]`与`gaze_label_masks.npy` uint8 `[N]`。有效类别0..8，未知必须target0/mask0；类别0仅在mask1时代表left_mirror。清单新增`gaze_mapping={version,classes,rules}`，version为driveguard-gaze-v1，classes固定顺序见[全量指南](GAZE_FULL_RUNBOOK.md)。

Gaze只对face视角生成区域标签，其他子集/视角未知；四行为标签不从gaze区推断。来源/类别顺序/数组/规则与normalizer、checkpoint绑定校验；训练和重载拒绝旧19维缓存或更改的映射。原始导出仍无标签，批处理对齐后的监督另存，不覆盖原始数组。

九分类使用共享F时序表示与masked CE；必须有有效头姿和至少单眼虹膜信息，不能仅因Objects可用就计算gaze。行为raw/usable覆盖与gaze九类raw/usable覆盖分别报告。旧缓存缺少新增观测，需从原视频重新提取，不允许伪造填充升级。
