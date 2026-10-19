# Body 疲劳：拒绝头部区域外的假人脸

本轮处理的是截图中“手上的人脸关键点被算作有效眼部特征”的接受漏洞。
新增规则在特征计算前执行：全部人脸关键点必须处于该视频明确配置的头部区域，
同时仍须满足原驾驶员区域、坐标、候选唯一性等规则。拒绝后不输出人脸关键点或头姿，
不会插值补成有效眼部。手部/手机分支的区域不变。

这是可运行的感知对照实现，尚未替换生产感知配置、缓存或分类模型。三组顺序运行：

| 方案 | 输入与接受规则 | 要回答的问题 |
| --- | --- | --- |
| baseline | 原全图、原接受规则；额外审计头部区域 | 原来接受了多少区域外眼部数值？ |
| head_gate | 原全图，拒绝头部区域外候选 | 单独加空间约束会排除什么、保留什么？ |
| head_crop | 头部区域裁剪，映射回原图，再执行相同约束 | 裁剪后能否恢复更多区域内的眼部观察？ |

三组检测/存在/跟踪阈值均为 0.5。`head_crop` 相对 `head_gate` 只增加输入裁剪。
每视频独立创建模型状态。裁剪矩阵的头姿仍是实验坐标，不能直接发布为原相机头姿。
区域必须按 record_id、视频 SHA256、图像尺寸绑定；重复、遗漏、错配会失败。

## 1. 生成选框页面

同步代码后，在服务器仓库根目录执行。若上一轮报告目录不同，替换第一条路径。

```bash
.venv-perception-arm64/bin/python -m training.dmd.fatigue_head_regions prepare \
  --recovery-summary runs/body-fatigue-recovery-001/summary.json \
  --dataset-root Dataset \
  --output-dir runs/body-fatigue-head-preview-001
```

默认选择上一轮中**包含疲劳正例窗口且 baseline 复现成功**的 body 录像，并抽取每视频
24 张均匀样本；可用重复的 `--record-id` 显式选择其他记录。此步只解码，不运行模型。
根据此前报告默认应是两段疲劳录像，具体以命令返回的 records 数量为准。

将整个预览目录复制到本机，打开 `index.html`。各视频页面中选择覆盖低头、转头、
前倾的头部活动范围，留出余量，结合完整原视频检查。不要套用以前分心录像的裁剪框。
程序不会根据截图猜一个覆盖全视频的框，也不会将均匀抽样当作全程覆盖证明。

两个页面各导出一个 `face_crop_experiment.json`，分别改名保存为
`head-region-record-1.json`、`head-region-record-2.json`，传回服务器 `runs/`。
这些是手动选择的空间范围，不是新增的疲劳标签，也不证明框内检测都是真脸。

## 2. 合并成三组配置

```bash
.venv-perception-arm64/bin/python -m training.dmd.fatigue_head_regions build \
  --crop-config runs/head-region-record-1.json \
  --crop-config runs/head-region-record-2.json \
  --output runs/body-fatigue-head-regions.json
```

build 使用标准库，不加载感知模型。两段视频 × 三方案 = 六次顺序回放，**不是六次训练**。
不修改输入文件；配置输出和报告输出均要求新路径。

## 3. 后台运行对照

```bash
nohup bash scripts/server/compare_fatigue_head_regions.sh \
  runs/body-fatigue-head-regions.json \
  > runs/body-fatigue-head-regions-001.log 2>&1 &
echo $!
```

只有此处 `nohup ... &` 是后台运行。直接 `bash scripts/server/compare_fatigue_head_regions.sh ...`
是前台运行。默认使用原感知环境、CPU 推理，不下载模型或启动 NPU 训练。

`PYTHON_BIN`、`DIAGNOSTIC_SUMMARY`、`DATASET_ROOT`、`OUTPUT_DIR`、`WINDOW_STEPS`、
`STRIDE_STEPS` 的含义与 `recover_body_fatigue.sh` 相同；需要重跑时使用新的 OUTPUT_DIR。
v2 配置会让疲劳入口自动选中配置列出的录像。每段录像仍保留全部时间槽和全部窗口，
不按是否检测成功删样本。这个局部对照的总分母不等于全部 body 验证集分母。

## 结果判断

默认目录 `runs/body-fatigue-head-regions-001/`：

- `summary.json`：三组、每视频、按疲劳正例/负例/未知分组的完整结果。
- `fatigue_comparison.csv`：原眼部数值有效率、时序有效率、F 窗口恢复与损失，新增区域审计列。
- `index.html`：紫色框为头部区域，蓝色框为裁剪输入；红色含被空间规则排除的候选。
  原基线区域外但数值有效的眼部另设 `eye_pair_outside_head_region` 抽样类别。
- `record-*/candidates.jsonl`：逐源帧候选、原图坐标、区域内外及拒绝数量。
- `record-*/fatigue_frames.csv`、`fatigue_windows.csv`：逐槽数值、掩码和窗口明细。

新增 CSV 列均以 25 Hz 时间槽计数（并非唯一视频帧数）：

| 列 | 含义 |
| --- | --- |
| head_region_audited_slots | 存在配置区域、执行了区域审计的槽 |
| outside_head_region_candidate_slots | 驾驶员区域内存在头部区域外候选的槽 |
| head_region_rejected_slots | 应用区域规则并排除了至少一个候选的槽；可同时保留另一个候选 |
| eye_pair_outside_head_region | 最终接受的人脸在头部区域外，却算出了双眼数值的槽 |

先确认 baseline 复现；再检查 gate/crop 的最后一项应为零，并人工检查被拒绝的真脸
是否因为区域过紧而丢失。比较正例的连续眼部、PERCLOS 和 F 窗口恢复，也检查负例。
“区域外拒绝数”不能等同“假脸数”：空间约束可能拒绝越界真脸；手伸进头部区域仍可能
产生假脸。绿框/青色点和 `eye_pair_available` 仅表示通过数值规则，不表示已确认双眼准确。

如果裁剪恢复了可信眼部证据，再做独立版本的训练特征、疲劳训练及全任务评估；
如果仍不足，再考虑独立人脸定位/眼部质量验证或视觉疲劳分支。
此报告不产生分类 TP/FP，不能用来声称疲劳零检出已经修复。

## 工程验证

```powershell
.venv-training/Scripts/python.exe -m unittest tests.test_face_head_regions -q
.venv-training/Scripts/python.exe -m unittest discover -s tests -p 'test_fatigue*.py' -q
.venv-training/Scripts/python.exe -m unittest tests.test_face_experiments tests.test_face_diagnostic_replay tests.test_face_crop tests.test_gaze_perception -q
.venv-perception/Scripts/python.exe -m unittest tests.test_face_diagnostic_samples -q
```

外部解码器和感知模型用合成 API 测试工程链路；真实质量需在服务器原视频上验证。
