# Body 疲劳：恢复输入证据的第一阶段

当前诊断：body 疲劳 61 个正例窗口均漏检，全部未启用 F；54 个正例没有任何
有效面部/疲劳特征。增加轮数、降低分类阈值或取消有效性门槛不会补回观察证据。
这一步复用原始视频和受控感知对照，检验能否恢复眼部与疲劳时序特征。
它不运行分类器，因此不能把有效率当作疲劳召回率，也不宣称零检出已经修复。

## 运行

将代码同步到服务器，在仓库根目录执行：

```bash
bash scripts/server/recover_body_fatigue.sh
```

这是前台命令。后台执行：

```bash
mkdir -p runs
nohup bash scripts/server/recover_body_fatigue.sh \
  > runs/body-fatigue-recovery-001.log 2>&1 < /dev/null &
echo $!
tail -f runs/body-fatigue-recovery-001.log
```

默认使用 `.venv-perception-arm64/bin/python`，不使用 NPU 训练环境；不会安装依赖、
下载权重或重训模型。启动的是 **5 种感知配置 × 所有 body 验证录像**，顺序运行：
原配置、IMAGE 模式、检测阈值 0.3、存在阈值 0.3、跟踪阈值 0.3。
每个候选只改一个因素，不是 5 个训练任务。必须保留原 `Dataset`、特征缓存和感知工件。

可配置环境变量：`PYTHON_BIN`、`DIAGNOSTIC_SUMMARY`、`DATASET_ROOT`、`OUTPUT_DIR`、
`EXPERIMENT_CONFIG`、`WINDOW_STEPS`、`STRIDE_STEPS`。默认诊断报告来自
`runs/coverage25-composite-001/fatigue-coverage25/summary.json`，输出是
`runs/body-fatigue-recovery-001`，该输出目录必须不存在。默认窗口 100 槽/步幅 25
来自当前 coverage25/gap80 配置。更换模型时须同步实际窗口设置。

先小跑验证环境可使用：

```bash
OUTPUT_DIR=runs/body-fatigue-recovery-smoke-001 \
  bash scripts/server/recover_body_fatigue.sh --subset drowsiness --max-seconds 20
```

这只取各录像开头，可能不包含疲劳正例，不能代替完整对照。正式运行仍用默认 `all`：
不能只保留 drowsiness 正例，其他子集中的已知疲劳负例同样需要比较。
`--record-id` 可重复指定录像。已有人工确认的裁剪配置可通过 `EXPERIMENT_CONFIG`
单独试验，但必须限定对应录像；不能把之前分心录像的头部框套在疲劳录像上。

## 实现和结果

入口：`training.dmd.fatigue_recovery`。保留原视频解码时间轴、每源帧一次检测、
25 Hz 槽采样和各录像独立的跟踪/时序状态。只运行 FaceLandmarker；手部/物体在
疲劳重算包中明确为 missing，不伪造空检测。这六项特征不依赖手部或物体。

使用原 FeatureExtractor 重算 EAR 左右、MAR、PERCLOS、闭眼持续时间及眨眼率，
以原 group profile 和已保存 normalizer 支持列执行依赖/历史过滤。F 可用性只依赖
这些有效掩码和 group policy，无须加载模型权重或 normalizer 数值统计。
缺失帧不会插值成眼部观察；闭眼时虹膜缺失不等于 EAR 不可用；头姿矩阵失败也不
自动禁用有效眼部几何。

输出文件：

- `summary.json`：完整完成标记；每方案、每录像的原始/有效特征覆盖、窗口 F 可用
  数量和恢复/丢失数量、最长眼部缺失时间，以及按疲劳标签统计的失败原因。
- `fatigue_comparison.csv`：每方案 × 正例/负例/未知/全部的对照。
- `record-*/fatigue_frames.csv`：每槽原始数值、有效掩码、历史过滤后的掩码和失败原因。
- `record-*/fatigue_windows.csv`：窗口末端标签、F 可用性、各特征有效比例。
- `index.html` 和 JPEG：按疲劳标签及眼部失败原因分别抽样，并叠加关键点。
  下载整个目录查看，不能只下载 HTML。

共用回放也保留 `frames.csv`、`head_windows.csv` 等头姿附件。`head_windows.csv`
及 `head_continuity` **仍使用分心标签**，不要把它们当作疲劳结果；疲劳看专用文件。
原始帧重复映射到槽时，会按槽统计特征而不会重复调用感知模型。

完整、全部 body 验证录像的回放会将旧缓存窗口总数、正例/负例/未知数及 F 可用
数量与原疲劳诊断核对；不一致就失败，不发布成功 summary。旧诊断没有窗口几何
时，CLI 必须显式传 `--window-steps` 和 `--stride-steps`。新生成诊断会记录这两项
及 manifest 文件校验和。旧报告只能进行路径、特征定义、缓存完整性及窗口数量
核对，不能声称已验证旧报告未保存的 manifest 文件哈希。视频、工件和缓存仍使用
既有严格校验；输出与原始输入分离，原缓存不改写。

## 如何决定下一步

1. 检查 `variants.baseline.fatigue_evidence.baseline_fatigue_reproduced`。必须先复现
   原六项特征的掩码、共同有效位置的数值、源帧时钟和运行版本。false 时先排查
   环境/缓存差异。`status=complete` 只表示回放完成，不代表复现成功或性能提高。
2. 比较每方案 `fatigue_evidence.frames.known_positive.rates` 的眼部原始有效率，
   再看 PERCLOS 等有效率。原始恢复而有效仍为零，说明历史连续性仍不满足训练策略。
3. 比较 `fatigue_evidence.windows.known_positive` 的 `variant_f_eligible`、
   `f_recovered`、`f_lost`，分母始终保留全部原正例窗口。还要查看眼部/时序指标，
   不能把头姿带来的 F 可用当作疲劳证据恢复。对负例同样检查，但这些不是 FP/FPR。
4. 查看关键点是否可信：侧脸不可见的眼睛、遮挡和低置信度误检都可能产生错误数值。
   有效率提高不等于测量准确；报告不自动选赢家、不发布训练缓存。
5. 若可信证据恢复，再做独立版本的特征提取、疲劳训练和全任务回归；若这些方案
   仍不恢复眼部，则依据正例图像决定局部裁剪/更合适感知方案或独立视觉疲劳分支。
   不对已经没有输入信号的旧特征重复训练。

## 本地验证

```powershell
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p 'test_fatigue*.py' -q
.\.venv-training\Scripts\python.exe -m unittest tests.test_face_experiments tests.test_face_diagnostic_replay tests.test_face_crop tests.test_face_diagnostic_samples -q
.\.venv-perception\Scripts\python.exe -m unittest tests.test_face_diagnostic_samples -q
```

合成解码器/感知 API 测试验证工程正确性；真实视频上的恢复率和模型准确率仍需服务器测量。
