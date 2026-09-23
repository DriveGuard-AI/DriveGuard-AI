# Body 分心：感知对照与独立头姿补充分支

后续 H 使用条件实验和覆盖审计见 [HEAD_POLICY_EXPERIMENTS.md](HEAD_POLICY_EXPERIMENTS.md)。
单视频裁剪预览、绑定配置和连续性对照见 [FACE_CROP_EXPERIMENTS.md](FACE_CROP_EXPERIMENTS.md)。

本轮新增可运行的对照工具和模型，尚未在真实数据上证明精度提升。已确认的输入问题：body 76.60%采样槽没有人脸结果；47个分心正例窗口当前头姿有效且头姿覆盖>=50%，但F组不合格并漏检。按槽统计的感知覆盖与按窗口统计的任务召回不能直接等同。

## 1. 独立头姿补充分支

新配置`configs/training/relation_gaze_head.json`选择`relation_gaze_head_v1`。仍读取同一套23维特征缓存、同一manifest和训练主体划分。原四组F/O/HO/FO及gaze资格保持原定义，新H分支只使用三个头姿值和掩码，具有独立投影及因果TCN，只加入分心任务的五路门控。F仍包含头姿，因此这是补充分支实验，不是已经消除全部特征重叠的五组重分区。

H要求末槽三个轴都有效，窗口内三轴同时有效比例>=0.5。过期和训练集中不支持的字段沿用原预处理清除。H没有资格时权重为0；只有H可用时，只给distraction提供监督和概率，其他三个行为及gaze不因H而自动可用。不会把缺失值填成可靠证据。

在服务器`/mnt/workspace/DriveGuard-AI`仓库根目录，使用原NPU Python从头运行20轮，便于与原20轮实验比较：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze_head.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --epochs 20 \
  --output-dir runs/relation-gaze-head-npu-001
```

这是新模型，**不要传原模型的`--resume`**。新模型自己的`last.pt`可以按原规则精确续训到新目录；模型种类变化会被拒绝。这里只训练下游模型，不必重提取视频。新训练入口直接保存`best.pt`/`last.pt`在上述output-dir；旧实验的`model/`子目录不是本命令默认布局。

训练前可用同一命令加`--check-config`只检查配置。CPU/GPU改为`training.gaze.cpu_train`/`training.gaze.gpu_train`。NPU仍须使用原已验证环境，不用感知环境代替。

## 2. 比较模型效果与覆盖

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.diagnostics \
  --checkpoint runs/relation-gaze-head-npu-001/best.pt \
  --backend npu --view all \
  --output-dir runs/body-dist-head-diagnostics-001
```

工具沿用新checkpoint的原缓存路径和normalizer，不重新拟合。重点查看：

- `groups.view.body`与旧报告的召回、F1、误报率；同时检查face和其他行为是否退化。不能只看整体准确率。
- `windows.csv`新增`h_eligible`、`h_valid_ratio`、`h_dist_weight`和`distraction_head_evidence_available`；`f_eligible=False,h_eligible=True`说明新路径具备输入，是否正确检出仍由真实预测确认。
- 新模型分心五路权重之和为1：F/O/HO/FO四列加H列。原四列之和可能小于1。
- `groups.h_eligibility`保留头姿充足和不足两组的结果，不删除头姿缺失窗口。`known_input_coverage`与`detected_fraction_of_all_known_positives`用于检查覆盖率变化，不能只对有输入子集报高分。
- `history.json`新增按任务的`task_unavailable_windows`，新模型另有`head_evidence_windows`；`data_summary.json`保留原始正负标签数和可监督数量。

预测接口`JointPredictor`在新模型上额外返回`behavior_available`（四任务分别判断）、`distraction_head_evidence_available`及`distraction_evidence_status`。`head_supported`表示满足头姿质量门槛，**不表示预测正确或司机安全**；`limited_head_evidence`表示其他组仍有输入但头姿不足，此时仍返回概率用于完整评估；`unavailable`才返回空概率。gaze仍要求F和虹膜资格。

## 3. 感知提取对照

独立入口`training.dmd.face_experiments`只读原视频和已发布缓存，不启动YOLO/NPU服务，不发布训练特征。每个实验使用独立MediaPipe状态，从录制开头按原因果采样顺序处理。具体实验集合写在`configs/perception/face_experiments.json`，基线和各候选的实际参数都会进入报告。

默认五个方案：原VIDEO基线、IMAGE逐帧、仅detection改为0.3、仅presence改为0.3、仅tracking改为0.3。0.3只是本次明确指定的对照值，不是已验证的推荐生产阈值。每个候选只改变一个变量。

先对之前截图对应的第二段body视频前60秒做对照（会明确标记partial）：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.face_experiments \
  --experiment-config configs/perception/face_experiments.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --dataset-root Dataset \
  --record-id 3dcb48957dffd592fe200721 \
  --max-seconds 60 \
  --output-dir runs/body-face-experiments-001
```

目录必须全新。去掉`--max-seconds 60`可完整重放该段；再去掉`--record-id`可覆盖val/body/distraction全部录制。每个方案都重新解码并推理，耗时大致随方案数量增加，不会把完整视频载入内存。`--samples-per-reason 0`可禁用截图。

查看`summary.json`和`index.html`。`variants`按方案汇总所有选中录制，`records`保留逐段结果。先核对baseline的`baseline_runtime_and_masks_match`，确认仍能复现原缓存。随后比较各候选`coverage`中all、known_positive、known_negative的头姿／虹膜有效率，再检查`angle_jumps`及CSV角度、候选截图。连续相邻推理帧之间的角度差只是稳定性指标，不是角度误差或准确率；低变化也可能是错误估计。`timing`只作为当前CPU环境运行参考，测量的是Face推理加转换／诊断，不含Hand或YOLO服务。

每段每方案的`frames.csv`保留原缓存和候选的头姿／虹膜掩码、角度、采样时间和分心标签；`candidates.jsonl`保留具体拒绝原因及候选位置。`summary.json`只在模型、视频解码器和文件全部成功关闭后发布。

默认不猜司机头部裁剪框。若人工确认了某段完整头部活动范围，可复制实验JSON到新文件，向variants追加一个仅修改`crop_xyxy`的方案：四个数分别为原图归一化的左、上、右、下边界。其他字段保持基线值，并换一个唯一name。单段裁剪不应未经检查应用到所有录制。关键点会映回原像素后检查ROI和眼部几何；矩阵得到的角度标记`crop-experimental`，未经坐标与标定验证不能当作原相机头姿，也不能直接作为正式训练缓存。

感知对照和新H分支训练可独立运行。先用原缓存验证H分支，避免同时改变提取方式导致无法判断收益来自哪一项。候选感知方式只有在人工检查关键点、头姿和时序稳定性后，才另行决定是否生成新缓存。有效率提升不等同于准确率提升。

## 4. 实验边界

本次没有用两张截图直接决定裁剪框、阈值或新模型，没有降低正式缓存的几何有效性标准，没有恢复缺失标签，也不保证47个窗口全部恢复。当前val已参与模型选择和原因诊断，后续对照结果属于验证集开发指标；最终泛化需要未参与调整的数据评估。
