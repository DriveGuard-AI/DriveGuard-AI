# 四行为 + Gaze：正式数据全量训练指南

2026-09-21 · shadow。新版入口是 `training.dmd.gaze_batch` 和 `training.gaze.npu_train`；旧入口保持原四任务行为。

**服务器当前推荐入口：** 完成下方第2节配置后，按[YOLO11s NPU＋MediaPipe后台全量提取](SERVER_EXTRACTION.md)执行。新方案提供nohup、6/3/3并行和自动发布；下方第3、4节保留原YOLOX CPU串行方案，不要两个流程同时执行。新方案输出目录不同，训练时使用其成功发布的manifest。

本轮提供新版代码与本机工程验证，不表示服务器全量提取、新版 NPU 或真实精度已经通过。无需继续旧八段/十二段试跑，也不需要重新下载视频或升级已验证依赖。

## 1. 训练内容

- 四个独立行为二分类：fatigue、phone_use、drinking、distraction。
- 一个九分类辅助任务：left_mirror、left、front、center_mirror、front_right、right_mirror、right、infotainment、steering_wheel。not_valid/冲突/缺失为未知，不是安全或第十类。
- 输入原19个特征加左右虹膜各两个局部位置，共23维，拼接同形有效性mask后为46通道；新版同时启用三个头姿估计字段。
- 保留四流门控与共享因果TCN；gaze头使用F流表示，其损失更新共享编码器。gaze真值不输入模型，预测区域也不直接触发危险告警。
- 感知专家冻结：推荐YOLO11s在NPU检测、MediaPipe在CPU离线提取，再由NPU训练融合时序模型；原YOLOX CPU入口保留。本轮不微调感知工件，不是全流程NPU视频推理。

设计见[完整训练方案](../superpowers/specs/2026-09-21-full-training-gaze-design.md)，实际检查及已知边界见[工程验证记录](../experiments/20260921-gaze-joint-training.md)。

## 2. 生成全量配置

同步本轮全部源码、配置、测试和文档后，在服务器 `/mnt/workspace/DriveGuard-AI` 根目录执行：

```bash
python3 -S -m training.dmd.gaze_batch init \
  --preparation runs/dmd-preparation-001/preparation.json \
  --roi-from runs/dmd-pilot.local.json \
  --output runs/dmd-full-gaze.local.json
```

只读原preparation并核对hash，不修改原清单。现有35个distraction、12个drowsiness各取face/body，11个gaze取face，共105个任务；最终以实际输出为准。gaze body原文件保留，不自动生成注视区域监督。

`--roi-from`只复制同一preparation且job/video完全匹配的已确认区域。若已保存十二任务配置，可改用该文件，或重复传入另一个`--roi-from`；冲突会报错。没有旧配置可省略该参数，所有区域从未确认开始。新文件已存在时拒绝覆盖人工检查结果。

检查剩余视频后，修改对应`normalized_roi`与`confirmed_exclusive`。全画面无其他人的脸/手时，可以保留`[0,0,1,1]`并确认；不能把已检查十二段的确认扩大到全部数据。

## 3. 新版全量提取

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.gaze_batch run \
  --config runs/dmd-full-gaze.local.json \
  --dataset-root Dataset \
  --output-dir runs/dmd-full-gaze-features-001
```

这是正式数据准备，不是新一轮临时测试。旧八段缓存缺少新版特征，不能补零升级；它们的视频随全量自然重新提取，不另排小批次。旧缓存和原视频不删除，新版用独立目录。

只保存录制级数组，不导出全部图片或复制重叠窗口。CPU感知可能较慢，日志打印源帧数和视频秒数，不能把视频秒数当处理耗时。建议在服务器已有持久终端会话中执行。

普通任务失败保留报告并继续；汇总`failed>0`时先处理失败。原样重跑复用来源/代码/依赖/配置一致且校验通过的新缓存，未完成的整段重新提取。长任务期间不要改代码、依赖、工件、标签或配置。Ctrl+C会退出；强制终止留下锁时，应确认没有运行进程后再处理，不自动抢锁。

## 4. 发布训练清单

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.gaze_batch publish \
  --config runs/dmd-full-gaze.local.json \
  --dataset-root Dataset \
  --output-dir runs/dmd-full-gaze-features-001 \
  --training-config configs/training/relation_gaze.json
```

不重新运行感知模型，但会读取校验来源及缓存。使用成功stdout中的`manifest`路径；失败目录的`candidate.json`不是训练输入。

报告包含行为raw/usable正负数、gaze九类raw/usable支持数、未知/不可用数，并按录制、被试、视角分组。四行为在train/val需要可用正负样本，gaze九类需要有效窗口。raw有而usable为零时应检查感知质量，不把未知改成负例、不移动被试、不自动关闭任务。

gaze输入资格：F组可用，末槽三个头姿字段及至少一只眼的两个虹膜字段有效，窗口内同时满足条件的比例至少0.5。闭眼、眼镜反光、侧脸可能导致无效；这只是工程质量规则，不是精度保证。

## 5. 原 NPU 环境正式训练

把下方`实际发布ID`替换为上一步成功输出的目录名：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze.json \
  --manifest runs/dmd-full-gaze-features-001/manifests/实际发布ID/manifest.json \
  --data-root runs/dmd-full-gaze-features-001 \
  --epochs 10 \
  --output-dir runs/relation-gaze-npu-full-001
```

即使shell显示感知虚拟环境，也使用上述原NPU Python；不要给训练加`env -u PYTHONPATH`，保留原CANN环境。GPU入口为`training.gaze.gpu_train`，CPU为`training.gaze.cpu_train`。设备不可用时报错，不偷偷回退CPU。

默认FP32、batch16、学习率0.001、Adam weight_decay0.0001、梯度裁剪1、seed42、num_workers0。10轮是首轮基线，不是收敛保证。各任务损失先按各自有效样本归一化，再对有监督任务等权组合，未知标签不贡献损失。

归一化只拟合train；保留八人train、两人val的跨子集统一划分。val参与选择best.pt，不称独立test。

### 续训

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze.json \
  --manifest runs/dmd-full-gaze-features-001/manifests/实际发布ID/manifest.json \
  --data-root runs/dmd-full-gaze-features-001 \
  --resume runs/relation-gaze-npu-full-001/last.pt \
  --epochs 20 \
  --output-dir runs/relation-gaze-npu-full-002
```

续训使用last.pt及新输出目录；仅运行时路径/设备和总轮数可变。数据、类别、特征、结构或任务权重变化不属于精确续训。旧四任务或合成checkpoint不能作为本轮真实五任务断点。

## 6. 输出、推理与边界

训练目录保存config.json、environment.json、data_summary.json、history.json、result.json、best.pt、last.pt及校验文件。history.json提供每轮损失和指标曲线数据：四行为指标，gaze混淆矩阵、逐类支持数/召回/F1、macro-F1、准确率，分被试/视角指标及不可用计数。

重载接口`training.gaze.predict.JointPredictor`接收新版特征窗口，不接收视频。返回行为概率及独立gaze概率/类别/可用性；gaze不可用时类别和概率为空，不把默认argmax作为预测。公共Web API未自动扩字段。

头姿是canonical-face相对相机估计，虹膜位置是局部图像线索，不是车辆标定视线。DMD固定区域采集顺序可能产生捷径，禁止绝对视频时间/标签/主体身份作为输入；数据集分数不代表自然驾驶泛化。

真实权重训练完成后按[权重发布约定](../../models/releases/README.md)公开，附新profile、类序、数据划分及许可。本次不包含尚未训练的真实权重；YOLOX上游权重再发布许可仍需单独确认。

## 7. Body 分心逐窗口诊断

训练效果差时，先在原缓存上诊断，不需要重新提取视频或启动训练。工具 `training.gaze.diagnostics` 只读取可信 checkpoint 和原 manifest/特征，复用训练数据加载器及 checkpoint 中的归一化参数，并核对完整数据指纹。不要移动验证主体、删掉失败样本或改写缓存来提高分数。

先同步新增的 `training/gaze/diagnostics.py` 到服务器同名目录。在服务器仓库根目录执行本次第15轮最佳权重诊断（需要 `best.pt` 及同目录 `best.pt.sha256.json`）：

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.diagnostics \
  --checkpoint runs/relation-gaze-npu-iQNl4LJ1-epochs20/model/best.pt \
  --backend npu \
  --view all \
  --output-dir runs/body-dist-diagnostics-epoch15-001
```

默认选择 val/distraction；这里用 `--view all` 同时导出 face/body，便于直接对照，`groups.view.body` 才是 body 指标。仅诊断 body 可改为 `--view body`（默认值）。CPU 推理改成 `--backend cpu`，CUDA 使用 `--backend cuda`。默认 batch_size=64，可通过 `--batch-size 16` 减少批量；跨平台数值不保证逐位一致。

默认复用 checkpoint 保存的 manifest、data_root。路径变化时通过 `--manifest` 和 `--data-root` 指向原始同一套缓存；内容必须与训练一致。全量哈希校验可能耗时，stderr 会先提示校验，再报告推理窗口进度。输出目录必须全新，失败后不覆盖原报告；只有 `summary.json` 的 `status=complete` 才代表本次报告完整。

输出：

- `windows.csv`：所有选中窗口，包含未知标签、输入不可用和预测正确的窗口，不只收集错误。UTF-8 BOM 编码，可导入表格软件。
- `summary.json`：权重轮次/哈希、数据指纹、统计口径、整体以及按视角/受试者/录制/F资格分组的指标、0.1–0.9固定阈值对照。阈值表针对选中的整个集合；需要 body 专用阈值分析时单独运行 `--view body`。它不自动选择或部署阈值。

CSV重点字段：

| 字段 | 含义 |
| --- | --- |
| recording_id、view、subject_id | 录制/视角/主体定位；manifest不存原视频路径，不猜测文件名 |
| window_start_s、window_end_s | 相对该视频首个解码帧PTS的秒数；end为最后采样槽时间，标签取该槽，不是整窗多数标签；不是原标注帧号 |
| target、label_known | 视线离路真值及标签是否已知；未知target留空 |
| probability、prediction、outcome | 分心概率、当前阈值预测和TP/FP/TN/FN；输入不可用时概率和预测留空，outcome为unavailable；未知标签优先标unknown |
| logit | 模型原始分数；判定使用与概率阈值等价的logit边界（0.5对应0），避免浮点概率四舍五入改变训练判定 |
| behavior_available | 与现有训练一致：任意特征组可用；不等价于存在足够视线证据 |
| raw_*_valid_ratio | 特征缓存中的原始mask有效比例；并非人工验证的几何精度 |
| effective_*_valid_ratio | 依赖/过期过滤及normalizer支持掩码后的比例，与训练输入一致 |
| *_head_pose_valid_ratio | 同一槽yaw/pitch/roll三者同时有效的比例 |
| *_iris_pair_valid_ratio | 同一槽至少一只眼x/y成对有效的比例 |
| f_eligible、f_valid_ratio、face_age_ms | F组资格、按训练支持特征归一的组有效比例、当前面部结果年龄 |
| f_block_reasons | F过滤条件，可同时出现no_supported_features、no_current_valid_feature、low_valid_ratio、stale_face；只描述条件，不直接认定错误根因 |
| f_dist_weight、o_dist_weight、ho_dist_weight、fo_dist_weight | 该窗口分心任务的四组门控权重；未通过资格的组权重为0，权重不是因果重要性 |
| head_yaw_deg_last、head_pitch_deg_last、head_roll_deg_last | 最后采样槽有效头姿的原始数值；无效留空，用于检查偏移/异常 |

解读顺序：

1. 查看 `groups.view.body` 和 `groups.view.face` 的召回率及F1，确认与训练history的distraction统计相符；跨设备允许小数值差异。
2. 在CSV筛选 `view=body, outcome=FN`。比较正负样本原始头姿/虹膜比例；原始就低时优先检查视频/提取；原始高而effective低时检查过期、依赖和训练支持字段。
3. 查 `effective_head_pose_valid_ratio` 高、`f_eligible=False` 的漏检窗口，结合 `f_block_reasons` 判断组过滤是否挡住仍有效的头姿。
4. 对F可用的漏检查看概率和头姿末槽值，按recording_id、view与秒数抽查原视频和标签对齐；不能仅凭低概率断定标签错误。
5. 再考虑阈值或训练策略。`precision/recall/f1` 沿用训练的有效监督口径；`known_input_coverage` 记录已知标签中有输入的比例，`detected_fraction_of_all_known_positives` 的分母还包含输入不可用的正例。禁止只报筛选后的高分而省略覆盖率。

此工具不改变可用性规则、不重新标注、不微调模型。验证集已参与best选择，且窗口可能重叠；这些统计不是独立测试集效果或事件级告警效果。本机合成测试仅证明导出和计算流程，真实body根因以服务器诊断结果为准。

## 8. Body 人脸／头姿缺失原因重放

第7节已经发现原始头姿大量缺失时，用 `training.dmd.face_diagnostics` 检查缓存写入前发生了什么。它读取原 `complete.json`、`input.json`、模型工件和视频哈希，按原ROI、MediaPipe参数和25Hz因果采样从视频开头连续重放。仅启动CPU上的Face/Hand任务，不需要YOLO服务或NPU训练环境；原缓存、标签和训练规则不变。

同步本次源码后，在服务器仓库根目录执行本次实际缓存的诊断：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.face_diagnostics \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --dataset-root Dataset \
  --view body \
  --output-dir runs/body-face-causes-001
```

`--dataset-root` 指向原Dataset根目录，视频相对路径从原job读取，不由recording_id猜测。Face/Hand工件仍须位于原配置路径且哈希一致；不要重新下载不同版本工件。默认val/distraction、完整重放所有匹配body录制；`--view all`可增加face对照。可重复指定 `--record-id` 选择manifest中的record_id。`--max-seconds 60`只检查每段前60秒，报告标记partial，不能代表整段的分布；默认0不限制。资源清理或推理失败时没有完整 `summary.json`，重跑使用新目录。

输出及阅读顺序：

1. **`summary.json`**：先看 `groups.view.body.runtime_and_masks_match`；false时先查每段的版本、模型ID、源PTS及掩码差异，不能把重放原因直接认定为原缓存根因。true只证明这些身份和有效性对照一致，不证明每个头姿数值或人工判断准确。
2. 查看 `groups.view.body.reason_counts` 和 `positive_slot_reason_counts`：分别统计所有槽、已知分心正例槽的头姿不可用原因。统计单位是25Hz槽，包含复用帧，**不是前一份windows.csv中的窗口或FN数**。`invalid_candidate_counts`按唯一推理帧中的非法候选脸计数。
3. **`index.html`**：在本地浏览器查看有限样例。黄框是ROI；绿框是ROI内候选脸；红框是ROI外；橙框是非法关键点。框来自关键点外接范围，不是检测器原始框。默认每段每原因最多2张、间隔至少1秒，样例不是随机统计样本。`--samples-per-reason 0`禁用图片。
4. **`record-*/frames.csv`**：逐槽时间、原因、原始／重放头姿及虹膜有效性、逐字段掩码是否一致、分心标签。头姿须三个轴同时有效；虹膜须至少一只眼x/y成对有效；mismatch则比较全部三个／四个字段，避免掩盖单眼差异。时间与第7节窗口时间同为首个解码PTS起点，可用record_id加时间关联。
5. **`record-*/candidates.jsonl`**：每个实际推理帧的候选脸数、关键点数量、越界原因、框、ROI外点数、选中索引；复用同一源帧时不重复推理。ROI和原视频路径在summary的对应record中。

| reason | 含义和下一步 |
| --- | --- |
| no_face_detected | MediaPipe未返回人脸关键点；结合截图检查脸太小、侧脸、遮挡、反光。该API不能进一步区分内部检测与跟踪／presence失败，也不代表画面中没有人脸 |
| invalid_landmarks | 至少一张候选脸关键点数量异常、非有限或超出图像；原适配器会拒绝整组人脸，具体候选和子原因在JSONL |
| outside_roi | 候选脸关键点无法全部落入原ROI；结合截图检查ROI边界与真实司机范围 |
| ambiguous_faces | ROI内有多张候选脸，原规则无法确认司机 |
| driver_unconfirmed | 原区域未明确标记为司机独占；须检查实际标注，不能自动改成确认 |
| pose_matrices_missing / pose_matrix_count_mismatch | 变换矩阵不存在，或矩阵数量与人脸数不一致 |
| pose_matrix_invalid | 已选中脸，但矩阵没有通过现有头姿几何校验 |
| stale_face | 复用帧的结果超过原profile规定的有效年龄 |
| ok | 头姿可用；虹膜是否有效另看iris_reason，不等于F组通过 |

`iris_reason=geometry_rejected`可由闭眼、极端眼形、虹膜位置等几何条件触发，不能单凭该字段认定某一种原因。此次定位感知缺失；“头姿已有效但F仍被眼部时序特征挡住”的问题仍需独立评估F组拆分，不靠增加训练轮数解决。
