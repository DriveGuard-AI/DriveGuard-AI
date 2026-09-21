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
