# 感知特征桥接工程验证

2026-09-18 · shadow · model 分支。继 TCN 增量后实现 P1/P3 的模型无关桥接部分；不是 YOLO/MediaPipe 真实运行或 DMD 跑分。

## 已实现范围

- 严格 driveguard-experts-v1 packet，原图像素、专家状态/年龄、明确驾驶员归属、ROI、确定性候选与同目标关联。
- EAR/MAR、候选头姿接收、PERCLOS、闭眼/眨眼、手物距离/IoU/持续邻近，输出19个值及19个mask；独立候选配置不覆盖24通道基线。
- 40ms连续槽、100槽统计、录制/视角/图像/ROI/模型版本隔离；不插值未来、不把缺失当负例。
- JSONL流式导出四个.npy，输出新目录保护、哈希和成功索引；targets/label_masks全部未知。
- 四录制合成特征通过既有 FeatureCorpus/normalizer→`[1,100,38]`→TCN `[1,4]`；没有可用标签时训练监督检查拒绝。

操作与完整语义见[桥接指南](../training/PERCEPTION_BRIDGE.md)。

## 验证环境与命令

沿用已有忽略环境：Windows / Python3.12.14 / torch2.14.0+cpu / NumPy2.3.5。未安装依赖、下载模型或调用摄像头。Git HEAD仍为`fe3c5b301eeff47b2391b33c86d04fd3aa7efbea`，dirty=true；未提交实现不能单凭该提交号复现。

```powershell
.venv-training/Scripts/python.exe -B -m unittest discover -s tests -v
.venv-training/Scripts/python.exe -S -B -m training.temporal.gpu_train --config configs/training/temporal_interaction.json --check-config
.venv-training/Scripts/python.exe -S -B -m training.temporal.npu_train --config configs/training/temporal_interaction.json --check-config
```

最终全量 **62项测试通过，11.877秒**：新测试包括几何16项、时序10项、导出7项，原训练工程29项保留。独立规格与质量复核通过。双平台配置预检返回config_valid/device_verified=false，不代表加速器已验证。耗时是本机测试运行记录，不是模型性能基准。

独立审查期间补充失败回归并修复：

1. 极小几何分母导致float32溢出：1e−6像素退化阈值、整数元数据上界、有限float32校验；时序状态先在小型副本更新，成功后再提交。
2. 眨眼起点证据在窗口外仍被计数：完整事件包含此前确认睁眼槽，越过窗口后移除。
3. 导出清理失败却已发布成功索引：清理完成后才进行最终索引替换；故障注入测试覆盖临时文件与marker删除失败。
4. 导出元数据引用调用方可变ROI：保存独立快照；生成器后续修改输入不改变工件的ROI记录。

导出测试跨越4096行分块边界（4101槽），并检查数组dtype/shape、哈希、未知标签和输出目录保护。临时测试数据放在忽略的runs下并随测试清理；没有产生需发布的模型checkpoint。

## 未验证与下一步

GPU/NPU实机、服务器torch2.7.1、真实YOLO/MediaPipe输出、DMD标注映射、精度/召回/事件延迟/FPS/真实内存峰值均未验证。几何阈值只是工程候选，尤其ROI与外部driver_confirmed不等于已实现自动驾驶员归属。

下一步选定上游模型工件、许可与调用配置，完成真实推理适配、坐标恢复、归属/跟踪/重采样；数据任务继续暂停。完整计划见[实施计划](../training/IMPLEMENTATION_PLAN.md)。API/数据库公共契约本轮未变，文档影响检查后无需改字段。
