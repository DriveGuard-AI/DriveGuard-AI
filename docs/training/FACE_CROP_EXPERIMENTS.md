# 单视频头部裁剪感知对照

目标：在同一视频、同一模型和阈值下，仅比较全图与人工确认的头部区域。原始特征缓存只读；输出只供诊断，不能直接用于重新训练。裁剪后模型矩阵得到的角度仍在实验裁剪坐标中，不能当作已经标定的原相机头姿。

## 1. 在服务器生成整段抽样预览

先同步本次代码，在 `/mnt/workspace/DriveGuard-AI` 运行：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.face_crop_preview \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --dataset-root Dataset \
  --record-id 3dcb48957dffd592fe200721 \
  --samples 12 \
  --output-dir runs/body-face-crop-preview-001
```

本命令只解码这段视频，不启动 MediaPipe、YOLO 或 NPU 推理。验证缓存和原视频后，按整段均匀位置输出 12 张 JPEG、`index.html` 与预览 `summary.json`。原视频解码仍会顺序遍历整段；短视频的实际抽样数可能少于请求数。输出目录必须是新目录，且位于数据和缓存目录之外。

将 **整个 `runs/body-face-crop-preview-001` 文件夹**下载到本机，双击其中 `index.html`。不要只下载 HTML，否则浏览器找不到图片。

## 2. 确认范围并下载配置

1. 在大图拖出包围完整头部活动范围的矩形，或填入 0～1 的左、上、右、下边界。工具没有预设裁剪框。
2. 查看所有抽样画面上的同步黄色框，点击缩略图可放大。留出头部转动余量，不要仅贴着一帧的人脸边界。
3. 勾选检查确认后，点击“下载此视频的对照配置”，得到 `face_crop_experiment.json`。
4. 将下载文件上传到服务器，例如 `configs/perception/face_crop_experiment.json`。

JSON 仅包含 baseline 与 head_crop 两方案，并绑定 record_id、原视频 SHA256 和图像尺寸。执行时不匹配会拒绝。抽样不能保证覆盖每个动作，后续需要检查完整对照；更换录像需重新生成预览和配置。

可先只检查 JSON 格式，无须推理：

```bash
.venv-perception-arm64/bin/python -S -m training.dmd.face_experiments \
  --experiment-config configs/perception/face_crop_experiment.json --check-config
```

`--check-config` 不验证实际视频；视频绑定在正式对照前校验。

## 3. 先做前 60 秒对照

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.face_experiments \
  --experiment-config configs/perception/face_crop_experiment.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/f356b9789f604cf5850d74fefc61261e/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --dataset-root Dataset \
  --record-id 3dcb48957dffd592fe200721 \
  --max-seconds 60 \
  --samples-per-reason 4 \
  --output-dir runs/body-face-crop-experiments-001
```

裁剪实验只允许选择一段录像。全图 baseline 使用原阈值和模式，head_crop 只改变输入裁剪范围；各方案使用独立模型状态。预览导出坐标回映到原图，仍执行原 ROI、关键点几何有效性和多脸拒绝规则。

## 4. 看哪些结果

先核对 `variants.baseline.baseline_runtime_and_masks_match`；不能复现旧缓存时，先排查环境差异。

- `coverage`：所有槽、正例、负例的头姿和虹膜有效率。检出率提升不等于角度准确。
- `head_continuity.cached/variant`：缺失槽数、缺失段数、最长连续缺失时长和缺失段长度分布。跨录制不连接缺失段，短片末端尚未结束的缺失段只计已观察长度。
- `head_continuity.windows.coverage25.known_positive`：在 100 槽窗口、25 槽步幅下的可用窗口、gained 和 lost。同样检查负例；新增正常窗口可能引入误报。
- `head_windows.csv`：逐窗口末槽标签、覆盖和间隔，以及 strict/coverage25/gap80 三规则的 cached/variant 代理资格。小于 4 秒的片段没有完整窗口，文件仍有表头。
- `index.html`：查看样例关键点是否贴合眼、鼻、嘴和脸部。蓝框表示推理裁剪范围，青色点表示映射回原图的有效关键点。样例数量有上限，不代表所有画面。
- `angle_jumps`：相邻有效推理帧的角度变化，遇缺失重置。平滑不等于准确，crop 与全图矩阵角度的绝对值不能直接当成角度误差比较。

窗口代理统计基于原始且新鲜的头姿掩码，没有执行 normalizer 支持过滤、分类模型或坐标标定；不等于训练 H 分支实际可用性，更不等于分心召回。不能直接以这些统计宣布新方案胜出。

短片有收益且关键点质量正常后，使用同一配置去掉 `--max-seconds 60`，将输出目录换成 `runs/body-face-crop-experiments-full-001`，再覆盖整段。先取得完整对照和质量证据，再决定是否开发坐标标定或其他感知模型对照。
