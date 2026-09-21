# DMD 批量提取、标签对齐与 NPU 训练

本文是旧版四任务/19特征流程。包含gaze的正式全量训练使用[新版联合训练指南](GAZE_FULL_RUNBOOK.md)，无需先重跑本文pilot或十二任务补充；旧缓存不能补零冒充新版23特征。

当前代码已接通准备清单到训练输入的流程。已有环境和三个工件无需重新安装，也无需再执行临时 test.sh。下面命令在服务器 `/mnt/workspace/DriveGuard-AI` 仓库根目录执行；先同步本轮源码、配置和测试。

## 1. 从已有清单生成小批次配置

```bash
python3 -S -m training.dmd.batch init \
  --preparation runs/dmd-preparation-001/preparation.json \
  --output runs/dmd-pilot.local.json
```

此命令仅使用标准库。默认按 train/val × distraction/drowsiness × face/body 各选第一段，当前资料通常得到 8 个任务；gaze 未映射监督，不进入提取。如果 preparation 已显式指定闭眼时长，原值保留；旧清单为 null 时，填入用户确认的首轮 `closed_eye_min_ms=1000`。新批量配置可显式修改该值，不能把 null 用于训练。

**pilot 不是保证正负样本齐全的训练集。** 首段录制可能没有手机使用等行为。它用于测提取速度、面部/手部可用性和标签覆盖；若 publish 显示某任务缺少正负窗口，应增加对应录制，不把未知改负例或自动关闭任务。

### 必须确认驾驶员 ROI

打开 `runs/dmd-pilot.local.json`，每个 job 都列出了原视频路径及以下字段：

```json
{
  "normalized_roi": [0, 0, 1, 1],
  "confirmed_exclusive": false
}
```

这里 `[0,0,1,1]` 只是**完整画面的候选范围**，不是已确认区域。实际原视频宽高由程序读取，ROI 按 `[左/W, 上/H, 右/W, 下/H]` 换算回原图坐标；不要直接填截图中的像素坐标。

逐任务查看视频，确认该区域内不会纳入乘客的脸或手后，才把 `confirmed_exclusive` 改为 `true`。如果有乘客，先缩小 ROI 并检查驾驶员活动时仍在范围内；同机位可以复用经过检查的范围，但不同机位不得自动照搬。两张样例截图不能替全部任务确认归属。没有任何自动批量设 true 的命令；未确认的任务会在加载模型前给出具体 job ID。

full-frame 候选范围只影响空间归属筛选，当前运行时仍对完整原图推理，并非裁剪加速选项。face/body 分别生成时序记录，不直接拼接跨相机坐标。

## 2. 在感知环境提取与生成标签

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.batch run \
  --config runs/dmd-pilot.local.json \
  --dataset-root Dataset \
  --output-dir runs/dmd-features-001
```

不要对这个命令使用 `-S`，因为感知阶段需要虚拟环境中的 NumPy、PyAV、PyTorch CPU、YOLOX、MediaPipe 等现有包。这里清除的是**本次进程**的 CANN Python 注入，不改变之后 NPU 训练使用的环境。

默认加载 `configs/perception/dmd_cpu.json`：三个已核实工件的路径/hash/来源已填写；不使用旧 `runtime.local.json` 中用于空白帧测试的 640×480 ROI。文件 hash 不匹配会停止，不自动下载、替换或升级依赖。YOLOX 权重再发布许可仍标记 pending_permission，本功能不发布上游权重。

运行会显示当前任务及每 100 个源帧的进度。结果包含：

- 原始特征 `values/masks/expert_age_ms` 和原始全未知标签数组，保持不变。
- 根据实际源帧 PTS 和相机 offset 对齐的 `targets.npy`、`label_masks.npy`，形状 `[N,4]`；未知仍为 mask 0。
- `source_pts_ms.npy`：完整解码源帧的时间，数组下标就是源帧序号。
- `complete.json`：特征报告、标签正负/未知数量、实际/声明帧数诊断及 hash。
- `reports/<ID>.json`：本次 completed/reused/failed 及每个任务的结果。

完整结果位置为 `records/<job_id>/<attempt_id>/`，不会复制视频、解出全部图片或覆盖原始标注。输入视频做流式 SHA256 读取，会产生额外磁盘读取时间；CPU 感知不保证实时，先根据 pilot 估算全量耗时。

### 失败和恢复

普通单段解码/推理错误会记入 `failure.json`，继续其他任务；有失败时最终退出码为 2。Ctrl+C 停止进程，不伪装成普通失败后继续。

原样重跑同一 run 命令即可：仅复用来源、ROI、标签阈值、模型配置、特征规则、源码、依赖版本与数组校验全部一致的完整记录。不完整或损坏的 attempt 不复用；新 attempt 单独创建，旧失败文件保留。更改阈值/ROI/实现后会重新提取，首版不做跨规则的原始特征二级缓存。

输出根目录使用 `.batch.lock` 排除同时 run/publish；正常结束、Python 异常或 Ctrl+C 都释放锁。机器断电/强制 kill 可能留下锁，确认没有进程在使用此输出目录后才能手动移除该锁文件；程序不会自动抢锁。输出目录必须在 Dataset 外，也不能是 Dataset 的父目录。

## 3. 发布训练清单

同样使用感知环境，确认所有选中任务完成后执行：

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
  .venv-perception-arm64/bin/python -m training.dmd.batch publish \
  --config runs/dmd-pilot.local.json \
  --dataset-root Dataset \
  --output-dir runs/dmd-features-001
```

此阶段不再运行感知模型。它核对来源和缓存完整性，调用真实训练数据加载器，只用 train 拟合归一化，并统计窗口末帧标签与可用输入。每个激活任务在 train/val 都必须有有效正例和负例；缺少任何一个时，不生成 `manifest.json`，也不会悄悄丢掉任务。

失败报告中的 `raw_positive/raw_negative` 是原始已知标签窗口；`positive/negative` 是有可用输入的窗口。`unavailable_windows` 表示四组证据都不合格。当前资格检查采用现有模型的“任一证据组可用”规则，不等于每个任务都具备理想视觉证据；还应检查 face 视角对 fatigue 的可用性，不能把通过此检查当精度证明。

成功时 stdout 给出 `manifest` 的完整路径，同时生成 `coverage.json` 和 `report.json`，位于 `runs/dmd-features-001/manifests/<发布ID>/`。只有最后出现的 `manifest.json` 是成功标志；失败目录里的 `candidate.json` 不能用于训练。

## 4. 扩大到全部已定义监督任务

```bash
python3 -S -m training.dmd.batch init \
  --preparation runs/dmd-preparation-001/preparation.json \
  --selection all \
  --output runs/dmd-all.local.json
```

确认新增任务的 ROI，把 pilot 中已审查的对应 job 设置原样带到 all 配置。仍使用同一个 `runs/dmd-features-001` 输出目录，依次运行 run、publish，只需将 `--config` 改为 `runs/dmd-all.local.json`。匹配的 pilot 缓存自动复用。

用户当前 58 录制中，35 distraction + 12 drowsiness × 两个视角，对应 94 个已定义监督任务；11 gaze × 两个视角暂不进入监督提取。不是删除 gaze 文件，而是避免把未定义的 gaze_zone 强行当危险分心。

## 5. 切回原 NPU 环境训练

使用 publish 成功输出的 **manifest 完整路径**填写 `--manifest`。下面 `<发布ID>` 必须替换为实际输出中的目录名，不要原样执行带尖括号的路径。

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.relation.npu_train \
  --config configs/training/relation_gated.json \
  --manifest 'runs/dmd-features-001/manifests/<发布ID>/manifest.json' \
  --data-root runs/dmd-features-001 \
  --epochs 1 \
  --output-dir runs/relation-dmd-npu-pilot-001
```

这使用原 NPU Python 和原 CANN 环境，不使用感知虚拟环境中的 CPU PyTorch。输出目录必须全新；一轮确认后按训练计划设置轮数或用现有 last.pt 续训流程。真实首训不加载此前合成 checkpoint。

**验证边界：** 本机已验证的是自编标注/模拟感知输出 → 真正的标签转换/缓存/manifest → CPU 一轮训练。真实 DMD 视频的批量提取速度、覆盖、NPU 首训和模型精度，必须以服务器实际执行结果为准。
