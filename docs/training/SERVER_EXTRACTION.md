# 服务器全量提取：YOLO11s NPU + MediaPipe CPU

2026-09-21 · shadow。以部署与兼容为目标，替换离线检测专家；不改变四行为＋九类 gaze 训练目标、23维特征及原被试划分。代码提供自动预热、后台提取、缓存恢复和发布；**尚未在用户 Ascend910B3 上运行新版检测器，不承诺实测速度或精度提升**。

## 配置决定

| 组件 | 首轮配置 | 执行位置 |
| --- | --- | --- |
| 目标检测 | YOLO11s、640、FP32、阈值0.25、NMS 0.45；保留 bottle/cup/cell phone 映射 | 单个 NPU 服务；CPU做后处理 |
| 面部 | 现有 MediaPipe 0.10.18/Face Landmarker，VIDEO模式，最多2脸，检测/存在/跟踪阈值0.5 | 每个视频独立CPU实例 |
| 手部 | 现有 Hand Landmarker，VIDEO模式，最多4手，同上阈值0.5 | 每个视频独立CPU实例 |
| Gaze | 开启面部变换矩阵、头姿及虹膜线索；关闭未使用的blendshapes | CPU特征计算 |
| 调度 | distraction最多6路、drowsiness最多3路、gaze最多3路；每路Torch/OpenCV设1线程 | spawn进程池 |

保留2脸/4手是为了发现歧义，不因已确认驾驶员ROI而假设模型永远只看到一个人。MediaPipe原生内部线程不一定服从全部线程环境变量。保留原始帧感知与25Hz因果采样，不通过降采样、跳帧或删掉gaze加速。

12路是候选并发配置，不是12份NPU模型。NPU服务当前逐帧、batch=1；它串行处理带鉴权的本机请求，视频解码和MediaPipe并行。没有宣称实现NPU批推理或最大吞吐。实际瓶颈也可能是MediaPipe、解码和帧传输；主机可见240GiB内存不等于容器CPU/内存额度无限。

## 在服务器执行两条命令

先提交并同步本轮源码到服务器，在 `/mnt/workspace/DriveGuard-AI` 根目录执行。以下安装和下载**只在服务器**发生，不在本机执行。

```bash
bash scripts/server/setup_yolo11.sh runs/dmd-full-gaze.local.json
bash scripts/server/start_extraction.sh
```

前提：上一轮全量配置已经生成，里面每段视频的ROI确认符合实际；已有数据、Face/Hand工件与 `.venv-perception-arm64` 保留。如果只完成配置生成而未确认全部ROI，启动会列明需完成确认，不会替你批量改成true。

第一条创建专用 `.venv-yolo-npu`，用 `--system-site-packages` 复用原 PyTorch 2.7.1 / torch_npu 2.7.1.post4，安装固定 Ultralytics 8.3.199 及其依赖。固定CPU torchvision 0.22.1用于Python依赖兼容，不代表检测前向在CPU。不会往原NPU或感知虚拟环境安装包。

随后从[官方固定来源](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)下载权重到 `models/local/yolo11s.pt`，记录本地hash/来源receipt；生成独立 `runs/dmd-full-yolo11.local.json`、runtime和本机鉴权token。原配置、全部ROI、preparation hash和MediaPipe工件都保留。已有目标文件拒绝覆盖；配置已存在时setup不会重复安装。

第二条nohup启动监督进程，依次执行：NPU真实前向预热 → 全量提取 → 自动发布训练manifest → 关闭检测服务。预热不通过则失败并保留日志，**不自动退回CPU，也不开始浪费时间提取**。这不是额外的手工test.sh流程。nohup防终端断开，不保证容器销毁、关机或平台回收后继续运行。

默认新缓存目录 `runs/dmd-full-yolo11-features-001`。更换检测器后不能复用YOLOX旧特征；旧缓存不删除，新配置原样重跑则复用校验通过的完整记录，不支持视频中段续提。

## 日志、停止与恢复

启动会输出实际 `runs/extraction-session-XXXXXXXX`。把下面SESSION替换为它：

```bash
bash scripts/server/status_extraction.sh runs/extraction-session-XXXXXXXX
tail -F runs/extraction-session-XXXXXXXX/supervisor.log runs/extraction-session-XXXXXXXX/extraction.log
```

- `detector.log`：设备、模型加载及推理错误。
- `extraction.log`：任务进度；完整逐视频日志在新缓存目录 `logs/`。
- `publish.log`：监督覆盖及成功manifest路径；`candidate.json`不是训练输入。
- `status.json`：阶段快照；status命令额外校验监督进程PID及开始时间，避免把残留文件误当运行中。

停止及稍后恢复：

```bash
bash scripts/server/stop_extraction.sh runs/extraction-session-XXXXXXXX
# 等 status 显示 stopped/failed/complete 且 supervisor_alive=false 后再启动：
bash scripts/server/start_extraction.sh --resume
```

停止后不再派发新视频，已运行视频完成当前整段；发布阶段等当前校验/发布释放锁再退出。可能需要等待，并非立即杀进程。重复启动由输出锁和服务端口拒绝；不自动抢锁。强制kill或机器故障留下锁时，先确认相关进程都退出，再人工处理明确的残留文件，不能直接删除整个输出目录。

需要降低并发时，在停止完成后：

```bash
bash scripts/server/start_extraction.sh --resume --workers-dist 4 --workers-drows 2 --workers-gaze 2
```

运行期间不要改代码、依赖、权重、原标注或配置；这些都会影响缓存来源。服务指纹改变时拒绝混用同一记录。鉴权token只用于本机回环通信，不允许将端口对外暴露。

## 完成后开始NPU训练

只有 `status=complete` 且 `publish.log` 给出成功manifest才表示数据准备成功。四行为正负支持、gaze支持和有效性检查沿用现有规则，不把未知补成负类。如果失败，读取对应日志即可，不要从失败目录拿候选清单训练。

```bash
/opt/buildtools/Python-3.11.4/bin/python3 -m training.gaze.npu_train \
  --config configs/training/relation_gaze.json \
  --manifest runs/dmd-full-yolo11-features-001/manifests/实际发布ID/manifest.json \
  --data-root runs/dmd-full-yolo11-features-001 \
  --epochs 10 \
  --output-dir runs/relation-gaze-yolo11-npu-full-001
```

训练使用原NPU环境和CANN变量，不使用MediaPipe虚拟环境。训练策略、续训和评估见[全量训练指南](GAZE_FULL_RUNBOOK.md)。本轮不会自动开启训练，也没有微调YOLO11/MediaPipe；它们冻结生成输入，训练的是门控时序联合模型。

## 来源、许可和验证边界

Ultralytics固定为[8.3.199源码](https://github.com/ultralytics/ultralytics/tree/v8.3.199)。YOLO11采用[Ultralytics的AGPL-3.0／Enterprise许可路线](https://www.ultralytics.com/license)，并不继承YOLOX源码许可；闭源分发或商业集成前另行确认适用条款，本配置不是法律结论。checkpoint由Ultralytics加载，可能包含pickle，仅信任此固定官方来源，不能加载来历不明的同名文件。

下载后SHA-256是本地完整性记录，不是上游签名或真实性证明。官方权重仍是通用COCO模型，不能由启动成功推断DMD手机/杯子召回足够；提取质量报告和后续真实评估才是依据。当前未验证Ascend910B3上的YOLO11实测兼容性、速度和准确率，服务预热负责及早暴露算子问题。
