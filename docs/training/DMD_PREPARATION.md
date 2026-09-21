# DMD 真实训练数据准备

本页说明标注语义与准备清单。已接通**批量感知、标签数组、记录级恢复和训练 manifest 发布**，操作见 [DMD 批量训练数据流水线](DMD_BATCH.md)。`preparation.json` **仍不能直接传给训练器**，只有 publish 成功生成的 `manifest.json` 才是训练输入。

用户已回传 Ascend910B3 上 10 轮合成训练、断点续训、参数更新和 NPU 重载推理成功；这说明训练工程链路可运行，不代表已经训练真实 DMD。

## 1. 两份按被试划分

配置：`configs/dmd/split.subjects.json`。

| 用途 | 被试 |
| --- | --- |
| train | 1、5、6、9、10、13、23、29 |
| val（留出评估） | 7、14 |

没有第三份 test。同一个人的所有子集、录制、摄像头始终属于同一划分。训练器用 val 选 `best.pt`，因此这份评估不能再称为完全独立的测试成绩。新增被试必须明确加入其中一份，不自动分配。配置中尚未下载的被试会在报告里列出；这不是最终训练数据覆盖验证。

## 2. 首版四任务含义

策略描述保存在 `configs/dmd/labels.v1.json`，实际规则由 `training/dmd/labels.py` 实现。修改标签规则须同步代码、配置说明、测试和实验记录。

| 输出头 | 本版正例 | 本版负例 | 边界 |
| --- | --- | --- | --- |
| phone_use | distraction 中明确 phonecall_left/right、texting_left/right | 明确 safe_drive | 不是看见手机就算使用 |
| drinking | distraction 中明确 drinking | 明确 safe_drive | 不是看见杯子就算喝水 |
| fatigue | drowsiness 中明确哈欠，或已观察达到配置时长的持续闭眼 | 覆盖区间内明确睁眼且无哈欠事件 | **疲劳相关事件代理，不是真实疲劳诊断** |
| distraction | distraction 中明确 gaze_on_road/not_looking_road | looking_road | 仅离路注意代理，不等同危险分心 |

未标注、冲突、无法解释的对应标签轴均为未知：`target=0, label_mask=0`，不是负例。不同任务独立掩码，例如驾驶行为 unclassified 不必抹掉有效的 road/nonroad 标签。闭眼未达到时长、眼睛正在开闭、undefined 都不作为疲劳负例；明确哈欠可独立提供正例。

用户已确认首轮持续闭眼时长 `closed_eye_min_ms=1000`，现作为配置默认值。它是工程事件标签参数，不是医学阈值。可以显式设置其他正整数毫秒做对比；不同阈值不能复用原标签缓存。旧 preparation 中的 `null` 不必手改，batch init 会在新批量配置中明确填写 1000。

gaze 子集当前保留在准备清单中，`supervised_tasks=[]`，不自动把后视镜/仪表区域折算成危险分心；hands 标注作为可追溯辅助来源，不当成手机或喝水行为标签。待定义并验证 gaze_zone 映射后才能用于对应监督，不丢弃原始资料。

## 3. 标注与视频如何对齐

DMD 的标注帧处于公共 mosaic 时间线上，各原始摄像头开始时刻不一致。根据官方 [DEx 的帧换算代码](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/blob/master/exploreMaterial-tool/accessDMDAnn.py) 与 [stream shift 读取代码](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/blob/master/exploreMaterial-tool/vcd4reader.py)，实现使用以下等价换算：

```text
offset[view] = frame_shift[view] - min(face_shift, body_shift, hands_shift)
annotation_frame = decoded_source_frame_index + offset[view]
```

这里要求 face 的参考 shift 为 0，三个摄像头同步声明均存在；视频文件本身可以缺少 hands。换算使用声明，不因为手部视频未下载而改变时间原点。

实际解码使用 PyAV 的 PTS，经现有解码器转换为 `ceil(relative_PTS_ms)`。每个 25 Hz 槽选择最新过去帧，不能直接拿槽序号查原始标签，也不采用 `reported_fps` 猜时间。声明总帧数只作诊断，输出不超过实际解码的最后一帧；标注覆盖外与陈旧超过 80 ms 的源帧不产生监督。

持续闭眼计时只用已经看到的源帧 PTS：达到阈值之前不回填正例；未知/冲突眼状态、标注空洞或超过 80 ms 的源帧间隔打断连续计时。某摄像头晚开始时，计时从该视角首次实际观察开始，不推断此前闭眼了多久。

首版纯标签核心每段录制最多输出 100,000 个槽（25 Hz 约 66.7 分钟）。超过上限会明确拒绝，不静默截断；这是防止异常 PTS 跳变引起巨量内存分配的工程限制，不是模型窗口长度。更长素材需后续实现带连续状态的分块处理，不能随意切开后重置闭眼历史。

## 4. 服务器现在执行的正式命令

把本轮新增代码和配置同步到服务器后，在仓库根目录执行。下面假定已有审计报告位于 `runs/dmd-audit-001/audit.json`；如果你的报告在其他目录，只调整 `--audit`。

```bash
python3 -S -m training.dmd.prepare \
  --audit runs/dmd-audit-001/audit.json \
  --dataset-root Dataset \
  --output-dir runs/dmd-preparation-001
```

这是标准库准备命令，不需要安装依赖、激活感知环境或修改 NPU 环境。不再通过临时 `test.sh` 组织这一步。

输出：

- `preparation.json`：每段录制 face/body 视频的相对路径、被试划分、标注 hash、同步 offset、声明帧数、待配置 ROI、监督任务范围；gaze/hands 同录制的原始标注来源也记录 hash。
- `README.md`：清单用途和当前边界。

`status=needs_configuration` 是预期状态，不是训练失败。准备工具会重新读取标注并比对审计 hash；源标注变化、被试身份不一致、错误审计、跨划分重复被试、越界路径会停止。缺少某个 face/body 视频会明确列入 skipped。输出必须是 Dataset 外的新目录，不覆盖原资料或旧结果。它不对大视频做完整 hash，也不宣称视频可解码。

可通过 `--closed-eye-min-ms` 显式覆盖首轮 1000 ms。准备命令本身仍不会开始感知或训练；已有 116 个任务的 preparation 无需重新生成，直接使用 batch init 接续。

## 5. 标签转换 API 与批量接线

纯函数 `compile_labels(document, view, frame_times_ms, *, closed_eye_min_ms, subset)` 接受原始 OpenLABEL 字典与**完整实际解码源帧**的 PTS 毫秒序列，返回槽时间、源帧/标注帧索引、四任务 targets/masks 和统计。不能将均匀生成的时间序列冒充真实 PTS。它不加载模型、不分配 NPU、不写 `.npy`。

批量入口沿用现有感知与训练器：

1. 确认驾驶员 ROI 和排他性。同一画面可能有乘客，标注里出现 driver ID 不等于 ROI 已经排除乘客。不同摄像头像素坐标不跨视角拼接。
2. 批量感知执行时保存源帧 PTS/索引，调用标签核心，输出五数组及来源/规则 hash；中断后按已完整发布的录制恢复。
3. 检查每个激活任务在 train/val 中有效窗口的正、负例和有效输入覆盖，再发布兼容现有训练器的 manifest。没有覆盖的任务要明确调整，不能静默带入训练。
4. 在原 NPU 环境以真实 manifest **从头训练**关系门控 + TCN + 四头；不把此前合成 checkpoint 当预训练权重。

感知执行在独立 ARM64 CPU 环境，NPU 训练在已经验证的原环境。这个分工没有改变最终使用 NPU 训练时序模型的目标。当前本机通过的是合成感知输入到 CPU 训练的工程测试，服务器真实提取与训练尚需执行。
