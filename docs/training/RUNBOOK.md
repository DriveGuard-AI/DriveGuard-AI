# TCN 训练工程使用说明

负责人：shadow · 分支：model · 2026-09-19更新。已有四任务 TCN、[特征桥接](PERCEPTION_BRIDGE.md)及[YOLOX/MediaPipe入口](EXPERT_RUNTIME.md)；三个真实工件通过独立Windows CPU加载/空白帧前向，见[准备记录](PERCEPTION_SETUP.md)。真实视频验证、融合、后端和完整系统未完成。12特征/24通道基线与19特征/38通道候选均未真实训练；感知依赖与既有昇腾训练环境保持隔离。

## 1. 环境与入口

服务器目标 Python 3.11、PyTorch 2.7.1；昇腾叠加 torch_npu 2.7.1.post4/CANN 9.0.0。使用已配置环境，不盲目升级。依赖安装仍为 `python -m pip install -r requirements.txt`；昇腾用 `python -m pip install -r requirements-npu.txt`，后者已经引用前者，不必重复执行。

本机工程验证使用已有忽略环境：Python 3.12.14、torch 2.14.0+cpu、NumPy 2.3.5。测试使用通用接口，但不能据此宣称目标 torch 2.7.1、GPU 或 NPU 已通过完整训练验证。

所有命令从仓库根目录运行，用目标环境的 `python`。两个入口均通过模块方式调用，不直接运行子目录文件。

```bash
python -m training.temporal.gpu_train --check-config
python -m training.temporal.npu_train --check-config
```

预检只使用标准库，不加载 torch/torch_npu、不读取数据、不触发设备初始化；返回 config_valid、device_verified=false。`python -S` 同样可预检。它只证明配置格式有效，不证明硬件或算子可用。

## 2. 先做显式 CPU 合成测试

运行完整测试前先补齐测试依赖（OpenCV；Pillow 已随 `requirements.txt` 提供）：

```bash
python -m pip install -r requirements-test.txt
```

```bash
python -m unittest discover -s tests -v
python -m training.temporal.smoke --output-dir runs/temporal-smoke
```

smoke 明确使用 CPU、1 个计算线程，生成四个虚拟主体的录制级数组，训练两轮后从 last.pt 续训到第三轮。所有输出放指定新目录，不扫描/下载 DMD。重复运行需要换一个新目录，程序不会清空已有结果。

合成标签不代表驾驶行为；smoke 权重、loss、F1 仅证明工程闭环，不能作为 DMD 成绩或正式发布模型。

## 3. 真实特征训练（数据准备后）

当前没有从 DMD 原视频自动提取特征的脚本。不要将下载的压缩包或视频路径直接传给 --manifest。先按[数据契约](DATA_CONTRACT.md)准备有版本信息的特征清单、数组与训练/验证主体划分；本轮数据任务仍暂停。

若上游已有符合协议的专家观测 JSONL，可用感知桥接导出未标注数组；补齐真实标签和 manifest 后，使用 `--config configs/training/temporal_interaction.json`。导出成功不代表已有可训练标签，record.json 不能代替 manifest。

下面的 `/mnt/dmd-features` 是路径示例，必须替换为实际特征目录；清单中的数组路径相对 data-root。

```bash
# Huawei Ascend
python -m training.temporal.npu_train --config configs/training/temporal_baseline.json --platform-config configs/training/temporal_npu.json --data-root /mnt/dmd-features --manifest /mnt/dmd-features/manifest.json --output-dir runs/temporal-npu-001

# NVIDIA CUDA
python -m training.temporal.gpu_train --config configs/training/temporal_baseline.json --platform-config configs/training/temporal_gpu.json --data-root /mnt/dmd-features --manifest /mnt/dmd-features/manifest.json --output-dir runs/temporal-gpu-001
```

也可用 DRIVEGUARD_DATA_ROOT 指定数据根目录，显式 --data-root 优先。manifest/output 路径相对运行目录解析，配置文件位置不改变路径基准。机器路径不要提交到公共 JSON，可放被忽略的 `configs/training/*.local.json`。

设备不可用时明确失败，不自动退回 CPU。GPU 入口不导入 torch_npu；两入口最终调用同一个模型与训练循环。首版 FP32、Adam、num_workers=0，不包含 AMP/分布式/专用算子优化。

## 4. 输出、评估与恢复

每次运行使用全新输出目录：

| 文件 | 内容 |
| --- | --- |
| config.json | 解析后的公共和平台配置 |
| data_summary.json | 数据/profile/标签映射/数组哈希、主体划分与正负未知计数 |
| environment.json | Python/torch/NumPy、平台设备、Git revision/dirty 状态 |
| history.json | 每轮全局逐类聚合损失、P/R/F1、覆盖任务及验证分组指标 |
| last.pt | epoch 边界可续训状态：模型/优化器 CPU tensors、RNG、shuffle generator、最佳工件与历史 |
| best.pt | 验证损失最优的纯模型交付工件，kind=weights，不用于续训 |
| *.pt.sha256.json | 对应 checkpoint 的 SHA-256 完整性校验 |
| result.json | 完成轮次、步数、最佳轮次、工件路径与合成标记 |

每类只在已知标签上计算指标；未启用任务记 available=false、指标 null。阈值固定 0.5 为基线，零分母 P/R/F1 记 0；报告同时保留正负/未知计数，避免误读。验证按主体、子集、视角分组；测试集不用于选模型，独立测试集评估/阈值标定属于后续工作。

```bash
python -m training.temporal.npu_train --config configs/training/temporal_baseline.json --data-root /mnt/dmd-features --manifest /mnt/dmd-features/manifest.json --epochs 15 --resume runs/temporal-npu-001/last.pt --output-dir runs/temporal-npu-002
```

以上表示把原配置的总轮次从 10 增加到 15，而不是额外再跑 15 轮。last.pt 及其 checksum 文件须一起传输；CPU state_dict 允许平台交接，但不同设备不承诺逐位复现。

续训只允许改变 runtime（设备/路径）与增加总 epochs；改变结构、特征顺序、任务、数据文件内容、划分、normalizer 或训练超参数会拒绝。最佳权重也随 last.pt 保存，因此续训到新目录不会遗失先前最优模型。只加载可信来源工件，显式使用 weights_only=True；checksum 只能检测损坏，不证明来源可信。

保存采用单文件临时写入再替换；checkpoint 与 checksum 不是跨文件事务，掉电中断可能造成校验不匹配。发现不匹配应保留现场并使用已备份完整工件，不绕过校验。正式长训练前应配置独立备份/保留策略。

## 5. 常见拒绝原因

- 配置拼写/重复 JSON 键、非有限参数、未知字段、平台文件试图修改模型：修复公共配置，不忽略错误。
- 某启用任务在 train/val 缺少正例或负例：补齐合法标注，或显式命名为部分任务实验；不得自动将未知改负例。
- 同主体跨子集/视角泄漏：重新统一主体划分；同一录制/视角只引用一次完整数组。
- NaN、错误形状、非二值 mask、未知标签 target 非零、路径逃逸：修复数据生成器。
- 整个 epoch 无有效监督：失败而非“loss=0 成功”；全未知 batch 不执行 forward/backward/optimizer，但仍计入覆盖统计。
- 已有 output-dir：选择新目录，程序不删除或覆盖已有结果。

## 6. 正式交付边界

此工程不输出最终 SAFE/WARNING/DANGER，不包含实时质量判定或已校准概率。完整专家接入时升级 profile 与输入通道，并同步[完整模型设计](../architecture/MULTI_EXPERT_DESIGN.md)。正式权重必须开源且来源许可明确，发布流程见[权重目录说明](../../models/releases/README.md)；目前没有真实 DMD 训练成果。
