# 关系门控模型训练与服务器接入

负责人：shadow。范围：四证据流模型骨架、共享训练循环、CPU合成验证、GPU/昇腾入口。2026-09-21 用户回传昇腾910B3合成训练10轮/30次更新、续训及权重重载推理通过；不代表真实DMD精度、GPU验证或长程硬件验收。

**当前真实数据准备入口：[DMD 批量提取与训练](DMD_BATCH.md)**。已有[准备清单](DMD_PREPARATION.md)后，使用 init/run/publish 配置 ROI、提取带标签数组、检查覆盖并发布 manifest。持续闭眼首轮 1000 ms；无需重复环境测试。

**首次同步服务器先执行[服务器初始化指南](SERVER_SETUP.md)**：确认提交与测试源码齐全，安装 `requirements-npu.txt`，核对环境，再运行本页合成训练。Git 不同步本机虚拟环境、预训练工件和 `*.local.json`。

## 1. 已准备的文件

| 文件 | 职责 |
| --- | --- |
| [model.py](../../training/relation/model.py) | F/O/HO/FO投影、共享因果TCN、四任务门控与分类头；默认13,524个可训练参数 |
| [relation_gated.json](../../configs/training/relation_gated.json) | 模型超参数、启用任务、窗口、训练参数和平台路径 |
| [relation_groups_v1.json](../../configs/perception/relation_groups_v1.json) | 四流分组、年龄顺序、依赖与质量阈值；提取时显式选用 |
| [preprocessing.py](../../training/relation/preprocessing.py) | 缺失依赖、历史屏蔽、质量与资格计算；训练/推理复用 |
| [共享训练文件](../../training/temporal/train.py) | FP32训练、验证选best、Adam、有效监督计数、epoch续训 |
| [CPU入口](../../training/relation/cpu_train.py) / [GPU入口](../../training/relation/gpu_train.py) / [NPU入口](../../training/relation/npu_train.py) | 默认加载新模型JSON，分别选择CPU/CUDA/昇腾，不静默回退 |
| [模型工厂](../../training/common/model_factory.py) | 区分新旧模型、读取校验过的checkpoint并重建模型 |
| [合成训练工具](../../training/relation/smoke.py) | 独立生成小数据，验证2轮训练→续训第3轮→连续训练比较→best重载 |

现有24/38通道TCN配置及入口保持可用；新模型使用独立`driveguard-training-v2`配置，不能把新字段塞进旧v1配置。三个上游感知模型保持冻结，不需要在NPU训练进程导入YOLOX/MediaPipe。

## 2. 本机CPU合成验证

在项目根目录执行（输出目录必须不存在，每次复验换一个目录名）：

```powershell
.\.venv-perception\Scripts\python.exe -B -m training.relation.smoke --output-dir runs/relation-smoke-local-001
```

Linux上使用已安装NumPy/PyTorch的Python：

```bash
python3 -m training.relation.smoke --output-dir runs/relation-smoke-server-001
```

产物包含synthetic_data、initial/resumed/continuous下的best.pt、last.pt及SHA-256文件，和smoke_report.json。只有`resume_matches_continuous=true`、`reload_finite=true`才表示这个工程闭环通过。合成权重不是可部署成果，合成loss/F1不作为DMD成绩。命令不下载任何数据或模型。

## 3. 服务器训练输入是什么

`--data-root`是**特征数组根目录**，不是DMD原始视频目录；`--manifest`是包含真实标签规则和主体划分的v2清单。仅更换原始视频路径并不足以训练。DMD batch run 已接带标签五数组，publish 检查通过后生成 v2 manifest；`preparation.json` 或失败的 `candidate.json` 不能传给 `--manifest`。

每段录制、单一视角对应五个`.npy`数组（N为25Hz采样槽数）：

| 文件字段 | dtype / shape | 语义 |
| --- | --- | --- |
| values | float32 [N,19] | 固定顺序的观测特征，有限值 |
| masks | bool/uint8/float32 [N,19]，二值 | 几何/归属/有效性；0代表未知，不是假正常 |
| expert_age_ms | float32 [N,3] | objects、face、hands观测年龄；0..81，缺失/超过80ms以81表示 |
| targets | float32 [N,4]，二值 | fatigue、phone_use、drinking、distraction固定顺序 |
| label_masks | bool/uint8/float32 [N,4]，二值 | 真实标签是否已知；未知位置target必须为0占位 |

完整契约见[DATA_CONTRACT](DATA_CONTRACT.md)。窗口不跨录制或主体；启用任务在train/val中必须分别有**输入可用的正例和负例**。训练中从未观测到的头姿等字段保持unsupported，不能在验证/部署擅自开启。

若某任务没有足够的已知正负例，应先从独立JSON的training.active_tasks移除，并在报告中标记未训练；不得伪造负标签来绕过预检。四个输出头仍保持固定顺序，未训练头不能用于业务结论。

### 提取v2特征

在能运行感知模型的机器上准备好本地runtime配置、ROI和工件后：

```bash
python3 -m inference.perception.video --config configs/perception/runtime.local.json --input /data/recording.mp4 --output-dir /data/features/recording-001 --group-profile configs/perception/relation_groups_v1.json
```

已有规范化专家JSONL时：

```bash
python3 -m inference.perception.export --input /data/observations.jsonl --output-dir /data/features/recording-001 --group-profile configs/perception/relation_groups_v1.json
```

默认不传`--group-profile`仍导出旧v1。新导出按真实观测时间生成年龄，不从旧cache伪造；每个record.json包含profile与数组hash。导出的行为标签仍全部未知，需要按数据集真实标注填充targets/label_masks，并分配全局subject_id及train/val/test，不能用子集名或几何阈值假造真值。

### manifest组成

下面展示必需字段的组合方法（`extractor_profile`和`group_profile`取自实际v2 record.json；`records`使用相对data-root路径）：

```python
manifest = {
    "schema_version": "driveguard-feature-manifest-v2",
    "dataset_version": "your-audited-dataset-revision",
    "feature_schema": record_metadata["feature_schema"],
    "extractor_profile": record_metadata["extractor_profile"],
    "group_profile": record_metadata["group_profile"],
    "group_profile_sha256": record_metadata["group_profile_sha256"],
    "label_mapping": {
        "version": "driveguard-behavior-v1",
        "tasks": ["fatigue", "phone_use", "drinking", "distraction"],
        "rules": audited_label_mapping_rules,
    },
    "records": audited_record_entries,
}
```

每个record entry必须有：`record_id/subject_id/recording_id/view/subset/split`及上表五个文件字段。例如`values="recording-001/values.npy"`。同一主体不得跨集合，多个子集里的相同人也要统一ID。labels经过修改后需保留标注来源；训练会重新计算所有实际数组hash，并将标签规则hash写入checkpoint。不要求将数组上传Git。

可以查看合成工具产出的完整manifest来理解JSON结构，但**不能把synthetic标记/规则/标签复制成真实训练元数据**。旧v1缺年龄只能重新提取或继续使用旧基线，不原地升级冒充v2。

## 4. 昇腾NPU服务器

沿用已工作的CANN/PyTorch/torch_npu环境。训练依赖见[requirements-npu.txt](../../requirements-npu.txt)，不要拿Windows感知版本快照覆盖服务器环境。

先做只检查JSON结构的预检，不访问数据/设备：

```bash
python3 -S -m training.relation.npu_train --check-config
```

特征、标签与manifest已准备后启动（示例路径需替换）：

```bash
python3 -m training.relation.npu_train --config configs/training/relation_gated.json --data-root /mnt/data/driveguard-features --manifest /mnt/data/driveguard-features/manifest.json --output-dir runs/relation-npu-001 --epochs 30
```

也可设置`DRIVEGUARD_DATA_ROOT`替代`--data-root`，或修改独立JSON的runtime路径字段。配置中的默认路径为null，不包含本机盘符。启动会分块校验并hash全部特征文件、核对主体/标签覆盖，然后训练；大数据的首次预检需要I/O时间，不应误认为已经开始GPU/NPU计算。

## 5. NVIDIA GPU与CPU

相同JSON与manifest，仅更换入口；各平台需匹配自己的PyTorch环境：

```bash
python3 -m training.relation.gpu_train --data-root /data/driveguard-features --manifest /data/driveguard-features/manifest.json --output-dir runs/relation-cuda-001 --epochs 30
python3 -m training.relation.cpu_train --data-root /data/driveguard-features --manifest /data/driveguard-features/manifest.json --output-dir runs/relation-cpu-001 --epochs 2
```

GPU/NPU不可用会明确失败，不静默换CPU。当前是单卡FP32、num_workers=0；CPU合成验证及用户回传的NPU短程合成训练已通过，GPU尚未实机验证。不能把`--check-config`通过当作服务器算子兼容证明。

## 6. 续训与读取权重

epoch级续训使用last.pt，输出必须是新目录，总epochs大于已完成轮数：

```bash
python3 -m training.relation.npu_train --data-root /mnt/data/driveguard-features --manifest /mnt/data/driveguard-features/manifest.json --output-dir runs/relation-npu-002 --epochs 50 --resume runs/relation-npu-001/last.pt
```

只允许平台路径/backend与总epochs改变，模型/特征/标签/profile/normalizer/数据内容及划分必须一致。CPU同环境连续训练和epoch续训可验证完全一致；跨设备训练轨迹不承诺逐位相同。

```python
from training.common.model_factory import load_model
model, metadata = load_model("runs/relation-npu-001/best.pt", backend="cpu")
# 使用metadata的normalizer与group_profile处理同一契约的特征窗口。
# model(x, quality, eligible, return_aux=True) 返回logits、weights、available。
```

只加载可信来源checkpoint，并保留`.pt.sha256.json`；校验和不是数字签名。输出的available仅表示窗口存在合格证据，不是已完成业务验证的task_support。未训练任务和未验证缺失组合仍须由部署适配器屏蔽，原始logits不能直接作为告警或SAFE结论。

## 7. 本轮未包含的工作

- DMD原始标签映射、真实视频验收、真实训练/精度和公开正式权重。
- YOLOX检测器微调或NPU检测后端、MediaPipe迁移NPU。
- 可选专家缺失增强（目前只确定性处理真实/合成缺失）、完整在线任务支持适配与风险告警。
- GPU实机训练、NPU真实数据/长程训练、ONNX/TensorRT、前后端与部署验收；NPU短程合成验证已由用户回传通过。

这些不妨碍先交付模型和训练骨架，但服务器真实训练前必须先具备合规的v2数据与对应硬件环境。
