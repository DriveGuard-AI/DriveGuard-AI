# 模型训练职责与交付约定

版本：0.2.0 · 日期：2026-09-18 · 算法工程师：**shadow** · 开发分支：**model**。

本约定落实项目负责人的任务分配。shadow 负责训练工程、实验和模型权重交付；后端、前端和车端集成由相应模块维护者对接。尚未指定姓名的对接角色由项目负责人协调，不虚构其他成员。

## 1. 分工与工作顺序

| 工作包 | 负责人 | 交付内容 | 当前安排 |
| --- | --- | --- | --- |
| 数据准备与切分 | shadow | 数据版本说明、相对路径清单、主体独立划分和校验 | 首批实现清单校验；DMD 原始标注转换后续接入 |
| TCN 训练 | shadow | JSON 超参数、model.py、GPU/NPU 入口、评估、checkpoint | 首个训练闭环 |
| YOLO 训练/微调 | shadow | 类别与标注映射、选定模型、双平台入口、评估与权重 | 后续工作包；具体 YOLO 版本和预训练权重未选定 |
| Face/Head 特征 | shadow | 特征顺序、单位、缺失值掩码、提取及归一化协议 | 首批定义 TCN 输入协议；不承诺重训 MediaPipe |
| 权重与结果交付 | shadow | CPU 可加载权重、校验和、模型卡、超参数、实验记录 | 建立规范；真实跑分和发布尚未发生 |
| 推理集成 | 模块维护者，shadow 协助 | 加载说明、输入输出一致性、ONNX 导出与误差验证 | 在有效训练权重产出后实施 |

## 2. 本地与服务器工作区

- 原始数据、下载包、解压数据和派生特征放在 `data/`、`datasets/` 或仓库外目录，通过 `.gitignore` 排除。
- 使用 `--data-root` 或 `DRIVEGUARD_DATA_ROOT` 提供当前机器的数据根目录，命令行优先；相对根目录按执行时的工作目录解释。
- 清单中的文件路径相对数据根目录，统一使用 `/`。不提交个人机器或服务器绝对路径。
- 本地允许无完整数据集，服务器可以有完整数据与缓存。代码、公共配置、schema、切分规则和清单摘要应一致。
- 不自动下载数据集。真实样本清单可能包含主体标识，放在忽略的数据目录；公开复现实验时记录来源版本、切分协议和清单 SHA-256，公开清单前检查可共享范围。
- 合成小样本由测试临时生成，不占用数据集存储，也不作为真实数据集性能证据。
- 虚拟环境仅保留在本机，`.venv/`、`.venv-*/`、`venv/`、`env/` 由 Git 忽略；仓库使用根目录 `requirements.txt` 管理共享 Python 依赖，不提交环境目录。

### 依赖安装

先按目标平台安装匹配的 PyTorch；昇腾还需按官方兼容矩阵配置 CANN 和 `torch_npu`，再在选定环境中执行 `python -m pip install -r requirements.txt`。不对已配置服务器执行无差别升级。

当前清单只覆盖首批 TCN 的 NumPy/PyTorch 依赖，暂未锁定版本，不代表已验证的 GPU/NPU 环境；`torch_npu` 不放入共享清单，避免 CPU/CUDA 平台误装。待服务器版本确认、训练验证通过后补充平台版本约束，并在实验记录中保存实际依赖版本。`requirements.txt` 替代的是环境目录的版本管理，不是 Python 环境本身。

## 3. 双平台训练契约

- GPU：NVIDIA CUDA + PyTorch，入口 `gpu_train.py`。
- NPU：华为昇腾 + CANN + PyTorch + `torch_npu`，入口 `npu_train.py`。
- 两入口共用模型结构、数据协议、损失函数和训练循环，差异收敛在设备初始化与平台配置。不能复制两套模型后分别维护。
- NPU 型号、CANN/PyTorch/torch_npu 版本尚待服务器提供；依照官方兼容矩阵选择组合，不自动升级服务器环境。
- 缺少请求的加速器或驱动时显式报错，不能静默改在 CPU 训练。CPU 仅作为明确选择的小样本验证设备。
- 首版采用单设备 FP32 和小模型，降低验证成本；AMP、分布式和特定算子优化另行实测后加入。
- 跨平台恢复首先保证权重、优化器状态与 epoch 连续；不承诺不同设备的浮点结果或随机数轨迹逐位一致。

## 4. 目录约定

| 路径 | 用途 | Git 策略 |
| --- | --- | --- |
| `training/temporal/` | model.py、训练循环、双入口、评估 | 提交 |
| `training/common/` | 配置、设备和清单校验 | 提交 |
| `configs/training/` | 公共超参数 JSON | 提交；机器专用 `*.local.json` 忽略 |
| `data/`、`datasets/` | 数据、标注和特征 | 忽略 |
| `runs/` | 临时日志、best/last checkpoint、训练环境记录 | 忽略；重要工件转交到发布目录/公开存储 |
| `models/releases/` | 待公开的选定 checkpoint、模型卡、校验和、链接 | 不按权重扩展名一刀切忽略 |
| `docs/training/` | 操作说明、模型结构、数据协议 | 提交 |
| `docs/experiments/` | 实验汇总、真实跑分、失败记录 | 提交 |

## 5. checkpoint 和开源交付

**公开交付训练 checkpoint 权重是本项目要求。只交代码而不交权重不满足模型交付目标。**

- 每个正式模型版本至少公开选定的 `best` checkpoint 和可续训的 `last` checkpoint（或同一 checkpoint 满足两者），附模型结构、完整配置、类别顺序、特征协议、代码提交号、数据来源/切分摘要和 SHA-256。
- 权重保存为 CPU tensor 的 `state_dict`，不序列化整个 Python 模型对象。续训工件还包含优化器、epoch、最优指标和随机状态；设备专用图编译产物不替代通用权重。
- 中间 epoch 文件无需全部提交到 Git。大权重可通过 Git LFS 或公开 release/模型托管下载，本仓库存放可追溯索引与校验和。具体托管方式在首个正式权重产出时确定。
- 自有代码沿用仓库 MIT；正式权重逐项注明许可、数据来源和预训练权重来源，不能自动继承代码许可。
- DMD 当前官方标注 CC BY-NC-ND 4.0，真实 DMD 训练权重能否按预期许可公开发布仍需核实；未确认前记录 `pending_permission`，不得宣称已开源。可行方案是取得许可或更换满足权重公开目标的数据/预训练来源。
- 本轮不进行远程发布、不生成虚构 checkpoint、不将随机初始化/合成训练权重称为 DMD 成果。

## 6. 实验与验收

实验 Markdown 记录运行 ID、负责人、commit、数据/清单哈希、配置、设备型号、软件版本、随机种子、耗时、指标、权重路径/哈希、局限和下一步。

窗口分类先报告每类别及 macro Precision/Recall/F1、验证损失；事件级 False Alarm/Hour、Detection Delay 和端到端 FPS 需要连续视频与完整管线，缺失时写“未测”，不能由窗口分数冒充。

优先顺序：配置和路径预检 → 合成数据单步/续训验证 → 少量真实数据跑通 → 固定预算实验 → 独立测试集评估 → 权重公开交付。避免在未知环境直接启动昂贵训练。

每次迭代检查文档影响；训练协议、超参数语义、checkpoint 格式改变时同步相关文档。沿用轻量同步约定，不增加无关审批。

## 7. 依据

- 项目负责人 2026-09-18 的 shadow 分工、双平台、外置数据与权重开源要求。
- [昇腾 PyTorch 与兼容矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。
- [PyTorch checkpoint 指南](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。
- [DMD 当前数据与许可说明](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset)（核对日期：2026-09-18）。
