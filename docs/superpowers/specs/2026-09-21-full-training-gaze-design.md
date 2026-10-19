# 完整训练目标：四行为任务 + 注视区域辅助任务

日期：2026-09-21；负责人：shadow。

状态：用户于2026-09-21明确批准编码；新版实现、独立审核与本机CPU工程验证已完成，见[验证记录](../../experiments/20260921-gaze-joint-training.md)。服务器全量提取、新版NPU训练和真实精度尚未验证。旧缓存不作为新版训练前置条件；正式数据只按新版全量提取，不要求先重跑旧八段小批次。

## 1. 已确认目标与证据

- 用户要求面向部署、多平台兼容，Drows/Dist/Gaze 都有效参与训练；保留 CPU 离线感知、GPU/NPU 特征模型训练的环境分离，不增加论文型算法或大型视觉网络。
- 服务器已完成旧版 8 个视频任务。发布报告中手机任务 train/val 正例均为 0，其他三任务具备正负窗口。随后标注扫描定位到 s2 的 phonecall/texting；用户已确认 gA/1/s2、gB/7/s2 的 face/body 无其他人的脸或手。
- 本地真实 gaze 样例的动作类型为九种有效 gaze_zone、not_valid 以及 blinks/blinking，与 DMD 官方注视区域标签一致。该样例只证明这些类型存在，不能证明全体被试或各划分的类别覆盖。
- 设计前的旧版MediaPipe适配器未请求脸部变换矩阵，head_pose恒为空；新版已显式请求并校验。旧导出只保存特征数组、不保存完整关键点，不能从旧缓存推导新头姿/虹膜特征。
- 当前 NPU 合成训练验证不等于新版多任务训练验证，也不等于真实 DMD 精度。

## 2. 方案选择

1. **推荐：保留四行为输出，增加共享面部时序表示的九类 gaze 辅助头。** 不改四项行为的语义，新增计算量小，gaze 监督能更新共享编码器；需要扩展标签数组、损失、评估和 checkpoint。
2. 将 gaze_zone 合并成原 distraction 的二分类：改动较少，但镜子、侧窗与驾驶相关注视不能无条件解释为危险，且损失原始区域信息；本轮不采用。
3. 另建独立的图像级 gaze 网络：表达能力可能更高，但增加感知工件、图像训练、平台适配和存储成本；不作为第一版。

五任务是四个独立二分类任务加一个九分类任务，不是把所有行为变为五选一，也不是创建九个互不约束的 sigmoid。

## 3. 监督目标

四行为保持顺序 fatigue、phone_use、drinking、distraction。fatigue 仍是哈欠/持续闭眼相关事件代理，闭眼默认阈值 1000 ms；distraction 仍由明确的 gaze_on_road 标签提供道路注意力代理，不自动变成危险风险真值。

新增 gaze_zone，固定类别顺序如下：

| class_id | 原始 gaze_zone |
| --- | --- |
| 0 | left_mirror |
| 1 | left |
| 2 | front |
| 3 | center_mirror |
| 4 | front_right |
| 5 | right_mirror |
| 6 | right |
| 7 | infotainment |
| 8 | steering_wheel |

- not_valid、未标注、未知类型和同轴多类冲突均为未知；存储占位类别 0、独立 mask=0，不是 left_mirror 正例，也不是安全。
- 复用实际解码帧 PTS、三摄像头 offset 和 25 Hz 因果采样；窗口标签取末槽。保留 raw 覆盖与有效输入后的覆盖，不用固定 FPS 代替实际 PTS。
- gaze 标签仅在 gaze 子集产生；该子集的四行为标签继续未知。Drows/Dist 的 gaze 标签未知，不把行为标签反推成注视区域。
- 标注 blinks 不是疲劳真值，也不直接成为模型输入；眼部几何失效由观测质量决定。
- gaze 真值、subset、subject、recording_id、绝对视频时刻、采集顺序、标签 mask 不作为模型输入。

## 4. 输入特征与平台边界

建立显式新版特征 profile：原 19 维 + 左右虹膜在各自眼局部坐标中的水平/垂直相对位置，共 23 维数值，拼接同形有效性 mask 后为 46 通道。

- 在新版感知模式请求 MediaPipe 面部变换矩阵，提取可追溯的相对头姿估计，填充原 yaw/pitch/roll 三个位置。矩阵与被选中的同一张驾驶员脸配对；不能通过排序或取第一张脸错配。
- 头姿必须明确 canonical-to-observed 变换、坐标转换、旋转顺序和符号。使用对应版本 API/源码及已知旋转测试核验；非有限值、退化矩阵、无法稳定分解时 mask=0。该估计不是经相机标定的车辆坐标物理真值。
- 虹膜特征由 478 点拓扑计算，在眼角建立局部水平轴、垂直轴，以眼宽归一化；姿态旋转、尺度变化不能直接改变定义。468 点、闭眼、退化眼宽或虹膜位置明显不可信时屏蔽对应字段，不填“正视”。
- 虹膜位置是用于学习区域的视觉线索，不宣称 MediaPipe 已输出注视目标或精确视线射线。
- 新 gaze 特征进入现有 F 组，不新增重量级专家；保留 Objects、HO、FO 的数据含义。
- 感知继续使用已工作的 CPU 环境，训练保持 CPU/CUDA/Ascend 共用标准 PyTorch 运算。不得为了补 gaze 自动升级 CANN、torch、torch_npu 或 MediaPipe。

## 5. 模型耦合、损失与可用性

- 保留 F/O/HO/FO 四流投影、共享因果 TCN 和原四任务门控输出。
- 新 gaze 头从共享编码器的 F 流末槽表示输出九类 logits。gaze 损失更新 F 投影与共享编码器，因此不是独立部署一个不相连的分类器。
- 不将 gaze 真值或其 mask 拼进四行为输入；首版也不增加用预测概率再次反馈的循环耦合。
- gaze 有效输入要求 F 组可用，末槽有有效头姿和至少一只眼的两个虹膜字段，且相应观测在窗口中的有效比例达到显式 profile 阈值。仅有物体或嘴部特征不能支撑 gaze 输出。
- 训练与推理共用 gaze 可用性函数。不具备输入时输出 unavailable，不以零 logits 的 argmax 伪装某个区域。
- 四行为维持 masked BCE；gaze 使用 masked cross-entropy。每任务先按该任务有效样本数归一化，再按显式任务权重组合；初始各任务权重为 1，只对当前 batch 有监督的任务归一。全无监督 batch 跳过更新。
- 指标与验证损失按整轮任务累计统计，不平均 batch 均值。分类损失不因 gaze 样本数量多而直接淹没其他任务。
- 冻结 YOLOX/MediaPipe；本轮训练的是融合时序网络和新增辅助头。检测器微调、MediaPipe 训练、大模型预训练不在本轮承诺内。

## 6. 数据范围、缓存和清单

- 保持现有八人 train、两人 val 的跨子集统一划分，不按视频随机切分。val 参与模型选择，不称独立 test。
- 全量选择覆盖 distraction/drowsiness 的 face/body。gaze 首版使用 face 视角作为区域监督主来源，body 不自动产生额外 gaze 监督样本；对应已有统计为 94 + 11 = 105 个提取任务，最终以实际准备清单为准。
- 选择 face 视角是为区域/眼部证据与机位一致性，不是丢弃 gaze；原始 body 文件保留。若后续要求 gaze body 参与，先验证跨机位特征与类别效果，再显式扩展范围。
- 新版 record/manifest/training config 使用新 schema，不修改旧数据含义。除现有数组外增加 gaze_targets int64 [N] 和 gaze_label_masks uint8 [N]，数组加入 hash 与校验。
- **旧 8 段缓存不能直接满足新版 23 维输入，需要从原视频重新提取。** 旧缓存、失败报告、旧 checkpoint 均保留；新版使用独立输出目录，禁止静默补零升级或删除旧结果。
- 旧准备清单可显式迁移：核对 annotation/source hash 后，在新清单中定义 gaze 监督范围，保留任务身份及被试划分；不要绕过旧清单完整性检查或直接修改其 hash。
- 已确认 12 段的 ROI 可以按精确视频身份复制到新版配置，其他视频仍需检查。不把一个视频的确认自动扩散到整个数据集。
- 全量覆盖报告列出每任务/每类的 raw、usable、unknown、view/subject 分布。gaze 各类别在 train/val 的缺失、输入质量不足必须明确报告，不自动伪造类别、移动被试或缩减任务。
- 缓存只保存录制级数组，不落盘全部图片或重叠窗口，读取采用 mmap；断点按视频恢复，已完成且指纹一致的新版任务可复用。

## 7. 训练与发布

先做本机自动化 CPU 小样本工程闭环，随后交付服务器全量命令，不要求用户继续拼临时 test.sh。

首轮配置沿用 seed=42、FP32、batch_size=16、epochs=10、learning_rate=0.001、weight_decay=0.0001、gradient_clip_norm=1、num_workers=0，作为可调整基线，不承诺该轮数足够收敛。使用全量有效数据，保持四行为和 gaze 同时启用。

新增 checkpoint 记录模型类型、23 维 profile、四行为顺序、九类 gaze 顺序、任务权重、可用性 profile、归一化、来源 hash、优化器与 RNG。新版从头训练可训练层；旧版 checkpoint 禁止当作精确断点直接恢复。

保存 best.pt、last.pt、训练曲线、四行为逐任务 precision/recall/F1，以及 gaze 的混淆矩阵、逐类支持数/召回、macro-F1、准确率、不可用覆盖；分被试和视角报告。缺类指标明确标记，不宣称完整性能。正式权重按已有开源发布要求准备模型卡及来源许可记录，不自动上传。

推理适配返回原四行为结果及独立 gaze_zone 概率/可用性，公共 API 和数据库保持现有接口；若对外暴露 gaze 字段，另同步接口契约。不得直接把 mirror 类输出连成危险告警。上线阈值和驾驶上下文规则需要真实效果验证。

## 8. 实施批次与验收

1. **特征与监督契约**：真实标签样例核对、版本配置、头姿/虹膜几何、未知/冲突规则、时间对齐。单元测试覆盖已知旋转/符号、468/478 点、闭眼、归属、多脸配对及退化输入。
2. **模型与训练**：新增头、mask loss、指标、task availability、配置/模型工厂、checkpoint。测试单一任务 batch、混合监督、全缺失、有限梯度、旧模型兼容与错误版本拒绝。
3. **批处理与清单**：新版 prepare/init/run/publish 接线、ROI 精确迁移、105 任务实际选择、来源与数组校验、缓存隔离。模拟输入跑真实转换和发布，不把模拟输入报告成 DMD 精度。
4. **端到端与文档**：CPU 联合训练、续训一致性、重载推理与全套回归；同步数据契约、模型设计、训练指南、当前限制和实验记录。交付已有服务器环境可执行的全量提取及 NPU 训练命令。

本地没有服务器完整视频与 NPU，不宣称新版 NPU、真实 gaze 可用率或准确率已通过。目标平台的新头/损失组合仍以服务器执行结果确认。DMD gaze 固定注视顺序可能形成时序捷径，因此禁止输入绝对时间，并在报告中说明该数据集协议不能替代自然驾驶泛化验证。

## 9. 依据

- [DMD 官方 gaze 标注规范](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset/wiki/DMD-gaze-related-action-annotation-criteria)：九个有效区域、not_valid 的语义、face 主视角及固定区域顺序。
- [MediaPipe Face Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)：478 关键点、可选面部变换矩阵；不据此宣称物理视线估计已经成立。
- [MediaPipe Iris](https://chuoling.github.io/mediapipe/solutions/iris.html)：虹膜定位本身不推断注视目标。
- 当前实现：`inference/experts/mediapipe_adapter.py`、`inference/perception/export.py`、`training/relation/model.py`、`training/dmd/labels.py`、`training/dmd/batch_config.py`。

## 待确认的设计决定

用户已确认：四行为 + 九类 gaze 辅助头、gaze 首版 face 主视角监督、新版全量特征提取；不要求保留旧缓存为训练输入，本轮不主动删除用户文件。按此执行，无需再次询问是否开始编码。
