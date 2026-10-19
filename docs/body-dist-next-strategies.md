# body_dist 后续策略：先验证分支互补，再改变视觉信息

日期：2026-09-24。范围：只读分析与实验建议；本文件没有启动训练、修改数据、模型或推理代码。现有 C_fusion + gap80 全任务评估由另一条工作线负责。

## 1. 当前证据与结论边界

数据来自用户提供的 stability `comparison.json`、`comparison.csv`、`aggregate.csv`，分别位于附件目录 `cb6344d9-f1b2-42ad-aa76-969df2e49136`、`ce2ed38a-8cd5-45e1-aa02-7e27155893aa`、`92cc05a7-2fb1-46d2-8fc8-b04eb0f44423` 的 `已粘贴的文本.txt`。以下数字已交叉检查 JSON 与 CSV；没有读取服务器上的逐窗口预测或原视频，因而不声称已经完成误例视觉核验。

**目标语义：** body_dist 是 body 视角上的离路注意代理，正例严格来自 `gaze_on_road/not_looking_road`，负例来自 `looking_road`，见 [labels.py](D:/DriveGuard-AI/training/dmd/labels.py:136) 和 [DMD_PREPARATION.md](D:/DriveGuard-AI/docs/training/DMD_PREPARATION.md:29)。手机使用、饮水是另外两个独立标签头；看见手机、手部动作或杯子不能等同于 dist 正例。后续误例审计、裁剪与时序实验均围绕 road/nonroad 标签展开。

同一验证集：P=258，N=1138；H 不可用组 P=134/N=991，H 可用组 P=124/N=147；只有 subject 14、7。A 与视觉各臂均有 1396/1396 个已知且可预测窗口。H 不可用不等于 body_dist 不可预测：其他专家仍能提供输出，见 [head_evidence.py](D:/DriveGuard-AI/training/gaze/head_evidence.py:27)。

| 模型 | 默认 logit=0 的 TP / FP | 各自验证集 FPR≤3% 的 TP / FP | 该低 FPR 点 H 不可用组 TP / FP |
|---|---:|---:|---:|
| A_gap80 | 86 / 26 | 89 / 31 | 0 / 2 |
| D_context，seed42/43/44 | 91/34；93/39；90/35 | 91/31；92/34；90/30 | 均 0 / 0 |
| C_fusion，seed42/43/44 | 129/56；83/18；102/51 | 103/32；113/33；93/34 | 29/12；17/6；18/18 |
| E_small，seed42/43/44 | 98/30；70/17；99/38 | 106/34；103/34；97/34 | 25/17；14/9；6/8 |

低 FPR 的平均 TP 为 D=91、C=103、E=102；D 的 91.333 是默认阈值结果，不能混用。C 相对 D 平均多 12 TP，其中 H 不可用组多 21.333 TP，而 H 可用组少 9.333 TP。分 seed 的 H 可用 TP 差为 -17、+4、-15，说明“融合损失可靠分支”是两颗 seed 上的现象，不是所有 seed 都成立。E 缩小宽度没有在低 FPR 下明确超越 C；现在不宜继续把“缩小头”当主线。

这些是开发集证据：最佳 epoch、阈值和“最佳 seed”均接触同一验证集，不能解释成独立测试提升。3% 的比较是同一上限，不是各臂严格相同的 FP 数。当前最佳默认 F1 是 C42 的 0.58239，但 FP=56、FPR=4.92%；低 FPR 最佳是 C43 的 TP113/FP33、召回43.80%，两者是不同目标。

## 2. 代码指向的主要瓶颈

1. **C 直接重学整个输出，没有可靠分支保护。** gap80 的各专家 embedding、H embedding、质量、eligibility 和 body_dist logit 被串成 context，见 [prepare.py](D:/DriveGuard-AI/training/visual/prepare.py:17)。视觉时序表示与 context 投影拼接后接一个新线性头，见 [model.py](D:/DriveGuard-AI/training/visual/model.py:27)；旧 logit 只是输入特征，不是残差基线。H eligibility 已存在，适合先做无新训练的固定路由诊断。
2. **全 ROI 224×224 + 冻结 ImageNet 特征可能损失头部/面部朝向的局部信息。** 当前保留整个 driver_roi，缩放并补黑边；输出经过全局池化成为512维，见 [encoder.py](D:/DriveGuard-AI/training/visual/encoder.py:32)、[encoder.py](D:/DriveGuard-AI/training/visual/encoder.py:86)。优先检验与离路标签相关的头部轮廓、面部朝向是否在缩放中失真；手/物体仅是可能相关的上下文候选，不能充当目标定义。若 body 侧视角没有拍到眼睛，裁剪无法恢复不存在的信息；仅眼球转动、头姿不变的样本可能仍不可区分。尚未观看误例，以上都是假设；不能把新 crop 当成原 cache 的同一特征。
3. **标签是窗口末端，视觉却约每0.5秒取一帧。** 原标签取 `stop-1`，见 [data.py](D:/DriveGuard-AI/training/common/data.py:351)；视觉八帧为 `[12,24,37,49,62,74,87,99]`，见 [visual/data.py](D:/DriveGuard-AI/training/visual/data.py:12)。已包含末帧，故不能声称“模型看不到当前帧”；可以检验短暂离路注视或转头/回头切换是否需要末端更密的历史。TCN 是因果的三层膨胀卷积、最后位置读出，见 [model.py](D:/DriveGuard-AI/training/visual/model.py:28)。
4. **缺帧不是这批结果的瓶颈。** 可用性要求八帧全部新鲜，见 [model.py](D:/DriveGuard-AI/training/visual/model.py:40)。但附件中所有验证窗口均可用；当前放宽 mask 不会直接新增可预测验证正例。其他部署数据可单独审计。

## 3. 三个有界实验，按顺序执行

所有新实验先固定成功标准：全体已知窗口的 TP/P、FP/N、F1、coverage；同一阈值下按 H、subject、record 分组；保存相对基线的新增 TP、丢失 TP、新增 FP、消除 FP 的逐窗口清单。每颗 seed 都报告，不通过挑一颗 seed 宣布成功。先跑第一个诊断，再决定是否付出视频重提取成本。

### 实验一：H 可用时保护 A，H 不可用时允许 C 补漏（最高优先级）

**假设：** 两颗 seed 的 H 可用损失可以通过固定路由消除，并在剩余误报预算中保留部分视觉收益。该假设允许被 seed43 的结果否定。

**固定规则：** `H eligible -> A`；`H ineligible 且 RGB available -> C`；RGB 不可用时回退 A（若 A 也不可用则继续不可用）。只作用于 body 视角的 distraction 列，其他行为和 gaze 仍由 gap80 输出。H 依据冻结 A 的 eligibility，禁止按真值或“这条预测是否正确”路由。

先使用服务器现有 `A_gap80-windows.csv` 与各 `C_fusion-seed*-best-windows.csv`，按 `(record_id,start_slot,end_slot)` 一对一核对，校验 target、known、H 与总体 label hash。无需训练或重新解码。逐窗口文件由 [stability.py](D:/DriveGuard-AI/training/visual/stability.py:130) 和 [stability.py](D:/DriveGuard-AI/training/visual/stability.py:143) 已定义输出；本地附件只有汇总，当前尚不能做新的完整阈值扫描。

保留 A 的固定 logit0 决策时，H 可用组已有 TP86/FP26；全局上限 `floor(0.03*1138)=34`，所以 C 在 H 不可用组最多只能贡献8个FP。扫描 C 分数时保持 tied scores 整组纳入，不能把每个分支各自3%阈值拼起来，也不能假设 A/C logits 已校准到同一尺度。先固定 A 阈值，仅扫描 C 的一个阈值；不要同时搜索两分支阈值与路由条件。

附件可重建的**诊断拼接**：沿用原 C 各 seed 低 FPR 阈值时，总 TP/FP 为 seed42=115/38、seed43=103/32、seed44=104/44。只有43满足预算，而且少于全 C43 的113TP。这些数值由互斥 H 分组求和得到，不是新模型实测；它们证明路由不是保证增益的技巧。新的 C 阈值需逐窗口文件才能计算。

**通过条件：** 在预先冻结的外部评估阈值上，全局预算满足，H 可用决策完全等于 A，并在至少两颗 seed 中超过同预算 A 的 TP；与全 C 的得失同时报告。**停止条件：** 剩余8个FP无法换来稳定新增TP，或收益只来自一个 subject。若失败，保留全 C 基线，不升级为可学习门控；门控会增加选择自由度。

### 实验二：相同八帧、相同模型容量，仅改变视觉 crop

**假设：** 整幅ROI缩小损失了可见的头部/面部朝向细节，局部保真可能帮助离路判断。先按 road/nonroad 真值在 train 的 FN 与高分负例中抽样，区分明显转头、头姿近似但注视不同、遮挡/出框、侧视角眼睛不可见等情况，并核查末端标签对齐；同时记录手机/饮水等上下文与正负标签是否共存，防止把相关动作当作离路真值。不用 test 决定 crop。

候选为保留完整头部及可见面部的局部框，或覆盖手/物体的上下文框；前者直接检验朝向保真，后者仅检验上下文相关性。先根据 train 审计固定一个相对 driver_roi 的裁剪框，优先选择头部候选；若审计不支持任何候选则不训练。比较原 whole ROI 与该单 crop：均用同一个冻结 ResNet18、224×224、512维、原八帧、C width64、相同训练设置和42/43/44三颗 seed。第一轮不同时搜索多个框、不拼接双 crop，以免增加选择自由度或参数量；若要进一步比较两个候选，另登记一轮同容量对照，不混入本轮。

这不是现有 CLI 的一个现成参数：ROI 在 [extract.py](D:/DriveGuard-AI/training/visual/extract.py:203) 直接传入预处理，cache 的维度、预处理 metadata 和 source ROI 验证严格绑定。应新增明确的 crop 元数据/版本及新 cache、prepared、输出目录，保留原 manifest、标签、split、PTS、权重散列；不能只改 JSON 指向或覆写原 ROI。新视觉归一化仍只拟合 train；冻结 A context 可经身份校验复用。

新增训练预算3次；原 whole ROI 三颗 seed 作为已完成对照，运行条件变化时需重跑匹配对照。**通过条件：** 同预算召回提升跨 seed 且两个 subject 方向一致，H 可用损失和 H 不可用误报不恶化到抵消收益；手机/饮水但仍 looking_road 的窗口不能因动作相关性新增集中误报。**停止条件：** train 抽样不支持局部信息丢失假设；头部常在裁剪外；主要漏检属于侧视角不可见的眼动；收益只在一个 subject 或某种操作动作；crop 改善默认阈值但低 FPR 没改善。此时不继续堆多个裁剪。

### 实验三：保持八帧/512维/头容量，比较最近两秒的更密采样

**假设：** 末端附近的 road/nonroad 切换与短暂离路注视比更久历史更重要。对比原四秒窗口八帧与 offset `[56,62,68,74,81,87,93,99]` 的最近约两秒八帧。后者仍保留同一个末帧，模型输入维度相同；这是“近期密采样 vs 长历史稀采样”的策略比较，不能单独归因为频率变化，因为有效历史长度也改变了。

只在实验二未执行或已经结束后，使用固定的 whole ROI 作为本轮输入，避免与 crop 混在一起。原有新鲜度≤80ms与只能选当前或过去PTS的原则保留，见 [extract.py](D:/DriveGuard-AI/training/visual/extract.py:195)。每秒 slot 余数 `[6,12,18,24]` 可提供新采样；原 cache 没有6和18位置，不能靠八帧复制伪造密采样。

需要新采样 cache 版本：当前 [encoder.py](D:/DriveGuard-AI/training/visual/encoder.py:29)、[visual/data.py](D:/DriveGuard-AI/training/visual/data.py:117)、[visual/data.py](D:/DriveGuard-AI/training/visual/data.py:212)、[prepare.py](D:/DriveGuard-AI/training/visual/prepare.py:90) 都硬编码了旧 grid 或窗口索引；必须一起实现并校验，不能仅改 OFFSETS。标签仍来自原100-slot窗口末端，窗口ID和评估母集不变。仅新鲜度不同的窗口还需报告共同可用子集，避免覆盖率混淆。

新增训练预算3次，同样42/43/44。**通过条件：** 同预算TP增益在短暂离路注视/切换附近出现，且持续离路区间没有同量损失；报告离路标签起点后的检测延迟变化。**停止条件：** 增益只来自可用样本集合变化，或持续离路所需历史丢失抵消收益。稠密16帧、光流与新视频骨干暂不加入本轮。

## 4. 为什么暂不把视觉微调放在前三步

现在 `FrozenEncoder` 显式 `requires_grad_(False)`、`eval()` 并在 `inference_mode()` 下提取特征，见 [encoder.py](D:/DriveGuard-AI/training/visual/encoder.py:111)。训练器只接收缓存的512维特征，[train.py](D:/DriveGuard-AI/training/visual/train.py:70) 的梯度不可能返回图像骨干；多跑几个epoch或改probe学习率不等于视觉微调。PyTorch 官方也区分“冻结特征提取器”和“更新预训练网络”的迁移路径，见 [官方迁移学习教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)。

若 crop/时间实验支持视觉信息不足且跨 subject 稳定，再考虑单独的 layer4 微调实验：早层冻结、BatchNorm统计固定、train 图像训练、重新导出每个微调模型对应的全split cache，沿用严格身份和归一化隔离。这个配置是待验证设计，不是已有支持的命令；需要新的图像训练数据路径、显存与NPU算子兼容验证。只有两个验证subject且best常在1–3epoch，现阶段直接全骨干训练难以区分适配收益与过拟合。不能承诺提升，也不在本轮启动。

## 5. 可信增益前必须守住的评估边界

- **已存在的防泄漏机制：** [common/data.py](D:/DriveGuard-AI/training/common/data.py:125) 拒绝同subject跨split、recording不一致以及重复recording/view；[prepare.py](D:/DriveGuard-AI/training/visual/prepare.py:59) 检查源manifest与baseline数据fingerprint；[prepare.py](D:/DriveGuard-AI/training/visual/prepare.py:101) 只用train拟合视觉/context归一化。原始主体标识若被误标、同视频以不同身份重复，这些元数据校验仍不能证明绝无泄漏；正式评估应审计源视频SHA/主体映射跨split交集。
- **重叠窗口不能随机分组做校准。** 100-slot窗、25-slot步长使相邻窗共享75%的slot。原训练代码按完整record归split，没有发现把重叠窗随机分到train/val的实现。后续阈值校准也按subject或完整record划分；同subject不同record仍不能代表未见subject泛化。只报告两人的seed标准差不是统计置信区间。
- **避免在test重新挑threshold。** [report.py](D:/DriveGuard-AI/training/visual/report.py:38) 的 `operating_point` 使用传入行的标签挑阈值；[report.py](D:/DriveGuard-AI/training/visual/report.py:83) 的 `summarize` 会自动调用它。现有suite只传val；未来test评估不能直接调用同一个summarize后把结果称作锁定阈值test。应先在开发/校准集锁定模型、seed选择规则和阈值，再仅调用固定阈值metrics评估test。test结果可能超过3%，必须如实报告而不能再调低阈值。
- **选epoch与选工作点需分开解释。** [train.py](D:/DriveGuard-AI/training/visual/train.py:239) 以val BCE选best；[report.py](D:/DriveGuard-AI/training/visual/report.py:38) 再以同val标签找3%点。若继续用同val探索三种策略，只能称开发诊断。最好预先留出独立subject作最终评估；只有两位val时，交换两人作校准/检查仅是敏感性分析，不是新增外部证据。
- **上下文堆叠的训练分布问题不是已证实的跨split泄漏。** [prepare.py](D:/DriveGuard-AI/training/visual/prepare.py:72) 用同一个已训练A生成train/val/test context；train context可能来自A见过标签的样本，val context来自未见样本。可怀疑上层过度依赖训练期A输出，但不能据此宣称val泄漏。若最终要做按subject交叉验证，A、normalizer、C及阈值都必须只在各自训练折生成/拟合；仅把目前prepared context重新分折无法提供端到端无泄漏的OOF结论。
- **本轮禁止追着误报列表无限试参数。** 三个实验及其停止条件先登记；新模型和阈值均落盘保存hash、数据identity与选取理由。真实改进应同时在固定母集、可用率、H组和subject组上解释得通，再进入独立评估。

## 建议执行顺序

先完成主线全任务评估；随后在服务器现有逐窗口CSV上做实验一（0次训练），结果不支持则停止路由。再用train误例证据在实验二、三中选一个先实施，每个新增3次匹配seed训练，最多再推进另一个。微调、新骨干、更复杂门控与loss大范围搜索留待上述受控实验回答具体瓶颈后决定。
