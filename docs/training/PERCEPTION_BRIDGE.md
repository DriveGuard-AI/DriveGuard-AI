# 感知观测 → TCN 特征桥接

2026-09-19 · shadow · model 分支。已实现：严格观测协议、几何特征、因果统计、流式数组导出及 TCN 接线。上游新增了[YOLOX/MediaPipe 本地视频运行时](EXPERT_RUNTIME.md)、PTS重采样及保守跟踪；已做合成边界测试及[真实模型CPU加载与空白帧前向](PERCEPTION_SETUP.md)，**真实视频尚未验证，自动归属、头姿和检测器训练仍未实现**。本模块仍只接收规范观测，不直接加载模型。

## 1. 配置与特征

- [感知 profile](../../configs/perception/interaction_v1.json)：`interaction-candidate-v1`，特征版本 `driveguard-interaction-v1`。
- [TCN 配置](../../configs/training/temporal_interaction.json)：19 特征 + 19 有效性掩码，输入 `[B,100,38]`，四类输出不变。
- 原 [24 通道基线](../../configs/training/temporal_baseline.json)保留。新配置不是已训练权重，不能加载旧 24 通道 checkpoint。

| 特征组 | 有序名称 | 单位/语义 |
| --- | --- | --- |
| 眼/嘴 | ear_left, ear_right, mar | 原图像素几何比值 |
| 疲劳统计 | perclos_4s, closed_duration_s | 已知闭眼槽比例、连续观测闭眼秒数 |
| 上游头姿 | head_yaw_deg, head_pitch_deg, head_roll_deg | 度；缺少头姿适配器则 mask=0，不由 gaze 标签补齐 |
| 物体 | phone_conf, drink_object_conf | ROI 内候选检测分数；不是行为概率 |
| 面部邻近 | phone_face_distance, drink_mouth_distance | 到面部框中心/嘴部中心距离，除以面部框对角线 |
| 眨眼 | blink_rate_per_min | 完整有效 4 秒窗口内眨眼次数 ×15，短窗估计并非临床指标 |
| 手—手机 | phone_hand_distance, phone_hand_iou, phone_near_duration_s | 手物框最短间隙/ROI 对角线、IoU、连续观测近邻秒数 |
| 手—饮水物体 | drink_hand_distance, drink_hand_iou, drink_near_duration_s | 同上，候选为 bottle/cup；不证明饮水或物理接触 |

名称顺序以 JSON 为唯一输入顺序。相较旧基线，新 profile 将物体筛选限定为框中心位于驾驶员 ROI；因此不仅增加了维度，也变更了观测定义。缓存、normalizer、输入层与权重必须随新 profile 配套。

## 2. 上游 packet 契约

每条 JSONL 是一个 **25 Hz 采样槽** 的 `driveguard-experts-v1` 对象，不是任选一帧。所有层级拒绝未知键、重复 JSON 键、NaN/Infinity、bool 冒充数值。根对象必须恰好含：

| 字段 | 格式 |
| --- | --- |
| schema_version | driveguard-experts-v1 |
| recording_id / view / frame_id | 非空字符串；frame_id 不得与上一槽重复 |
| timestamp_ms | 同一录制时钟的非负整数毫秒；后续槽必须精确 +40；范围≤2^63−1 |
| width / height | 原图正整数尺寸，≤2^31−1 |
| driver_roi | `[x1,y1,x2,y2]`，原图像素，宽高至少1e−6、图内固定区域 |
| face / objects / hands | 下述独立专家对象 |

每个专家含 `status`（ok/missing/error）、`timestamp_ms`（该结果对应的采集时间，不是计算完成时间）、`model_id`（非空且可追溯的版本标识），再加下列全部字段：

| 专家 | 额外字段 |
| --- | --- |
| face | driver_confirmed 布尔；landmarks=null 或 468/478 个 `[x,y]`；head_pose=null 或 `{yaw_deg,pitch_deg,roll_deg}` |
| objects | items 数组；每项恰好 `{category,box,score,track_id}` |
| hands | items 数组；每项恰好 `{landmarks,driver_confirmed,track_id}`，landmarks 为21个 `[x,y]` |

所有点和框必须在原图内，x∈[0,width]、y∈[0,height]。category 仅 phone/bottle/cup，score∈[0,1]；box 是像素 xyxy，宽高至少1e−6。track_id 为非空字符串或 null；同槽物体非空 track_id 不得重复。它必须是稳定的对象身份，不能把类别 ID 或每帧数组索引冒充跟踪 ID。

失败专家 payload 必须为空：face 的 landmarks/head_pose=null、driver_confirmed=false；objects/hands 的 items=[]。成功但无人脸也可 landmarks=null，此时 head_pose 必须 null。专家过期的 payload 仍先校验，不以过期为理由接受脏数据。

`driver_confirmed=true` 必须来自外部明确的归属/标定证据，本模块不自动把最大脸或 ROI 内任意手判为驾驶员。面部/手部还要求所有点位于 ROI 内；这是保守有效性检查，不是身份证明。未知归属的手不产生“没有交互”的结论。

### 坐标和上游适配责任

只接受未镜像原图像素 xy。MediaPipe 的归一化 xy 应先分别乘原图 width/height，并恢复上游裁剪、缩放、镜像；直接在归一化 xy 上算纵横比会扭曲非正方形图像。[MediaPipe 坐标说明](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)

YOLO 输出也必须恢复到同一原图 xyxy，按锁定的类别名映射成 phone/bottle/cup；不要硬编码未经确认的 class_id。当前适配选用 [YOLOX 0.3.0](https://github.com/Megvii-BaseDetection/YOLOX/tree/0.3.0) Nano/Tiny，启动时核对 COCO80 类名/索引，再进行坐标恢复；不声称其 NPU 训练已兼容。

头姿约定：相机右手坐标 x 向右、y 向下、z 向前，`R=Rz(roll)Ry(yaw)Rx(pitch)`，角度均在[-180,180]。只有上游按此约定转换后的结果可传入；本桥接不从 MediaPipe 变换矩阵直接猜角度。头姿不等于 gaze。

实际专家 profile 应记录工件来源、版本、SHA-256 与调用配置；每次导出会保存三个 model_id，但仅凭任意字符串无法证明工件身份。合并训练 manifest 时应把经过核对的模型组合信息一并放入 extractor_profile，不能把不同来源的缓存混用。

## 3. 几何与时序口径

EAR 左眼索引 `[362,385,387,263,373,380]`，右眼 `[33,160,158,133,153,144]`，沿用 MediaPipe 的解剖左右；公式为 `(d(p1,p5)+d(p2,p4))/(2*d(p0,p3))`。MAR 为 `d(13,14)/d(78,308)`。索引与拓扑依据见[官方连接定义](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py)；具体比值/阈值是本项目候选算法，不是官方疲劳判据。

长度小于1e−6像素的眼宽/嘴宽/面部或手部框按几何退化处理，相应特征无效。框中心为 xyxy 中心，嘴部中心为13/14点中点。头姿是独立的上游结果，确认且新鲜的面部即使几何框退化，也不自动抹掉已给定的头姿。

物体须满足 score≥0.25、框中心位于 ROI；按分数降序、框坐标、类别、track_id 排序。phone 单独选一个最高候选，bottle/cup 合并选一个；分数、到脸/嘴距离及手物关联始终使用同一候选。手物关联选框间隙最小的已确认驾驶员手，IoU 也取同一对。ROI 外高分背景物体被排除；ROI 内背景物体仍可能入选，不能据此宣称已消除全部背景误检。

默认有效年龄≤80ms，未来观测一律拒绝；桥接不自行查找最近帧或插值。降频专家由上游填入最近过去结果和原采集时间；超过年龄限制即未知。

- 双眼 EAR 均有效且平均<0.2 才为闭眼候选，任一无效则闭眼状态未知。
- PERCLOS 满100槽后，已知眼部槽至少80个才有效；分母只用已知槽，缺失不当作睁眼。
- 闭眼时长必须观察到 open→closed 起点；从闭眼开始的录制、或缺失后仍闭眼，时长未知，直到再观察到睁眼后重新开始。
- 眨眼只计完整 open→closed→open，闭眼80–500ms；包括确认起点的睁眼槽在内，所有证据都须位于当前100槽窗口，且窗口眼部信息全部有效。未完成/跨缺失闭眼不计数。
- 近邻阈值为框间隙/ROI对角线≤0.03。手缺失、检测器失效或未跟踪的近邻不能累计时长；先观察到确定远离，再到近邻才可获得已知起点。持续近邻中切换对象 track_id，从当前槽0秒重新计数，不拼接两个对象；这只是连续**观测**的时长。
- 录制、视角、尺寸、ROI、任一专家 model_id 改变必须新建提取器。时间跳槽/重复/倒退拒绝，掉帧必须显式保留 missing 槽。非法 packet 和数值转换失败不改变历史。

上述阈值均未在 DMD 或目标硬件上标定，不作为行为标签或最终告警规则。

## 4. Python 与 CLI 使用

```python
from inference.perception.profile import load_profile
from inference.perception.sequence import FeatureExtractor

extractor = FeatureExtractor(load_profile())  # 每录制/单视角一个实例
result = extractor.step(packet)              # packet 须来自符合第2节协议的上游
# result['values']: float32[19]，未归一化；result['masks']: uint8[19]
```

从仓库根目录、已有环境执行，示例路径须换为实际观测 JSONL：

```bash
python -m inference.perception.export --input data/observations/recording.jsonl --output-dir data/features/recording-001
python -m training.temporal.npu_train --config configs/training/temporal_interaction.json --check-config
```

CLI 不读取 DMD 视频/压缩包，不启动 YOLO 或 MediaPipe。每行最多1,000,000个字符，空行拒绝；每条录制至少100槽。输出目录必须不存在，不清空已有结果。

导出 values/masks/targets/label_masks.npy，及最终 `record.json`：含 profile/hash、专家版本、图像尺寸/ROI、时间范围、数组路径与哈希。`targets` 与 `label_masks` **全部为0（未知）**；record.json 不是完整训练 manifest，不能直接交给 --manifest。

导出先流式写临时特征行，后分块建立 `.npy` mmap，最后发布成功索引。常驻内存不随录制长度线性增长；临时行与最终数组短暂共存，占用约每槽267字节（最终115+临时152，不含头和元数据）。25 Hz 下每小时峰值约24 MB，不包含输入 JSONL 和原视频；不是测得的峰值内存。

中途失败保留新目录用于排查，不自动删除；无 `record.json` 就不能当作成功缓存，`incomplete.json` 仅是辅助提示。程序只清理自己创建的临时行文件；文件锁/权限失败可能留下它。最终索引替换不构成抗断电的文件系统事务，完整性还应核对数组 hash。

正式训练前：核对上游真实模型与采样 → 补充经过审核的正/负/未知标签 → 统一主体/视角切分 → 生成[训练 manifest](DATA_CONTRACT.md) → 使用新 TCN 配置。没有标签时现有训练器会拒绝启用任务，不能通过把未知改负例来绕过。

## 5. 验证与剩余工作

```bash
python -m unittest discover -s tests -p "test_perception_*.py" -v
python -m unittest discover -s tests -v
```

测试只使用合成关键点/检测框；包括几何、过期/未来、归属/背景反例、缺失/眨眼/换对象、流式导出、失败发布保护以及 FeatureCorpus→38通道TCN 前向。

剩余工作：准备并核对实际工件许可，用真实视频验证已接入的 Face/Hand/YOLOX 运行时、坐标/跟踪/重采样；完善自动驾驶员归属和头姿；再恢复 DMD 标签核对，分别验证 GPU/NPU。当前未生成正式模型权重或真实跑分。
