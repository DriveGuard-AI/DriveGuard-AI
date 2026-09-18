# DriveGuard AI 测试策略

## 1. 目标

测试用于证明需求、模型行为、数据契约和运行性能达到可接受水平，而不只验证代码可以启动。当前无实现代码，本文件定义后续测试基线和证据格式。

## 2. 测试层次

| 层次 | 对象 | 重点 | 建议工具 |
| --- | --- | --- | --- |
| 单元 | 特征、窗口、融合、状态机、校验器 | 边界值、状态转换、缺失输入 | pytest |
| 契约 | Pydantic/OpenAPI/TS 类型/WS schema | 字段、枚举、兼容性 | FastAPI TestClient、schema validator |
| 集成 | 推理管线、SQLite Repository、API、WS | 模块交互、事务、错误传播 | pytest、临时 SQLite |
| 模型 | Detector、Landmarker、TCN、Fusion | 指标、泛化、延迟、可靠性 | PyTorch、评估脚本 |
| E2E | 录像/摄像头到告警、事件与 Dashboard | 完整行为和降级 | 固定回放集、浏览器测试 |
| 性能 | GPU/Jetson 推理与 API | FPS、延迟、内存、稳定运行 | benchmark、profiler |
| 安全隐私 | 鉴权、输入、路径、日志、存储 | 越权、注入、泄露、默认隐私 | API 安全测试、静态扫描 |

## 3. 数据集与模型评估

### 3.1 数据切分

- DMD 必须按 `driver_subject_id` 划分训练、验证、测试集。
- 三个集合的主体交集必须为空；摄像头视角与同一段视频衍生帧不得跨集合泄漏。
- 切分清单应版本化，并记录数据集版本、随机种子和过滤规则。

### 3.2 指标

| 指标 | 用途 |
| --- | --- |
| Precision / Recall / F1 | 分类质量，优先关注危险行为 Recall |
| False Alarm / Hour | 实际驾驶体验与告警疲劳 |
| Detection Delay | 行为出现到首次有效告警的延迟 |
| FPS / P50/P95 latency | 实时性和尾延迟 |
| Calibration error | 概率是否可用于风险融合 |
| Missing/occlusion slices | 低光、遮挡、眼镜、姿态变化鲁棒性 |

报告必须注明硬件、输入分辨率、窗口长度、模型版本、运行时和精度模式；不同条件的数字不得直接横向比较。

## 4. 核心测试场景

### 4.1 正常与边界

- 空画面、无人脸、单帧、窗口未填满、时间戳倒退和重复帧。
- 正常眨眼、持续闭眼、连续打哈欠、短暂低头和持续低头。
- 手机只在背景出现、手机靠近驾驶员但未交互、持续手机交互。
- 多风险同时出现、单专家缺失、所有关键专家不可用。
- 风险在阈值上下抖动、升级、恢复、冷却和重复告警抑制。

### 4.2 数据库

- 空库、单条事件、10 万条事件分页、时间范围边界和稳定排序。
- 活动事件创建/更新/结束、服务重启后活动事件恢复。
- 配置乐观锁成功与冲突、非法 JSON、事务回滚。
- 数据库锁定或写入失败时，本地告警仍继续且错误可观测。

### 4.3 REST/WebSocket

- 每个端点至少覆盖成功、鉴权失败、校验失败和资源不存在/冲突中适用的场景。
- `page_size=0/1/100/101`、反向时间范围、非法 UUID 和未知枚举。
- WebSocket 初始完整快照、序列单调性、心跳、慢客户端、断线重连和 Token 失效。
- FastAPI 生成的 OpenAPI 与 `docs/api/openapi.yaml` 进行结构差异检查。

### 4.4 隐私和安全

- 默认配置运行后，数据库、日志和文件系统中不存在原始视频片段。
- 路径穿越、超长输入、未知配置键和越权配置修改被拒绝。
- 错误响应不包含堆栈、Token、绝对模型路径或数据库路径。

## 5. E2E 验收集

| E2E-ID | 场景 | 预期 |
| --- | --- | --- |
| E2E-001 | 正常驾驶录像连续播放 | 风险保持 SAFE，无持续误报警 |
| E2E-002 | 正常眨眼片段 | 不创建长闭眼危险事件 |
| E2E-003 | 持续闭眼片段 | 在定义延迟内 WARNING/DANGER，事件字段完整 |
| E2E-004 | 持续手机交互片段 | phone_use 概率和风险升级符合已校准规则 |
| E2E-005 | 面部遮挡 | quality 变为 degraded/unknown，不把缺失解释为 SAFE |
| E2E-006 | 运行中断网 | 本地推理与告警继续，Dashboard 显示断连并可恢复 |
| E2E-007 | SQLite 暂时不可写 | 告警继续，产生数据库错误指标和有界恢复行为 |
| E2E-008 | Dashboard 慢消费 | 推理不被阻塞，恢复后拿到最新完整快照 |

## 6. 性能测试

### 6.1 基准矩阵

- 设备：开发 RTX GPU、目标 Jetson Orin 型号。
- 运行时：PyTorch、ONNX Runtime、TensorRT FP16。
- 输入：至少两种目标分辨率和目标摄像头帧率。
- 工况：稳定画面、多人/遮挡、持续一小时和资源压力。

### 6.2 最小报告字段

```text
commit, model_bundle_version, device, power_mode, runtime, precision,
resolution, input_fps, processed_fps, latency_p50_ms, latency_p95_ms,
gpu_memory_mb, cpu_percent, dropped_frame_rate, duration_minutes
```

25 FPS 是项目目标；如果未达到，应报告瓶颈与实测差距，不得把短时峰值作为持续性能。

## 7. 测试与需求追溯

| Spec | 主要测试 |
| --- | --- |
| REQ-001 | E2E-001、E2E-006、性能矩阵 |
| REQ-002 | 特征单元测试、E2E-005 |
| REQ-003 | E2E-002/003/004、窗口边界测试 |
| REQ-004 | 融合表驱动测试、多风险和缺失专家测试 |
| REQ-005 | Alarm Controller 状态机测试 |
| REQ-006 | Event Service/Repository 集成、隐私检查 |
| REQ-007 | REST/WS 契约与 E2E-008 |
| REQ-008 | 配置校验、权限与并发冲突测试 |
| REQ-009 | 主体泄漏审计、模型指标和工件校验 |
| REQ-010 | 文档影响检查与 OpenAPI diff |

## 8. CI 阶段建议

1. 快速检查：格式、lint、类型、单元测试、OpenAPI/YAML 解析。
2. 集成检查：SQLite migration、Repository、API 与 WebSocket 测试。
3. 模型检查：固定小样本回归，不在每次提交运行完整训练。
4. Nightly/候选发布：完整回放集、性能、长稳、隐私和目标设备测试。

## 9. 通过标准与证据

训练工程新增检查（REQ-011，负责人 shadow）：配置类型与范围、外置路径、训练/验证/测试主体交集、设备不可用时显式失败、checkpoint CPU 可加载、resume 与训练配置一致。双入口共享测试；真实 GPU 与昇腾 NPU 运行各留设备和环境记录。合成 smoke 只能证明代码闭环，不能作为模型正式跑分。预算实验顺序见[训练职责约定](../training/TRAINING_CHARTER.md)。

- 新增或修改 Requirement 时，至少有一个可执行或明确待实现的测试映射。
- 代码合并前相关自动化测试通过；无法自动化的验收需附复现步骤、环境和结果。
- 模型或阈值变更必须附相同数据切分和硬件口径下的前后对比。
- 已知失败不得隐藏，应记录影响、临时处理和后续验证入口。

## 资料来源

- 2026-09-10 项目初始规划基线：评估、数据集、部署和隐私要求。
- `docs/specs/PROJECT_SPEC.md`：全部 Requirement 与 Scenario。
- `docs/api/API_DESIGN.md`、`docs/database/DATABASE_DESIGN.md`：接口和持久化边界。
