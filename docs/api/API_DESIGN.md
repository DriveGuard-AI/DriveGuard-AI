# DriveGuard AI API 设计

## 1. 概述

MVP 使用 RESTful JSON API 处理查询和配置，使用 WebSocket 推送实时风险快照；可选实时视频使用独立 MJPEG HTTP 流。基础路径为 `/api/v1`，WebSocket 路径为 `/ws/v1`。本文为 0.2.0 设计契约，机器可读草案见 `openapi.yaml`，尚未实现。后端与前端选型见[应用层设计](../architecture/APPLICATION_DESIGN.md)。

## 2. 通用规范

| 项目 | 规则 |
| --- | --- |
| 数据格式 | JSON，字段使用 `snake_case` |
| 时间 | UTC ISO 8601，例如 `2026-09-10T08:30:00.123Z` |
| 标识 | UUID 字符串 |
| 版本 | REST 使用 URL 主版本；WS 消息包含 `schema_version` |
| 认证 | `/health` 可匿名；其余接口在生产部署使用 Bearer Token |
| 请求关联 | 客户端可发送 `X-Request-ID`，服务端响应回传或生成 |
| 分页 | `page` 默认 1，`page_size` 默认 20、最大 100 |
| 限流 | 读接口默认每 Token 120 次/分钟；写接口 30 次/分钟；本地可信部署可配置 |
| 错误 | 统一 `code`、`message`、`details`、`request_id` |

成功响应直接返回资源或分页对象；错误响应：

```json
{
  "code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "details": [{"field": "page_size", "reason": "must be between 1 and 100"}],
  "request_id": "5f59693d-5c67-44e8-8fd8-f36d06bc41e2"
}
```

## 3. 资源模型

### 3.1 Event

`eyes_off_road` 是 Phase 2 预留事件类型；MVP 在对应模型和 Spec 场景启用前不得生成该类型。

```json
{
  "event_id": "35fd8f0e-611e-42b1-b46c-a99456fc44e8",
  "session_id": "9e809f40-1fe3-480b-968a-68257b80969e",
  "event_type": "fatigue",
  "started_at": "2026-09-10T08:30:00.123Z",
  "ended_at": "2026-09-10T08:30:04.623Z",
  "duration_ms": 4500,
  "max_confidence": 0.91,
  "max_risk_level": "DANGER",
  "status": "ended",
  "quality": "good",
  "risk_scores": {"fatigue": 0.91, "distraction": 0.34},
  "reason_codes": ["EYES_CLOSED_SUSTAINED"],
  "model_version_id": "5bd97315-925c-46c1-9e12-9c72027c4d12",
  "clip_available": false
}
```

### 3.2 RiskSnapshot

只有证据足以形成风险判断时才生成本对象。窗口预热、全部关键观测无效或状态过期时不发送伪造 SAFE；REST 返回 503，WS 发送 system.status。不可用任务从 risk_scores 省略，不填 0；部分任务缺失只在剩余证据足以支持 WARNING/DANGER 时发布 degraded 快照。SAFE 需要完整已验证任务与覆盖条件满足。客户端通过 system.status 和 captured_at 使旧快照失效，旧风险不是当前状态。

features 的缺失项用 null 或省略；confidence 是融合配置定义的决策支持分数，未做校准前不得解释为事故概率。detections 的 bbox 约定为原始未镜像图像的归一化 `[x_min,y_min,x_max,y_max]`，各值 0–1 且 x_min<x_max、y_min<y_max；跨预览缩放时由前端映射。

`task_details` 是逐任务判决参数（六参数 + 派生 `margin`），字段与 live 服务 `tasks[name]` 一一对应：

| 字段 | 含义 |
| --- | --- |
| `available` | 该任务本帧是否有可用证据；为 false 时 `logit/score/detected/margin` 必须为 null |
| `logit` | 模型原始输出（未过 sigmoid） |
| `threshold_logit` | 该任务的判定阈值 |
| `margin` | `logit - threshold_logit`，由接口层派生：>=0 已触发，<0 是距触发的差距 |
| `score` | 未校准 sigmoid（`score_semantics=uncalibrated_sigmoid`），仅供相对比较 |
| `detected` | 服务端判定结果（`logit >= threshold_logit`） |
| `source` | 证据来源：`coverage25` / `C_fusion` / `gap80_rgb_fallback` / `yawn_eye_dense_track` |

判定依据是 logit 与阈值，不是 score；`margin` 用于回答"离触发还有多远"。不可用任务不得补 0，客户端也不得自行插值。结果过期（`RESULT_STALE`）或回放模式时整表为空。

```json
{
  "schema_version": "1.0",
  "session_id": "9e809f40-1fe3-480b-968a-68257b80969e",
  "sequence": 18342,
  "captured_at": "2026-09-10T08:30:04.623Z",
  "risk_level": "WARNING",
  "primary_risk": "fatigue",
  "confidence": 0.84,
  "quality": "good",
  "risk_scores": {
    "fatigue": 0.84,
    "phone_use": 0.12,
    "drinking": 0.03,
    "distraction": 0.67
  },
  "features": {
    "ear": 0.18,
    "mar": 0.31,
    "perclos": 0.42,
    "head_pose": {"yaw": 4.2, "pitch": 11.8, "roll": 1.1}
  },
  "detections": [],
  "task_details": {
    "fatigue": {
      "available": true,
      "logit": 1.42,
      "score": 0.81,
      "threshold_logit": 0.0,
      "margin": 1.42,
      "detected": true,
      "source": "yawn_eye_dense_track",
      "score_semantics": "uncalibrated_sigmoid"
    },
    "drinking": {
      "available": false,
      "logit": null,
      "score": null,
      "threshold_logit": 0.0,
      "margin": null,
      "detected": null,
      "source": "coverage25",
      "score_semantics": "uncalibrated_sigmoid"
    }
  },
  "reason_codes": ["PERCLOS_HIGH"]
}
```

### 3.3 AnalysisJob

一次离线分析任务的指针与进度。素材与抽帧产物不在数据库里，只存路径。

```json
{
  "job_id": "8f0f6f0a-1f1b-4b8c-9a2e-8a0c1e5f0d21",
  "session_id": "5c1b1d3a-4a41-4f1e-9d5b-0b9c0a6f2e77",
  "source_kind": "frames",
  "source_path": "D:/driveguard/var/media/analysis/8f0f.../frames",
  "source_sha256": null,
  "status": "running",
  "frames_total": 961,
  "frames_done": 320,
  "step_ms": 40,
  "timeline_ms": 12800,
  "error": null,
  "source_available": false,
  "created_at": "2026-10-19T09:00:00Z",
  "started_at": "2026-10-19T09:00:00Z",
  "ended_at": null,
  "updated_at": "2026-10-19T09:04:12Z"
}
```

- `status`：`queued | running | completed | failed | cancelled`；`frames_done` 单调不减。
- `step_ms`：抽帧步长；事件时间轴 ÷ 步长 = 帧号，回放定位靠它。
- `source_available`：原始视频是否还在。true 时前端用 `<video>` 回放，false 时逐帧取图。
- `timeline_ms`：已处理到的素材时间轴位置，用于判断进度是否还在推进。
- 每个任务对应一个新的 `monitoring_sessions` 记录，事件按该会话隔离。

## 4. REST 端点

### API-001 健康检查

`GET /api/v1/health`

- 鉴权：无。
- 成功：`200`，返回服务进程存活状态和版本。
- 说明：只表示 API 存活，不代表摄像头或模型健康。

```json
{"status": "ok", "service_version": "0.1.0", "time": "2026-09-10T08:30:00Z"}
```

### API-002 系统状态

`GET /api/v1/system/status`

- 鉴权：监控读取权限。
- 成功：`200`，包含摄像头、推理、数据库、模型和实时性能状态。
- 错误：`401 AUTH_REQUIRED`、`503 PIPELINE_UNAVAILABLE`（仅在无法形成有效状态响应时）。
- `pipeline.playback=true` 表示推理源已被离线分析任务换成一段素材（不是实时采集），
  页面必须据此提示“离线回放”，不得把回放当成实时监控。

### API-003 事件列表

`GET /api/v1/events`

查询参数：

| 参数 | 类型 | 必填 | 规则 |
| --- | --- | --- | --- |
| `page` | integer | 否 | `>=1`，默认 1 |
| `page_size` | integer | 否 | `1..100`，默认 20 |
| `event_type` | string | 否 | 事件枚举 |
| `risk_level` | string | 否 | `SAFE/WARNING/DANGER` |
| `status` | string | 否 | `active/ended/discarded` |
| `started_from` | datetime | 否 | UTC，含边界 |
| `started_to` | datetime | 否 | UTC，含边界且不早于 from |
| `session_id` | UUID | 否 | 监测会话 |

成功响应：

```json
{
  "items": [],
  "page": 1,
  "page_size": 20,
  "total": 0,
  "has_next": false
}
```

排序固定为 `started_at DESC, event_id DESC`。错误：`400 INVALID_TIME_RANGE`、`422 VALIDATION_ERROR`。

### API-004 事件详情

`GET /api/v1/events/{event_id}`

- 成功：`200 Event`。
- 错误：`404 EVENT_NOT_FOUND`、`401 AUTH_REQUIRED`。
- 幂等：是。

### API-005 统计摘要

`GET /api/v1/statistics/summary?started_from=2026-09-01T00:00:00Z&started_to=2026-09-10T23:59:59Z&session_id=<uuid>`

- 查询参数：`started_from` / `started_to` 必填；`session_id` 可选，用于只看某个监测会话
  （每跑一次离线分析都会产生新会话，不按会话限定会把不同素材混在一起）。
- 成功：总事件数、各类型数量/持续时间、各风险等级数量和按时间桶趋势；并回显 `session_id`。
- 无数据：返回数值 0 和空数组，状态仍为 `200`。
- 时间跨度过大：`422 TIME_RANGE_TOO_LARGE`，MVP 默认最大 90 天，可配置。
- 非法 `session_id`：`422 VALIDATION_ERROR`。

### API-006 读取配置

`GET /api/v1/settings`

- 鉴权：管理员。
- Secret：只返回是否已配置，不返回原值。
- 成功：配置数组、全局 `version` 和每项生效状态。

### API-007 更新配置

`PUT /api/v1/settings`

```json
{
  "version": 3,
  "changes": {
    "alarm.cooldown_ms": 5000,
    "temporal.window_seconds": 4.0
  }
}
```

- 鉴权：管理员。
- 幂等：相同 `version + changes` 在首次成功后重试，返回当前一致结果；不同旧版本返回冲突。
- 成功：`200`，返回新版本、应用结果和需重启项。
- 错误：`409 CONFIG_VERSION_CONFLICT`、`422 VALIDATION_ERROR`、`403 FORBIDDEN`。

### API-008 模型列表

`GET /api/v1/models`

- 查询参数：可选 `model_type`、`status`。
- 成功：返回模型名称、语义版本、运行时、特征 schema、指标摘要和激活状态。
- 不返回宿主机绝对路径；仅返回工件逻辑标识。

runtime 枚举为 `pytorch/onnxruntime/tensorrt/mediapipe/python`；mediapipe 用于复用关键点任务，python 用于规则融合工件，不代表两者均为 PyTorch 网络。组合版本通过 API-002 的 model_bundle_version 对应受控 manifest，事件绑定的会话可追溯全部专家，见[数据库设计 §4.4](../database/DATABASE_DESIGN.md#44-model_versions)。

### API-009 当前风险快照

`GET /api/v1/monitoring/snapshot`

- 用途：页面首次加载或 WebSocket 重连后获取完整最新状态。
- 成功：`200 RiskSnapshot`。
- 无活动会话：`409 MONITORING_NOT_ACTIVE`。
- 活动会话存在但尚无有效/新鲜结论：`503 PIPELINE_UNAVAILABLE`；前端展示预热、不可用或降级提示，不以零分代替。

### API-010 实时视频预览（默认关闭）

`GET /api/v1/monitoring/preview`

- 鉴权：monitoring:read；使用 Authorization Header，浏览器以 fetch 流读取，不把 Token 放 URL。
- 开关：管理员通过 API-007 设置 `preview.enabled`（boolean，默认 false）；这是运行时开关，不需要重启。关闭时终止已有预览连接，不影响推理。
- 成功：200，`Content-Type: multipart/x-mixed-replace; boundary=frame`、`Cache-Control: no-store`。每个 part 是 JPEG，流持续到取消/断开；最大输出帧率、分辨率和客户端数由部署配置限制，不承诺等于推理 FPS。
- 错误：401 AUTH_REQUIRED；403 FORBIDDEN（权限不足或预览关闭，details 区分原因）；409 MONITORING_NOT_ACTIVE；503 PIPELINE_UNAVAILABLE（摄像头/编码不可用、画面尚未就绪、离线回放、连接数已满）。校验顺序为鉴权→开关/权限→会话→设备。
- 返回 200 后的故障关闭流，通过 system.status 说明原因，不能再把 JSON 错误混进 JPEG 流；客户端取消时释放连接资源，不关闭共享摄像头。
- 只在内存读取最新帧，不落盘，不默认对云端开放；启用预览不自动启用事件录像。预览与 WS 状态通道分离。

**part 头（用于验证"看到的就是模型输入"）**：每个分片在 JPEG 之前带

```
Content-Type: image/jpeg
Content-Length: <字节数>
X-Seq: <送进推理的帧序号，从 0 起>
X-Timestamp-Ms: <该帧在推理时间轴上的毫秒位置>
X-Sha256: <jpeg 字节的 sha256，小写十六进制>
```

`X-Seq` 与 `X-Timestamp-Ms` 描述的是**该帧在推理输入里的位置**，不是预览自己的计数；因此预览显示的第 N 帧就是推理消费的第 N 帧。客户端不得对 JPEG 重新编码后比对摘要。

**帧来源**：`DGB_LIVE_SOURCE=camera` 时为设备摄像头（`DGB_CAMERA_DEVICE_INDEX`、`DGB_CAMERA_WIDTH/HEIGHT`、`DGB_CAMERA_BACKEND`），`=frames` 时为预抽好的 JPEG 序列（联调/无摄像头环境）。两种来源都经过同一层 `PreviewTap`，字节不做二次编码；离线回放（分析任务接手）期间不提供预览。

`GET /api/v1/monitoring/preview/state`（API-010a）

- 鉴权：monitoring:read。
- 用途：页面在开流前判断是否可预览，并展示对齐信息。
- 成功：`200 PreviewState`：`enabled`、`available`、`clients`、`max_clients`、`max_fps`、`pipeline`（`realtime|playback|stopped`）、`source`（`camera|frames`）、`device_index`、`sequence`、`source_timestamp_ms`、`sha256`、`bytes`、`age_ms`、`frames_out`、`last_error`。
- `sequence`/`sha256` 为最近一帧**已送进推理**的画面；尚未收到帧时为 null，不得用 0 或空摘要代替。
- 错误：401 AUTH_REQUIRED；403 FORBIDDEN。

### API-010 创建分析任务

`POST /api/v1/analysis/jobs`（权限 `analysis:write`）

- 请求：`{"source_kind": "frames", "source_path": "<media_root>/analysis/<job_id>/frames", "force": false}`。
- 用途：用一份已抽好的 JPEG 目录替换当前推理素材，重跑分析并新开会话。
- 成功：`201 AnalysisJob`，`status="running"`，返回新 `session_id`、`frames_total`。
- **互斥**：实时监控正在跑且 `force=false` 时返回 `409 MONITORING_ACTIVE`，不产生任何副作用。
  确认切换请带 `force=true`。
- 错误：`400 INVALID_ANALYSIS_SOURCE`（路径不存在／不在媒体根目录内／无 JPEG／超帧数或磁盘上限／暂不支持的 `source_kind`）。
- 副作用：停掉当前推理循环与旧会话（旧会话 `ended_at` 落库），再按新素材启动。

### API-011 分析任务列表

`GET /api/v1/analysis/jobs`（权限 `analysis:read`）

- 查询参数：可选 `status`（`queued|running|completed|failed|cancelled`），`page`、`page_size`。
- 成功：`200`，`items` 按 `created_at` 倒序，`has_next` 表示是否还有下一页。
- 非法 `status`：`422 VALIDATION_ERROR`。

### API-012 分析任务详情

`GET /api/v1/analysis/jobs/{job_id}`（权限 `analysis:read`）

- 成功：`200 AnalysisJob`，含 `frames_total`／`frames_done`／`timeline_ms`／终态时间。
- 不存在：`404 ANALYSIS_JOB_NOT_FOUND`。
- 进度由服务端后台跟踪写入，不依赖客户端轮询，故列表页看到的状态同样准确。

### API-013 取消分析任务

`DELETE /api/v1/analysis/jobs/{job_id}`（权限 `analysis:write`）

- 用途：停止正在跑的分析并结束其会话。
- 成功：`200 AnalysisJob`，`status="cancelled"`、`ended_at` 非空，已完成的 `frames_done` 保留。
- 任务已结束：`409 ANALYSIS_JOB_CONFLICT`。
- 不存在：`404 ANALYSIS_JOB_NOT_FOUND`。

### API-014 上传视频并开始分析

`POST /api/v1/analysis/jobs/upload`（`multipart/form-data`，权限 `analysis:write`）

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `file` | file | — | 视频文件，后缀限 `mp4/mov/avi/mkv/webm/m4v` |
| `step_ms` | integer | `40` | 抽帧步长，必须是 `40/80/120/160/200` 之一（25 Hz 栅格） |
| `size` | string | `1280x720` | 输出尺寸，宽高必须为偶数 |
| `quality` | integer | `85` | JPEG 质量 1..100 |
| `retain_source` | boolean | 由配置决定（默认 `false`） | 是否保留原始视频；保留时可用 `<video>` 回放 |
| `force` | boolean | `false` | 实时监控在跑时必须显式确认；否则返回 `409 MONITORING_ACTIVE`，且不会开始抽帧 |

- 处理：存盘（同时算 sha256 并卡大小上限）→ 抽帧到 `<media_root>/analysis/<job_id>/frames/`
  → 写 `job.json` → 删除原视频（`retain_source=false` 时）→ 开新会话并启动推理。
- 成功：`201 AnalysisJob`（`source_kind="frames"`、`source_sha256` 非空、`frames_total` 为实际写入帧数）。
- 错误：`400 INVALID_ANALYSIS_SOURCE`（非视频后缀／文件为空或超 `analysis.max_upload_mb`／磁盘超 `max_disk_mb`／
  参数不在栅格上／解码失败／帧数超上限）、`503 PIPELINE_UNAVAILABLE`（缺少解码依赖）。
- 抽帧在请求内完成，失败时整目录回滚，**不会留下空任务**，因此不会出现 "queued 但永远不跑" 的记录。

### API-015 回放证据

`GET /api/v1/analysis/jobs/{job_id}/evidence`（权限 `analysis:read`）

- 用途：把「检测结果在哪几帧」一次性给前端：逐 tick 风险曲线 + 事件帧区间。
- 成功：`200 AnalysisEvidence`：
  - `step_ms` / `frames_total` / `duration_ms` / `source_available`；
  - `timeline`：`{ms, frame, risk_level, primary_risk, confidence}`，服务端降采样到
    最多 1200 点，每个区间保留最高风险点（尖峰不会被抹平）；
  - `events`：`{event_id, event_type, risk_level, started_ms, ended_ms, start_frame, end_frame, duration_ms}`。
- 换算规则：事件时钟是「会话起点 + 素材偏移」，所以 `started_at - session.started_at`
  就是素材毫秒，除以 `step_ms` 即帧号。
- 逐点数据来自任务目录下的 `timeline.ndjson`（运行期逐 tick 记录，任务结束时落盘）。
- 不存在：`404 ANALYSIS_JOB_NOT_FOUND`。

### API-016 原始视频

`GET /api/v1/analysis/jobs/{job_id}/source`（权限 `analysis:read`）

- 用途：`<video>` 回放。仅当该任务 `retain_source=true` 时可用。
- 成功：`200`，`video/*`，支持 `Range`（可拖动）。
- 未保留或已清理：`404 ANALYSIS_SOURCE_MISSING`。
- 鉴权：`<video src>` 不能带请求头，因此额外支持 `?access_token=<token>`（与 WS-001 同一取舍）。

### API-017 单帧图像

`GET /api/v1/analysis/jobs/{job_id}/frames/{index}`（权限 `analysis:read`）

- 用途：原视频未保留（或浏览器无法解码）时逐帧回放；`index` 从 0 开始。
- 成功：`200 image/jpeg`。
- 帧号越界或抽帧产物已过期：`404 ANALYSIS_SOURCE_MISSING`。
- 鉴权同 API-016。

### API-018 恢复实时监控

`POST /api/v1/monitoring/resume`（权限 `monitoring:write`）

- 用途：离线回放结束后管线处于停止状态，用它切回实时采集（会新开一个会话）。
- 成功：`200 {state, playback, session_id}`；`playback=false` 表示确实是实时实现。
- 幂等：已在实时模式时直接返回当前状态，不会重复开会话。
- 有分析任务仍在运行：`409 ANALYSIS_JOB_CONFLICT`（先取消或等它结束）。
- 未认证：`401 AUTH_REQUIRED`（匿名只读不允许启停管线）。

## 5. WebSocket

### WS-001 实时监控

`GET /ws/v1/monitoring?access_token=<token>`，升级为 WebSocket。若部署环境支持，应优先通过安全 Cookie 或 `Sec-WebSocket-Protocol` 传递短期凭证，避免 Token 进入 URL 日志。

服务端消息 envelope：

```json
{
  "type": "risk.snapshot",
  "schema_version": "1.0",
  "sequence": 18342,
  "sent_at": "2026-09-10T08:30:04.630Z",
  "data": {}
}
```

消息类型：

| `type` | 说明 |
| --- | --- |
| `connection.ready` | 连接就绪、心跳参数和当前 session |
| `risk.snapshot` | 完整实时风险快照 |
| `system.status` | 摄像头/模型/管线状态变化 |
| `event.changed` | 事件创建、更新或结束通知 |
| `error` | 可恢复或终止错误 |
| `heartbeat` | 连接保活 |

规则：

- 服务端默认最高推送 10 Hz，可配置但不得超过推理快照产生率。
- 客户端过慢时丢弃旧 `risk.snapshot`，保留最新状态；事件变化消息不得静默丢弃。
- 客户端发现序列跳跃时通过 REST 获取最新完整快照，不要求服务端回放每个帧级消息。
- 客户端使用带抖动的指数退避重连，最大间隔 30 秒。
- 序列号在同一 session 内单调递增；新会话可以重置，客户端清理旧曲线。无有效快照时发送 system.status，不强行发送 risk.snapshot。
- `system.status` 的 data 使用 OpenAPI 的 SystemStatus；warming_up 等细节放入 inference.message，status 使用现有 ok/degraded/unavailable 枚举。断开或超出配置的新鲜度阈值后 UI 标记旧快照过期。
- `event.changed` 为查询提示，需包含 event_id、session_id 和变更状态；事件通知无法入队时显式关闭连接并提示重查，恢复后由 API-003/004 补查，不承诺无界缓存或逐帧回放。

## 6. 认证与授权

| 权限 | 能力 |
| --- | --- |
| `monitoring:read` | 状态、实时快照、事件、统计读取及显式启用的预览 |
| `monitoring:write` | 启停推理管线（从离线回放切回实时） |
| `settings:read` | 配置读取 |
| `settings:write` | 配置更新 |
| `models:read` | 模型元数据读取 |
| `analysis:read` | 分析任务列表与详情读取 |
| `analysis:write` | 创建与取消分析任务 |

MVP 可以使用单个本地管理员 Token，但代码和契约应保留权限声明；生产部署不得只依赖前端隐藏按钮。

## 7. 错误码

| HTTP | Code | 场景 |
| --- | --- | --- |
| 400 | `INVALID_TIME_RANGE` | 时间范围不合法 |
| 400 | `INVALID_ANALYSIS_SOURCE` | 分析素材路径不存在、越出媒体根目录或不合规 |
| 401 | `AUTH_REQUIRED` | 缺少或无效凭证 |
| 403 | `FORBIDDEN` | 权限不足 |
| 404 | `EVENT_NOT_FOUND` | 事件不存在 |
| 404 | `ANALYSIS_JOB_NOT_FOUND` | 分析任务不存在 |
| 404 | `ANALYSIS_SOURCE_MISSING` | 回放素材不在（原视频未保留、抽帧过期、帧号越界） |
| 409 | `CONFIG_VERSION_CONFLICT` | 配置乐观锁冲突 |
| 409 | `MONITORING_NOT_ACTIVE` | 当前无监测会话 |
| 409 | `MONITORING_ACTIVE` | 实时监控正在跑，离线任务需要 `force=true` |
| 409 | `ANALYSIS_JOB_CONFLICT` | 分析任务已是终态，无法取消 |
| 422 | `VALIDATION_ERROR` | 字段类型或范围错误 |
| 422 | `TIME_RANGE_TOO_LARGE` | 统计时间跨度超限 |
| 429 | `RATE_LIMITED` | 超过限流策略 |
| 500 | `INTERNAL_ERROR` | 未分类内部错误，不返回堆栈 |
| 503 | `PIPELINE_UNAVAILABLE` | 推理管线不可用 |

## 8. 兼容、幂等与弃用

- `/api/v1` 内允许新增可选响应字段，不得改变已有字段语义或类型。
- 删除/重命名字段或改变枚举语义需要新主版本或明确的弃用周期。
- GET 无副作用；配置 PUT 使用版本号实现幂等和并发控制。
- 当前 API 无长任务端点。未来若增加模型导入、批量导出或大规模数据处理，必须采用 Init → Step → Poll/Cancel，而非单请求阻塞。

## 9. Requirement → Endpoint 映射

| Requirement | 端点/消息 |
| --- | --- |
| REQ-001/002 | API-002、API-009、WS-001 |
| REQ-003/004 | API-009、`risk.snapshot` |
| REQ-005 | API-006/007、`event.changed` |
| REQ-006 | API-003/004 |
| REQ-007 | API-001 至 API-009、WS-001 |
| REQ-008 | API-006/007 |
| REQ-009 | API-008 |
| REQ-010 | 本文档与 `openapi.yaml` |
| REQ-013 | API-002/008/009、无效快照与专家运行时语义 |
| REQ-014 | API-010、WS-001、预览鉴权和进程解耦 |

## 10. 契约验证清单

- [x] 每个端点定义用途、成功状态和至少一类错误。
- [x] 管理端点定义认证和权限。
- [x] 列表端点定义分页、过滤、稳定排序和最大页大小。
- [x] 写操作定义幂等与冲突行为。
- [x] WebSocket 定义版本、序列、背压、重连和错误消息。
- [x] 当前无长任务；未来长任务的 Init-Step-Poll 约束已声明。
- [ ] OpenAPI 草案需在后端实现后通过框架生成结果进行 diff 验证。

0.2.0 草案变更：增加预览端点、快照 503 和 runtime 枚举；同步 OpenAPI 与数据库约束。尚无已发布客户端；实现时 TypeScript/Pydantic 的穷举分支应同步枚举。公共 URL 主版本仍为 v1，RiskSnapshot 消息 schema 仍为 1.0。

## 资料来源

- 2026-09-10 项目初始规划基线：FastAPI、REST API、WebSocket、Dashboard 展示字段和 SQLite 事件记录。
- `docs/specs/PROJECT_SPEC.md`：REQ-004、REQ-006、REQ-007、REQ-008、REQ-010。
- `docs/database/DATABASE_DESIGN.md`：事件、配置和模型字段。
