# DriveGuard AI API 设计

## 1. 概述

MVP 使用 RESTful JSON API 处理查询和配置，使用 WebSocket 推送实时风险快照。基础路径为 `/api/v1`，WebSocket 路径为 `/ws/v1`。本文为设计契约，机器可读草案见 `openapi.yaml`。

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
  "reason_codes": ["PERCLOS_HIGH"]
}
```

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

`GET /api/v1/statistics/summary?started_from=2026-09-01T00:00:00Z&started_to=2026-09-10T23:59:59Z`

- 成功：总事件数、各类型数量/持续时间、各风险等级数量和按时间桶趋势。
- 无数据：返回数值 0 和空数组，状态仍为 `200`。
- 时间跨度过大：`422 TIME_RANGE_TOO_LARGE`，MVP 默认最大 90 天，可配置。

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

### API-009 当前风险快照

`GET /api/v1/monitoring/snapshot`

- 用途：页面首次加载或 WebSocket 重连后获取完整最新状态。
- 成功：`200 RiskSnapshot`。
- 无活动会话：`409 MONITORING_NOT_ACTIVE`。

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

## 6. 认证与授权

| 权限 | 能力 |
| --- | --- |
| `monitoring:read` | 状态、实时快照、事件和统计读取 |
| `settings:read` | 配置读取 |
| `settings:write` | 配置更新 |
| `models:read` | 模型元数据读取 |

MVP 可以使用单个本地管理员 Token，但代码和契约应保留权限声明；生产部署不得只依赖前端隐藏按钮。

## 7. 错误码

| HTTP | Code | 场景 |
| --- | --- | --- |
| 400 | `INVALID_TIME_RANGE` | 时间范围不合法 |
| 401 | `AUTH_REQUIRED` | 缺少或无效凭证 |
| 403 | `FORBIDDEN` | 权限不足 |
| 404 | `EVENT_NOT_FOUND` | 事件不存在 |
| 409 | `CONFIG_VERSION_CONFLICT` | 配置乐观锁冲突 |
| 409 | `MONITORING_NOT_ACTIVE` | 当前无监测会话 |
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

## 10. 契约验证清单

- [x] 每个端点定义用途、成功状态和至少一类错误。
- [x] 管理端点定义认证和权限。
- [x] 列表端点定义分页、过滤、稳定排序和最大页大小。
- [x] 写操作定义幂等与冲突行为。
- [x] WebSocket 定义版本、序列、背压、重连和错误消息。
- [x] 当前无长任务；未来长任务的 Init-Step-Poll 约束已声明。
- [ ] OpenAPI 草案需在后端实现后通过框架生成结果进行 diff 验证。

## 资料来源

- 2026-09-10 项目初始规划基线：FastAPI、REST API、WebSocket、Dashboard 展示字段和 SQLite 事件记录。
- `docs/specs/PROJECT_SPEC.md`：REQ-004、REQ-006、REQ-007、REQ-008、REQ-010。
- `docs/database/DATABASE_DESIGN.md`：事件、配置和模型字段。
