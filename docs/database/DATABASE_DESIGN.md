# DriveGuard AI 数据库设计

## 1. 目标与边界

MVP 使用 SQLite 保存监测会话、危险事件、系统配置、模型版本与必要审计信息。数据库不保存逐帧特征流，默认不保存原始视频或图像。推理热路径不应同步等待复杂统计查询。

本文是设计基线；仓库当前没有迁移脚本或 ORM 模型，实现时应以版本化 migration 落地并回写实际差异。

## 2. 设计原则

- UTC 时间：存储 ISO 8601 UTC 字符串，应用层负责时区展示。
- 稳定标识：业务主键使用 UUID 文本，避免迁移 PostgreSQL 时暴露本地自增语义。
- 枚举约束：风险等级和状态通过 CHECK 约束限制。
- 元数据优先：默认不保存生物特征图像和完整视频。
- 可迁移性：避免依赖 SQLite 特有业务语义；JSON 字段由应用 schema 校验。
- 单写者友好：开启 WAL、短事务、索引命中，避免推理线程直接写库。

## 3. 实体关系

```text
monitoring_sessions 1 ────── N driving_events
model_versions      1 ────── N driving_events
system_settings     N ────── 1 logical configuration history
audit_logs          N ────── 1 administrative actions
```

## 4. 表结构

### 4.1 `monitoring_sessions`

| 字段 | SQLite 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | TEXT | PK | UUID |
| `started_at` | TEXT | NOT NULL | UTC 开始时间 |
| `ended_at` | TEXT | NULL | UTC 结束时间 |
| `camera_id` | TEXT | NOT NULL | 本地摄像头逻辑标识 |
| `device_id` | TEXT | NOT NULL | 边缘设备逻辑标识 |
| `status` | TEXT | CHECK | `active/ended/failed` |
| `model_bundle_version` | TEXT | NOT NULL | 本次运行使用的模型组合版本 |
| `created_at` | TEXT | NOT NULL | 创建时间 |

### 4.2 `driving_events`

| 字段 | SQLite 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `event_id` | TEXT | PK | UUID |
| `session_id` | TEXT | FK, NOT NULL | 所属监测会话 |
| `event_type` | TEXT | NOT NULL | `fatigue/phone_use/drinking/distraction/eyes_off_road/other` |
| `started_at` | TEXT | NOT NULL | UTC 开始时间 |
| `ended_at` | TEXT | NULL | 活动事件可为空 |
| `duration_ms` | INTEGER | NOT NULL, `>= 0` | 持续时长 |
| `max_confidence` | REAL | CHECK `0..1` | 最大置信度 |
| `max_risk_level` | TEXT | CHECK | `SAFE/WARNING/DANGER` |
| `status` | TEXT | CHECK | `active/ended/discarded` |
| `model_version_id` | TEXT | FK, NULL | 主时序/融合模型版本 |
| `risk_scores_json` | TEXT | NOT NULL | 各风险分数快照，JSON object |
| `reason_codes_json` | TEXT | NOT NULL | 解释原因，JSON array |
| `quality` | TEXT | CHECK | `good/degraded/unknown` |
| `clip_path` | TEXT | NULL | 显式启用片段保存时的受控相对路径 |
| `created_at` | TEXT | NOT NULL | 创建时间 |
| `updated_at` | TEXT | NOT NULL | 最后更新时间 |

### 4.3 `system_settings`

| 字段 | SQLite 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `setting_key` | TEXT | PK | 点分命名，如 `alarm.cooldown_ms` |
| `value_json` | TEXT | NOT NULL | JSON 编码值 |
| `value_type` | TEXT | CHECK | `string/integer/number/boolean/object/array` |
| `version` | INTEGER | NOT NULL, `>= 1` | 乐观锁版本 |
| `requires_restart` | INTEGER | CHECK `0/1` | 是否需要重启 |
| `is_secret` | INTEGER | CHECK `0/1` | 敏感值标记；API 不返回原值 |
| `updated_at` | TEXT | NOT NULL | UTC 更新时间 |
| `updated_by` | TEXT | NULL | 操作者标识 |

保留键 `system.settings_revision` 的 `version` 作为整组配置的全局乐观锁版本；任意配置成功更新时，该版本在同一事务内递增。单项 `version` 仍用于审计具体键的变化。`eyes_off_road` 事件类型为 Phase 2 预留值，功能未启用前不得由 MVP 推理管线生成。

### 4.4 `model_versions`

| 字段 | SQLite 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `model_version_id` | TEXT | PK | UUID |
| `model_name` | TEXT | NOT NULL | 模型逻辑名 |
| `model_type` | TEXT | CHECK | `detector/landmarker/temporal/fusion` |
| `semantic_version` | TEXT | NOT NULL | 模型版本 |
| `runtime` | TEXT | CHECK | `pytorch/onnxruntime/tensorrt` |
| `artifact_path` | TEXT | NOT NULL | 受控模型目录内相对路径 |
| `sha256` | TEXT | NOT NULL | 工件校验值 |
| `feature_schema_version` | TEXT | NULL | 输入特征 schema 版本 |
| `metrics_json` | TEXT | NOT NULL | 评估指标 JSON |
| `status` | TEXT | CHECK | `registered/active/retired/failed` |
| `created_at` | TEXT | NOT NULL | 注册时间 |
| `activated_at` | TEXT | NULL | 激活时间 |

### 4.5 `audit_logs`

| 字段 | SQLite 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `audit_id` | TEXT | PK | UUID |
| `occurred_at` | TEXT | NOT NULL | UTC 时间 |
| `actor_id` | TEXT | NULL | 操作者；本地系统可为 `system` |
| `action` | TEXT | NOT NULL | 如 `settings.update` |
| `resource_type` | TEXT | NOT NULL | 资源类型 |
| `resource_id` | TEXT | NULL | 资源标识 |
| `result` | TEXT | CHECK | `success/failure` |
| `request_id` | TEXT | NULL | API 请求关联 ID |
| `details_json` | TEXT | NOT NULL | 去敏后的结构化详情 |

## 5. SQLite DDL 基线

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE monitoring_sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    camera_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'ended', 'failed')),
    model_bundle_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE model_versions (
    model_version_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL CHECK (model_type IN ('detector', 'landmarker', 'temporal', 'fusion')),
    semantic_version TEXT NOT NULL,
    runtime TEXT NOT NULL CHECK (runtime IN ('pytorch', 'onnxruntime', 'tensorrt')),
    artifact_path TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    feature_schema_version TEXT,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL CHECK (status IN ('registered', 'active', 'retired', 'failed')),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    UNIQUE (model_name, semantic_version, runtime)
);

CREATE TABLE driving_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES monitoring_sessions(session_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('fatigue', 'phone_use', 'drinking', 'distraction', 'eyes_off_road', 'other')),
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
    max_confidence REAL NOT NULL CHECK (max_confidence BETWEEN 0.0 AND 1.0),
    max_risk_level TEXT NOT NULL CHECK (max_risk_level IN ('SAFE', 'WARNING', 'DANGER')),
    status TEXT NOT NULL CHECK (status IN ('active', 'ended', 'discarded')),
    model_version_id TEXT REFERENCES model_versions(model_version_id),
    risk_scores_json TEXT NOT NULL DEFAULT '{}',
    reason_codes_json TEXT NOT NULL DEFAULT '[]',
    quality TEXT NOT NULL CHECK (quality IN ('good', 'degraded', 'unknown')),
    clip_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK ((status = 'active' AND ended_at IS NULL) OR status <> 'active')
);

CREATE TABLE system_settings (
    setting_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('string', 'integer', 'number', 'boolean', 'object', 'array')),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    requires_restart INTEGER NOT NULL DEFAULT 0 CHECK (requires_restart IN (0, 1)),
    is_secret INTEGER NOT NULL DEFAULT 0 CHECK (is_secret IN (0, 1)),
    updated_at TEXT NOT NULL,
    updated_by TEXT
);

CREATE TABLE audit_logs (
    audit_id TEXT PRIMARY KEY,
    occurred_at TEXT NOT NULL,
    actor_id TEXT,
    action TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    result TEXT NOT NULL CHECK (result IN ('success', 'failure')),
    request_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}'
);
```

## 6. 索引

```sql
CREATE INDEX idx_sessions_started_at ON monitoring_sessions(started_at DESC);
CREATE INDEX idx_events_started_at ON driving_events(started_at DESC);
CREATE INDEX idx_events_type_started ON driving_events(event_type, started_at DESC);
CREATE INDEX idx_events_risk_started ON driving_events(max_risk_level, started_at DESC);
CREATE INDEX idx_events_session_started ON driving_events(session_id, started_at);
CREATE INDEX idx_events_status ON driving_events(status) WHERE status = 'active';
CREATE INDEX idx_models_name_status ON model_versions(model_name, status);
CREATE INDEX idx_audit_occurred_at ON audit_logs(occurred_at DESC);
```

索引对应事件按时间分页、类型/风险筛选、会话内聚合和活动事件恢复。实现后必须用代表性数据和 `EXPLAIN QUERY PLAN` 验证，未命中的索引不应保留。

## 7. 事务边界

- 创建会话：单条事务。
- 风险进入活动状态：创建事件与必要审计记录在同一短事务内。
- 活动事件更新：以 `event_id + status=active` 限定更新，避免结束后继续累加。
- 配置更新：读取版本、校验、`version + 1` 更新和审计记录组成同一事务。
- 模型激活：将同名旧版本置为 `retired`、新版本置为 `active`，单事务完成。

## 8. 保留与隐私

- 事件元数据保留周期由 `storage.event_retention_days` 配置，默认值需产品确认。
- 审计日志保留周期不得短于事件元数据保留周期。
- `clip_path` 只能是媒体根目录下的相对路径；删除数据库记录时应通过受控清理任务处理文件。
- Secret 配置不应以明文写入本表；`is_secret=1` 只保存外部密钥引用或加密后的值。

## 9. 迁移与回滚

- 每个 schema 变更使用单调递增 migration，不直接在生产库手工修改。
- SQLite 破坏性变更采用“新表 → 数据复制 → 校验 → 原子改名”，并在操作前备份数据库。
- migration 必须提供向前验证和回滚/恢复说明；涉及数据丢失的变更必须改为可恢复迁移。
- PostgreSQL 迁移时将 UUID、JSONB、TIMESTAMPTZ 和原生枚举/检查约束映射为等价类型，并重新评估部分索引。

## 10. 一致性检查

实现数据库层时应核对：

- DDL、ORM/Repository、Pydantic schema 与 OpenAPI 字段一致。
- 新增枚举值同时更新数据库约束、API、TypeScript 类型和测试。
- 时间范围、分页过滤和排序均命中预期索引。
- 空库、单条、10 万条、非法 JSON、活动事件恢复和并发配置更新均有测试。

## 资料来源

- 2026-09-10 项目初始规划基线：MVP 使用 SQLite、记录事件元数据、后续迁移 PostgreSQL、默认不上传视频。
- `docs/specs/PROJECT_SPEC.md`：REQ-006、REQ-008、跨层一致性规则。
