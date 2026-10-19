# 离线视频分析：实施方案与分步计划

> 2026-10-19 · 分支 `feat/mvp-backend-frontend`
> 目标：用户在页面上提交一个 mp4，后台按**素材时间轴**跑完整链路，产出事件与统计。
> 实时摄像头能力保留在接口层，不在本计划内实现。

## 1. 目标与非目标

**目标**

- 支持上传 mp4（H.264 等常见编码），服务端抽帧后复用现有 live 推理链路。
- 每次分析是一个独立任务：有状态、有进度、可取消、可重跑。
- 事件与统计按**素材时间轴**计量，不按墙钟。
- 页面可查看任务进度，完成后可跳转到该任务的事件与统计。

**非目标（本轮不做）**

- 实时摄像头 / 浏览器推流采集（仅保留 `AnalysisSource.kind` 的扩展位）。
- 模型精度改进、阈值调整、`drinking` 误报（KL-01）处理。
- 多任务并行（live 服务单活动会话，KL-06）。

## 2. 现状盘点（2026-10-19）

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| JPEG 序列离线推理 | ✅ 可用 | `DGB_INFERENCE_MODE=live` + `DGB_FRAME_SEQUENCE`，环境变量启动期生效 |
| 事件按素材时间轴计时 | ✅ 已修复 | `commit 830f5cc`；时长不再被墙钟放大 |
| 运行期切换素材 | ✅ 引擎层完成 | `InferenceEngine.run_analysis(AnalysisSource)`，无 HTTP 入口 |
| 任务模型 / 进度 / 取消 | ❌ | — |
| mp4 解码 | ❌ | 后端无 `cv2`/`av`/`PIL` |
| 前端分析页 | ❌ | 仅有监控/事件/统计/设置四页 |

## 3. 关键决策

**决策 D1：采用服务端解码（方案 A），不采用浏览器解码。**

| 依据 | 说明 |
| --- | --- |
| 时间戳确定性 | 25 Hz 栅格要求严格 40 ms 间隔；浏览器 `currentTime` seek 是「关键帧近似」，帧级精确需 WebCodecs + 自建解封装 |
| 编解码兼容 | ffmpeg/cv2 可吃常见码流；浏览器对 HEVC 支持不足 |
| 可批量、可重跑 | 素材留在服务端，失败重跑零成本 |
| 成本 | 比前端方案多约 1–2 人天 |

**决策 D2：解码依赖选 `opencv-python-headless`。**

与模型仓库 `scripts/local/extract_frames.py` 使用同一套 API，抽帧逻辑已验证；轮子约 40 MB。

**决策 D3：采集方式用 `AnalysisSource` 隔离。**

引擎只认 `kind`，不认采集方式：

| kind | 谁发起 | 形态 | 状态 |
| --- | --- | --- | --- |
| `frames` | 服务端拉 | 目录中的 JPEG | ✅ 已实现 |
| `camera` | 服务端拉 | RTSP / USB 视频流 | 预留 |
| `push` | 浏览器推 | 帧经 HTTP 打入 | 预留 |

## 4. 产物存放

所有运行期产物都在 `backend/var/` 下，该目录已在 `.gitignore` 中（注释：运行期数据：
SQLite 与媒体目录默认不入库（隐私 NFR-003））。

```
backend/var/                          # BACKEND_DIR/var，默认数据库与媒体根
├── driveguard.db                     # 关系数据；analysis_jobs 只存指针，不存二进制
└── media/                            # DGB_MEDIA_DIR（事件片段也用此处）
    └── analysis/
        └── <job_id>/                 # 每个任务一个目录
            ├── source.mp4            # 原始上传；retain_source=true 才保留
            ├── frames/               # 抽帧产物，补零命名 → 字典序即播放序
            │   └── 000000.jpg ...
            └── job.json              # step_ms/size/quality/帧数/素材 sha256（复跑凭证）
```

按任务分目录的理由：保留期到期清理是一次 `rmtree`；配额统计是一个目录的 `du`；
同名素材不互相覆盖；失败任务可整目录删除。

**路径包含性校验（必需）**：`source_path` 解析后必须位于 `media_root` 之下，否则返回
400。否则该接口等于允许读取服务器任意目录。模型仓库的 `verified_path()` 是同一模式。

## 5. 分步实施

每一步都可独立验收，不必等全部完成才可用。

### 步骤 2：任务模型与接口（预估 0.5–1 人天）

**数据表 `analysis_jobs`**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `job_id` | TEXT PK | UUID |
| `session_id` | TEXT FK | 关联 `monitoring_sessions`，事件按此隔离 |
| `source_kind` | TEXT | `frames` / `camera` / `push` |
| `source_path` | TEXT | 素材路径（`frames` 为目录） |
| `source_sha256` | TEXT | 原始文件摘要，用于去重与追溯；上传方案实现后填充 |
| `status` | TEXT | `queued` / `running` / `completed` / `failed` / `cancelled` |
| `frames_total` | INTEGER | 抽帧后总帧数；`frames` 来源直接数文件 |
| `frames_done` | INTEGER | 已处理帧数（由推理状态 `frames_sent` 回填） |
| `timeline_ms` | INTEGER | 当前素材时间轴位置 |
| `error` | TEXT | 失败原因（截断 400 字符） |
| `created_at` / `started_at` / `ended_at` | TEXT | 时间戳 |

约束：`frames_done <= frames_total`；`status='completed'` 时 `ended_at` 非空。

**接口草案**

```
POST   /api/v1/analysis/jobs          创建并启动任务
GET    /api/v1/analysis/jobs          列表（分页、按状态过滤）
GET    /api/v1/analysis/jobs/{id}     详情（含进度）
DELETE /api/v1/analysis/jobs/{id}     取消（仅 queued/running）
```

```jsonc
// POST 请求（步骤 2 仅支持已有目录；步骤 3 增加上传）
{ "source_kind": "frames", "source_path": "<media_root>/analysis/<job_id>/frames" }

// 响应
{
  "job_id": "...", "session_id": "...", "status": "running",
  "frames_total": 961, "frames_done": 0, "timeline_ms": 0, "error": null,
  "created_at": "...", "started_at": "...", "ended_at": null
}
```

**权限**：新增 `analysis:read` / `analysis:write`；未配置 Token 时沿用现有匿名只读策略。

**验收标准**

1. 创建任务后 `GET /jobs/{id}` 的 `frames_done` 单调递增；
2. 任务对应的 `session_id` 与上一个任务不同，且旧会话 `ended_at` 已填充；
3. `DELETE` 后任务为 `cancelled`，引擎停止推进；
4. 已有任务的 `source_path` 不存在时返回 400，不静默失败；
5. `source_path` 越出 `media_root`（如 `../../`）时返回 400，不读取外部目录。

### 步骤 3：上传与抽帧（预估 1–1.5 人天）

**新增依赖**：`opencv-python-headless` → 重新生成 `requirements.lock.txt`（`run.ps1` 哨兵会因锁文件变化触发重装，属预期）。

**接口**：`POST /api/v1/analysis/jobs/upload`（`multipart/form-data`）

抽帧规则与模型仓库保持一致：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `step_ms` | 40 | 25 Hz 栅格；必须为 40 的整数倍 |
| `size` | `1280x720` | 输出尺寸 |
| `quality` | 85 | JPEG 质量 |

**NFR-003 配套（必做，否则违反需求）**

`docs/requirements/REQUIREMENTS.md` §NFR-003 要求「默认不上传原始视频，默认不持久化完整帧；启用时必须配置保存目录、保留期、磁盘上限和访问权限」。因此必须同时交付：

| 配置项 | 默认 | 作用 |
| --- | --- | --- |
| `analysis.retain_source` | `false` | 抽帧完成后立即删除原始 mp4 |
| `analysis.retain_frames_hours` | `24` | JPEG 序列保留期 |
| `analysis.max_disk_mb` | `2048` | 占用上限，超限拒绝新任务 |
| `analysis.media_root` | `var/media` | 素材根目录（与 `DGB_MEDIA_DIR` 对齐）；所有任务产物在其下 |

清理任务：按保留期删除过期任务目录，并写一条 `audit_logs`。

**验收标准**

1. 上传 10 秒 mp4 → 生成 250 张 JPEG → 任务跑完 → 事件时长落在 10 秒内；
2. `retain_source=false` 时原 mp4 在抽帧后消失；
3. 超过 `max_disk_mb` 时新任务被拒绝并给出明确错误；
4. 抽帧失败（非视频文件、编码不支持）时任务为 `failed` 且 `error` 可读。

### 步骤 4：前端分析页（预估 1 人天）

- 新页「分析」：选择文件 → 上传进度 → 任务进度（帧数/时间轴）→ 完成跳转；
- 事件页与统计页增加 `session_id` 过滤（当前只有时间范围过滤）；
- 监控页在 `source_kind != push` 时显示「当前为回放/离线，非实时」提示。

**验收标准**：从上传到看到事件全流程无需手动改配置；刷新页面能恢复任务进度。

### 步骤 5：模式互斥与回归（预估 0.5 人天）

`run_analysis` 会结束当前会话。接入实时后必须加闸门：

- `push`/`camera` 运行中时，创建离线任务返回 409，除非显式 `force=true`；
- 回归：`tests/test_analysis_switch.py` 扩充到覆盖互斥分支。

## 6. 风险与未决问题

| 项 | 风险 | 处置 |
| --- | --- | --- |
| KL-06 单活动会话 | 两个任务并发会互相踢掉 | 任务队列串行执行；拒绝并发 |
| 长视频耗时 | CPU 上 38 秒素材约 4 分钟，1 小时视频不可行 | 先限制单任务时长与帧数上限，超限拒绝 |
| 磁盘 | 961 帧 ≈ 117 MB，是原视频的 2.5 倍 | `max_disk_mb` + 保留期清理 |
| `drinking` 误报（KL-01） | 每个素材都会产出大量「喝水」事件 | 属模型问题，本计划不处理；页面需保留既有提示 |
| 精度未验证 | 发布包限定为联调候选 | 页面与分析报告必须标注「非精度结论」 |

**待确认**：单任务最长时长上限、`retain_frames_hours` 的实际取值、是否需要保留抽帧产物供复跑（影响磁盘估算）。

## 7. 里程碑

| 里程碑 | 交付 | 可演示的能力 |
| --- | --- | --- |
| M1（步骤 2） | 任务表 + 接口 | 用 curl 提交一个已有目录并观察进度 |
| M2（步骤 3） | 上传 + 抽帧 | 拖一个 mp4 跑完，`curl` 查到事件 |
| M3（步骤 4） | 前端页面 | 页面上完成全流程 |
| M4（步骤 5） | 互斥 + 回归 | 与实时能力共存 |

## 8. 与既有文档的关系

- 本计划的接口约定落地后需回写 `docs/api/API_DESIGN.md`；
- NFR-003 的配套配置需回写 `docs/requirements/REQUIREMENTS.md` 与设置项注册表；
- 任务表需新增 Alembic 迁移（当前仅 `0001_initial`）。
