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

**决策 D1：采用服务端解码（方案 A），不采用浏览器解码。**（✅ 2026-10-19 已实现）

| 依据 | 说明 |
| --- | --- |
| 时间戳确定性 | 25 Hz 栅格要求严格 40 ms 间隔；浏览器 `currentTime` seek 是「关键帧近似」，帧级精确需 WebCodecs + 自建解封装 |
| 编解码兼容 | ffmpeg/cv2 可吃常见码流；浏览器对 HEVC 支持不足 |
| 可批量、可重跑 | 素材留在服务端，失败重跑零成本 |
| 成本 | 比前端方案多约 1–2 人天 |

**决策 D2：解码依赖选 `opencv-python-headless`。**（✅ 已落地：`app/services/frame_extraction.py`）

与模型仓库 `scripts/local/extract_frames.py` 使用同一套 API，抽帧逻辑已验证；轮子约 40 MB。
同一段 3 秒 30 fps 合成视频，两套实现均输出 75 帧且逐帧像素差为 0。

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
            ├── source.<ext>           # 原始上传（mp4/mov/...）；retain_source=true 才保留
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
| M1（步骤 2）✅ 2026-10-19 | 任务表 + 接口（提交 `4c3270e`） | 用 curl 提交一个已有目录并观察进度 |
| M2（步骤 3）✅ 2026-10-19 | 上传 + 抽帧（API-014） | 拖一个 mp4 跑完，`curl` 查到事件 |
| M3（步骤 4）✅ 2026-10-19 | 前端「离线分析」页 + 事件/统计按会话过滤 + 回放提示 | 页面上完成全流程 |
| M4（步骤 5） | 互斥 + 回归 | 与实时能力共存 |
| M5（回放与证据）✅ 2026-10-19 | 逐 tick 时间轴 + 事件帧区间 + 原片/帧回放 | 点一条事件就能跳到出问题的那一帧 |

M1 验收对照（`backend/tests/test_analysis_jobs_api.py`，11 个用例）：

| 验收项 | 用例 |
| --- | --- |
| `frames_done` 单调不减 | `test_create_job_runs_to_completion` |
| 新 `session_id` 且旧会话 `ended_at` 落库 | `test_job_opens_new_session_and_closes_previous` |
| DELETE → `cancelled` 且引擎停止 | `test_cancel_running_job_stops_engine` |
| `source_path` 缺失/越界 → 400 | `test_missing_source_is_rejected`、`test_path_outside_media_root_is_rejected` |

M2 验收对照（`backend/tests/test_analysis_upload_api.py`，9 个用例）：

| 验收项 | 结果 |
| --- | --- |
| 上传视频 → 抽帧 → 任务跑完 | ✅ `test_upload_extracts_frames_and_runs`（合成 10 帧 mp4，断言 JPEG 数 = `frames_total`、任务 `completed`） |
| `retain_source=false` 时原视频消失 | ✅ `test_upload_extracts_frames_and_runs`；保留路径见 `test_retain_source_keeps_video` |
| 超 `max_disk_mb` 拒绝新任务 | ✅ `test_disk_budget_blocks_new_upload`、`test_upload_size_limit` |
| 抽帧失败可读 | ⚠️ **改为请求期 400**，不是 `failed` 任务（见下） |
| 抽帧规则与模型仓库一致 | ✅ 同一 3 秒 30 fps 合成视频：两套实现均为 75 帧、逐帧像素差 0 |

**与步骤 3 原文的偏差（抽帧失败的落点）**：原文写的是"抽帧失败时任务为 `failed`"。
实际实现把抽帧放在**建任务之前**同步完成（放线程里跑），因此坏文件、超限视频都在
建立 `analysis_jobs` 记录之前就返回 `400 INVALID_ANALYSIS_SOURCE`，并把半个目录删掉。
理由：`session_id` 是任务表的必填外键，若先建 `queued` 记录再抽帧，就要为"还没开始
推理的任务"引入一个空会话或可空外键，得不偿失。代价是长视频的上传响应要等抽帧结束
（10 分钟素材约数十秒），进度条只能给"正在准备"，这也是 M3 之前可接受的状态。

**尚未验证**：10 秒视频跑完后"事件总时长落在 10 秒内"。这条依赖真实模型输出，
而合成素材不会产生任何行为事件；等有真实驾驶素材时补测。

M3 落地情况（2026-10-19，真机浏览器验证：内置静态产物 + uvicorn，推理用脚本桩）：

| 要求 | 结果 |
| --- | --- |
| 上传进度 → 任务进度（帧数/时间轴） | ✅ 上传用 XHR 报百分比；任务卡每秒轮询，显示 `frames_done/total` 与素材时间轴 |
| 完成跳转 | ✅ “查看该会话事件 / 统计”切到对应页并自动带上 `session_id` |
| 事件页会话过滤 | ✅ 文本输入 + 跳转自动填充；带“仅显示会话 xxx”提示与一键清除 |
| 统计页会话过滤 | ✅ 后端 `summary(session_id=...)` + 前端输入；回显 `session_id` 便于确认空结果归属 |
| 监控页“非实时”提示 | ✅ `/system/status` 新增 `pipeline.playback`；页面显示“当前离线回放，不是实时采集” |
| 刷新/切标签不丢状态 | ✅ 进度在服务端；“当前任务”靠 `localStorage` 记住 job_id 并重新拉取 |

已验证的浏览器路径：上传 50 帧合成视频 → 进度 37/50 → 50/50 完成 → 事件页只剩该会话 1 条
（`fatigue / DANGER / ended / 1.7 s`）→ 统计页显示 1 条、累计 1.7 s。

**前端仍缺**：实时在跑时提交离线任务的确认/互斥提示（M4）。

M5（回放与证据，2026-10-19）

需求：离线回放时不仅要看“最终指标”，还要知道**哪几帧**出了问题。

| 决策 | 理由 |
| --- | --- |
| 逐 tick 风险写入 `analysis/<job_id>/timeline.ndjson` | 只在内存累加，任务结束时一次落盘；不逐 tick 做磁盘 IO |
| 事件→帧号由服务端换算 | 事件时钟 = 会话起点 + 素材偏移，除以 `step_ms` 即帧号，前端不做时间猜测 |
| 回放源两种：保留原片用 `<video>`，否则逐帧取图 | NFR-003 默认不存原片；帧序列同样精确到帧 |
| 媒体端点支持 `?access_token=` | `<video>`/`<img>` 不能带请求头（与 WS-001 同一取舍） |
| 浏览器解码失败自动退回帧序列 | 脚本生成的 mp4v 素材浏览器解不了；真实 H.264 手机视频才走 `<video>` |

验证（真机浏览器，内置 dist + 演示后端）：50 帧素材 → 回放卡显示「第 7 帧 / 共 50 帧」、
事件 `fatigue 7–48 帧 / 0:00.2–0:01.9`；点「定位」跳到第 7 帧并显示 `DANGER / fatigue 0.91`；
风险曲线绿→红→绿，点击曲线可任意跳转；原视频因 mp4v 编码自动退回帧序列并给出提示。
后端用例 `tests/test_analysis_evidence_api.py`（3 个）覆盖时间轴落盘、事件帧映射、Range 与原片缺失。

## 8. 与既有文档的关系

- 本计划的接口约定落地后需回写 `docs/api/API_DESIGN.md`；
- NFR-003 的配套配置需回写 `docs/requirements/REQUIREMENTS.md` 与设置项注册表；
- 任务表需新增 Alembic 迁移（当前仅 `0001_initial`）。
