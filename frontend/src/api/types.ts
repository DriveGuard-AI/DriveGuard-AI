/**
 * 与 `docs/api/API_DESIGN.md` / `backend/app/schemas.py` 对齐的前端类型。
 * 契约变化时必须同步（见 docs/governance/DOCUMENTATION_GOVERNANCE.md）。
 */

export type RiskLevel = 'SAFE' | 'WARNING' | 'DANGER'
export type Quality = 'good' | 'degraded' | 'unknown'
export type EventStatus = 'active' | 'ended' | 'discarded'
export type InferenceMode = 'fake' | 'real'

export interface HeadPose {
  yaw: number
  pitch: number
  roll: number
}

export interface FeatureSet {
  ear: number | null
  mar: number | null
  perclos: number | null
  head_pose: HeadPose | null
}

export interface Detection {
  label: string
  confidence: number
  bbox: number[]
}

/**
 * 单个行为的判决参数（六参数表）。
 *
 * 判定依据是 `logit >= threshold_logit`，不是 score；`score` 是未校准 sigmoid，
 * 只能横向比较。`margin = logit - threshold_logit`：>0 已触发，<0 是安全余量。
 */
export interface TaskDetail {
  available: boolean
  logit: number | null
  score: number | null
  threshold_logit: number | null
  margin: number | null
  detected: boolean | null
  source: string | null
  score_semantics: string | null
}

export interface RiskSnapshot {
  schema_version: string
  session_id: string
  sequence: number
  captured_at: string
  risk_level: RiskLevel
  primary_risk: string | null
  confidence: number
  quality: Quality
  risk_scores: Record<string, number>
  features: FeatureSet
  detections: Detection[]
  /** 四任务的六参数；fake 回放或结果过期时为空对象，不伪造数字。 */
  task_details?: Record<string, TaskDetail>
  reason_codes: string[]
  inference_mode: InferenceMode
  /** 素材时间轴位置（毫秒）；实时模式或尚无证据时为 null。 */
  source_timestamp_ms: number | null
}

export interface EventOut {
  event_id: string
  session_id: string
  event_type: string
  started_at: string
  ended_at: string | null
  duration_ms: number
  max_confidence: number
  max_risk_level: RiskLevel
  status: EventStatus
  quality: Quality
  risk_scores: Record<string, number>
  reason_codes: string[]
  model_version_id: string | null
  clip_available: boolean
}

export interface EventListResponse {
  items: EventOut[]
  page: number
  page_size: number
  total: number
  has_next: boolean
}

export interface CountByType {
  event_type: string
  count: number
  total_duration_ms: number
}

export interface CountByRisk {
  risk_level: string
  count: number
}

export interface TrendBucket {
  bucket_start: string
  count: number
}

export interface StatisticsSummary {
  started_from: string
  started_to: string
  /** 回显过滤条件；null 表示未按会话过滤。 */
  session_id: string | null
  total_events: number
  by_type: CountByType[]
  by_risk_level: CountByRisk[]
  trend: TrendBucket[]
}

export type AnalysisJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type AnalysisSourceKind = 'frames' | 'camera' | 'push'

/** 离线分析任务（API-010~014）。每个任务对应一个新会话。 */
export interface AnalysisJob {
  job_id: string
  session_id: string
  source_kind: AnalysisSourceKind
  source_path: string
  source_sha256: string | null
  status: AnalysisJobStatus
  frames_total: number
  frames_done: number
  /** 抽帧步长（ms）：事件时间轴 ÷ 步长 = 帧号。 */
  step_ms: number
  timeline_ms: number
  error: string | null
  /** 原始视频是否还在：在则用 <video> 回放，否则逐帧取图。 */
  source_available: boolean
  created_at: string
  started_at: string | null
  ended_at: string | null
  updated_at: string
}

/** 回放用的风险采样点（服务端已降采样）。 */
export interface RiskPoint {
  ms: number
  frame: number
  risk_level: RiskLevel
  primary_risk: string | null
  confidence: number
}

/** 事件在素材时间轴和帧号上的位置。 */
export interface EvidenceEvent {
  event_id: string
  event_type: string
  risk_level: string
  started_ms: number
  ended_ms: number
  start_frame: number
  end_frame: number
  duration_ms: number
}

export interface AnalysisEvidence {
  job_id: string
  session_id: string
  step_ms: number
  frames_total: number
  duration_ms: number
  source_available: boolean
  source_kind: AnalysisSourceKind
  timeline: RiskPoint[]
  events: EvidenceEvent[]
}

export interface AnalysisJobListResponse {
  items: AnalysisJob[]
  page: number
  page_size: number
  total: number
  has_next: boolean
}

/** POST /monitoring/resume 的返回：从离线回放切回实时后的管线状态。 */
export interface PipelineControlResponse {
  state: string
  playback: boolean
  session_id: string | null
}

/**
 * GET /monitoring/preview/state（API-010）。
 *
 * `sequence`/`sha256` 指向"最近一帧送进推理的画面"：页面显示的帧号与摘要
 * 必须与之一致，才能说"看到的就是模型看到的"。
 */
export interface PreviewState {
  enabled: boolean
  available: boolean
  clients: number
  max_clients: number
  max_fps: number
  pipeline: 'realtime' | 'playback' | 'stopped'
  source: 'camera' | 'frames'
  device_index: number | null
  sequence: number | null
  source_timestamp_ms: number | null
  sha256: string | null
  bytes: number | null
  age_ms: number | null
  frames_out: number
  last_error: string | null
}

export interface SettingItem {
  key: string
  value_type: string
  value: unknown
  is_secret: boolean
  is_configured: boolean
  requires_restart: boolean
  updated_at: string | null
  applied: boolean
}

export interface SettingsResponse {
  version: number
  items: SettingItem[]
}

export interface SettingsUpdateResponse {
  version: number
  applied: Record<string, unknown>
  restart_required: string[]
  rejected: Record<string, string>
}

export interface ModelInfo {
  model_version_id: string
  model_name: string
  model_type: string
  semantic_version: string
  runtime: string
  feature_schema_version: string | null
  metrics: Record<string, unknown>
  status: string
  activated_at: string | null
}

export interface SystemStatus {
  service_version: string
  camera: { available: boolean; device_id: string; source: string }
  pipeline: {
    state: string
    processed_fps: number
    latency_p50_ms: number
    latency_p95_ms: number
    dropped_frame_rate: number
    inference_mode: InferenceMode
    degraded: boolean
    /** true 表示当前推理源是分析任务接手的离线素材，而不是实时采集。 */
    playback: boolean
  }
  database: { writable: boolean; driver: string }
  websocket: { clients: number; max_hz: number }
  model: { loaded: boolean; bundle_version: string; active_models: string[] }
}

export interface ApiErrorPayload {
  code: string
  message: string
  details: Array<{ field?: string; reason?: string }>
  request_id: string | null
}

/** 从分析任务跳转到事件/统计页时携带的会话范围。 */
export interface SessionFocus {
  sessionId: string
  origin: string
}

/** WebSocket envelope（WS-001） */
export interface WsMessage<T = unknown> {
  type: string
  schema_version: string
  sequence: number
  sent_at: string
  data: T
}

export interface EventChangedPayload {
  action: string
  event: EventOut
}
