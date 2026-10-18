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
  reason_codes: string[]
  inference_mode: InferenceMode
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
  total_events: number
  by_type: CountByType[]
  by_risk_level: CountByRisk[]
  trend: TrendBucket[]
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
