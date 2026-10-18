/**
 * REST 客户端。
 *
 * 约定（`docs/api/API_DESIGN.md` 第 2 节）：
 * - 基础路径 `/api/v1`
 * - 鉴权使用 Bearer Token
 * - 统一错误结构 `code / message / details / request_id`
 */

import type {
  ApiErrorPayload,
  EventListResponse,
  ModelInfo,
  RiskSnapshot,
  SettingsResponse,
  SettingsUpdateResponse,
  StatisticsSummary,
  SystemStatus,
} from './types'

const API_BASE = '/api/v1'

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details: Array<{ field?: string; reason?: string }>
  readonly requestId: string | null

  constructor(status: number, payload: ApiErrorPayload) {
    super(payload.message || `请求失败（${status}）`)
    this.name = 'ApiError'
    this.status = status
    this.code = payload.code
    this.details = payload.details ?? []
    this.requestId = payload.request_id ?? null
  }
}

export interface EventQuery {
  page?: number
  page_size?: number
  event_type?: string
  risk_level?: string
  status?: string
  started_from?: string
  started_to?: string
  session_id?: string
}

async function request<T>(path: string, token: string | null, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  if (init?.body) {
    headers.set('Content-Type', 'application/json')
  }
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers })

  if (!response.ok) {
    let payload: ApiErrorPayload = {
      code: 'HTTP_ERROR',
      message: `请求失败（${response.status}）`,
      details: [],
      request_id: response.headers.get('X-Request-ID'),
    }
    try {
      const parsed = (await response.json()) as Partial<ApiErrorPayload>
      payload = {
        code: parsed.code ?? payload.code,
        message: parsed.message ?? payload.message,
        details: parsed.details ?? [],
        request_id: parsed.request_id ?? payload.request_id,
      }
    } catch {
      // 保留默认错误体
    }
    throw new ApiError(response.status, payload)
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function toQueryString(query: Record<string, unknown>): string {
  const params = new URLSearchParams()
  Object.entries(query).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    params.set(key, String(value))
  })
  const text = params.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: (token: string | null) => request<{ status: string; service_version: string; time: string }>('/health', token),

  systemStatus: (token: string | null) => request<SystemStatus>('/system/status', token),

  snapshot: (token: string | null) => request<RiskSnapshot>('/monitoring/snapshot', token),

  listEvents: (token: string | null, query: EventQuery = {}) =>
    request<EventListResponse>(`/events${toQueryString(query as Record<string, unknown>)}`, token),

  statisticsSummary: (token: string | null, startedFrom: string, startedTo: string) =>
    request<StatisticsSummary>(
      `/statistics/summary${toQueryString({ started_from: startedFrom, started_to: startedTo })}`,
      token,
    ),

  getSettings: (token: string | null) => request<SettingsResponse>('/settings', token),

  updateSettings: (token: string | null, version: number, changes: Record<string, unknown>) =>
    request<SettingsUpdateResponse>('/settings', token, {
      method: 'PUT',
      body: JSON.stringify({ version, changes }),
    }),

  listModels: (token: string | null) => request<ModelInfo[]>('/models', token),
}

export function websocketUrl(token: string | null): string {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const query = token ? `?access_token=${encodeURIComponent(token)}` : ''
  return `${protocol}://${window.location.host}/ws/v1/monitoring${query}`
}
