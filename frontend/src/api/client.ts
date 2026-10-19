/**
 * REST 客户端。
 *
 * 约定（`docs/api/API_DESIGN.md` 第 2 节）：
 * - 基础路径 `/api/v1`
 * - 鉴权使用 Bearer Token
 * - 统一错误结构 `code / message / details / request_id`
 */

import type {
  AnalysisJob,
  AnalysisJobListResponse,
  AnalysisJobStatus,
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

  statisticsSummary: (token: string | null, startedFrom: string, startedTo: string, sessionId?: string | null) =>
    request<StatisticsSummary>(
      `/statistics/summary${toQueryString({
        started_from: startedFrom,
        started_to: startedTo,
        session_id: sessionId || undefined,
      })}`,
      token,
    ),

  listAnalysisJobs: (token: string | null, query: { status?: string; page?: number; page_size?: number } = {}) =>
    request<AnalysisJobListResponse>(
      `/analysis/jobs${toQueryString(query as Record<string, unknown>)}`,
      token,
    ),

  getAnalysisJob: (token: string | null, jobId: string) =>
    request<AnalysisJob>(`/analysis/jobs/${jobId}`, token),

  cancelAnalysisJob: (token: string | null, jobId: string) =>
    request<AnalysisJob>(`/analysis/jobs/${jobId}`, token, { method: 'DELETE' }),

  createAnalysisJob: (token: string | null, sourcePath: string) =>
    request<AnalysisJob>('/analysis/jobs', token, {
      method: 'POST',
      body: JSON.stringify({ source_kind: 'frames', source_path: sourcePath }),
    }),

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

/**
 * 上传视频启动分析（API-014）。
 *
 * 走 XMLHttpRequest 而不是 fetch：浏览器没有上传进度事件，而抽帧要等服务端跑完，
 * 长素材下用户需要看到"传了多少"，否则接口看起来像卡住了。
 */
export function uploadAnalysisVideo(
  token: string | null,
  file: File,
  options: { step_ms?: number; size?: string; quality?: number } = {},
  onProgress?: (percent: number) => void,
): Promise<AnalysisJob> {
  const form = new FormData()
  form.append('file', file, file.name)
  if (options.step_ms != null) form.append('step_ms', String(options.step_ms))
  if (options.size) form.append('size', options.size)
  if (options.quality != null) form.append('quality', String(options.quality))

  return new Promise<AnalysisJob>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/analysis/jobs/upload`)
    xhr.setRequestHeader('Accept', 'application/json')
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`)

    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable || !onProgress) return
      onProgress(Math.min(100, Math.round((event.loaded / event.total) * 100)))
    }
    xhr.onerror = () => reject(new Error('上传失败：网络错误或连接被中断'))
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as AnalysisJob)
        } catch {
          reject(new Error('上传失败：响应不是合法 JSON'))
        }
        return
      }
      let payload: ApiErrorPayload = {
        code: 'HTTP_ERROR',
        message: `请求失败（${xhr.status}）`,
        details: [],
        request_id: xhr.getResponseHeader('X-Request-ID'),
      }
      try {
        const parsed = JSON.parse(xhr.responseText) as Partial<ApiErrorPayload>
        payload = {
          code: parsed.code ?? payload.code,
          message: parsed.message ?? payload.message,
          details: parsed.details ?? [],
          request_id: parsed.request_id ?? payload.request_id,
        }
      } catch {
        // 保留默认错误体
      }
      reject(new ApiError(xhr.status, payload))
    }
    xhr.send(form)
  })
}

export function isJobActive(status: AnalysisJobStatus): boolean {
  return status === 'queued' || status === 'running'
}
