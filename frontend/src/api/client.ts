/**
 * REST 客户端。
 *
 * 约定（`docs/api/API_DESIGN.md` 第 2 节）：
 * - 基础路径 `/api/v1`
 * - 鉴权使用 Bearer Token
 * - 统一错误结构 `code / message / details / request_id`
 */

import type {
  AnalysisEvidence,
  AnalysisJob,
  AnalysisJobListResponse,
  AnalysisJobStatus,
  ApiErrorPayload,
  EventListResponse,
  ModelInfo,
  PipelineControlResponse,
  PreviewState,
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

  createAnalysisJob: (token: string | null, sourcePath: string, force = false) =>
    request<AnalysisJob>('/analysis/jobs', token, {
      method: 'POST',
      body: JSON.stringify({ source_kind: 'frames', source_path: sourcePath, force }),
    }),

  /** 从离线回放切回实时采集；有分析任务在跑时后端返回 409。 */
  resumeMonitoring: (token: string | null) =>
    request<PipelineControlResponse>('/monitoring/resume', token, { method: 'POST' }),

  /** 预览开关、连接数与最近一帧的序号/摘要。 */
  previewState: (token: string | null) => request<PreviewState>('/monitoring/preview/state', token),

  /** 打开实时预览开关（管理权限）。返回新版本号与生效项。 */
  enablePreview: async (token: string | null) => {
    const current = await request<SettingsResponse>('/settings', token)
    return request<SettingsUpdateResponse>('/settings', token, {
      method: 'PUT',
      body: JSON.stringify({ version: current.version, changes: { 'preview.enabled': true } }),
    })
  },

  analysisEvidence: (token: string | null, jobId: string) =>
    request<AnalysisEvidence>(`/analysis/jobs/${jobId}/evidence`, token),

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
  options: {
    step_ms?: number
    size?: string
    quality?: number
    retain_source?: boolean
    force?: boolean
  } = {},
  onProgress?: (percent: number) => void,
): Promise<AnalysisJob> {
  const form = new FormData()
  form.append('file', file, file.name)
  if (options.step_ms != null) form.append('step_ms', String(options.step_ms))
  if (options.size) form.append('size', options.size)
  if (options.quality != null) form.append('quality', String(options.quality))
  if (options.retain_source != null) form.append('retain_source', String(options.retain_source))
  if (options.force != null) form.append('force', String(options.force))

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

/**
 * 媒体 URL。
 *
 * `<video>` / `<img>` 不能自定义请求头，凭证只能走 query（与 WS-001 同一取舍）。
 */
export function analysisSourceUrl(token: string | null, jobId: string): string {
  return `${API_BASE}/analysis/jobs/${jobId}/source${token ? `?access_token=${encodeURIComponent(token)}` : ''}`
}

export function analysisFrameUrl(token: string | null, jobId: string, index: number): string {
  return `${API_BASE}/analysis/jobs/${jobId}/frames/${index}${token ? `?access_token=${encodeURIComponent(token)}` : ''}`
}

/** 预览分片：字节与推理输入完全一致，`sha256` 可用于核对。 */
export interface PreviewFrameMessage {
  seq: number
  timestampMs: number
  sha256: string
  jpeg: Blob
}

const CRLFCRLF = new Uint8Array([13, 10, 13, 10])
const textDecoder = new TextDecoder('ascii')

function concatBytes(left: Uint8Array, right: Uint8Array): Uint8Array {
  const merged = new Uint8Array(left.length + right.length)
  merged.set(left, 0)
  merged.set(right, left.length)
  return merged
}

function indexOfBytes(haystack: Uint8Array, needle: Uint8Array, from: number): number {
  outer: for (let i = from; i <= haystack.length - needle.length; i += 1) {
    for (let j = 0; j < needle.length; j += 1) {
      if (haystack[i + j] !== needle[j]) continue outer
    }
    return i
  }
  return -1
}

/** 从缓冲里尽量多地解析出完整分片，返回剩余未消费字节。 */
function drainPreviewParts(
  buffer: Uint8Array,
  onFrame: (frame: PreviewFrameMessage) => void,
): Uint8Array {
  let offset = 0
  while (true) {
    const headerEnd = indexOfBytes(buffer, CRLFCRLF, offset)
    if (headerEnd < 0) break
    const headers = textDecoder.decode(buffer.subarray(offset, headerEnd))
    const lengthMatch = /content-length:\s*(\d+)/i.exec(headers)
    if (!lengthMatch) {
      // 结束标记 `--frame--` 等无长度分片，直接跳过
      offset = headerEnd + CRLFCRLF.length
      continue
    }
    const length = Number(lengthMatch[1])
    const bodyStart = headerEnd + CRLFCRLF.length
    // 分片尾部还有一个 CRLF
    if (buffer.length < bodyStart + length + 2) break
    const jpeg = buffer.slice(bodyStart, bodyStart + length)
    onFrame({
      seq: Number(/x-seq:\s*(\d+)/i.exec(headers)?.[1] ?? -1),
      timestampMs: Number(/x-timestamp-ms:\s*(\d+)/i.exec(headers)?.[1] ?? -1),
      sha256: /x-sha256:\s*([0-9a-f]+)/i.exec(headers)?.[1] ?? '',
      jpeg: new Blob([jpeg], { type: 'image/jpeg' }),
    })
    offset = bodyStart + length + 2
  }
  return offset > 0 ? buffer.slice(offset) : buffer
}

/**
 * 读取 MJPEG 预览流（API-010）。
 *
 * 文档要求预览用 Authorization Header、不把 Token 放进 URL，因此浏览器不能
 * 直接用 `<img src>`，必须 fetch + 手动解析 multipart。收到一帧就回调一帧，
 * 上层只保留最新画面（旧帧直接丢弃，慢渲染不会反压服务端）。
 */
export async function readPreviewStream(
  token: string | null,
  signal: AbortSignal,
  onFrame: (frame: PreviewFrameMessage) => void,
): Promise<void> {
  const headers = new Headers({ Accept: 'multipart/x-mixed-replace' })
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${API_BASE}/monitoring/preview`, { headers, signal })
  if (!response.ok) {
    let payload: ApiErrorPayload = {
      code: 'HTTP_ERROR',
      message: `预览请求失败（${response.status}）`,
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

  const body = response.body
  if (!body) throw new Error('当前浏览器不支持流式响应')
  const reader = body.getReader()
  let buffer: Uint8Array = new Uint8Array(0)
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (value) {
        buffer = drainPreviewParts(concatBytes(buffer, value), onFrame)
      }
    }
  } finally {
    void reader.cancel().catch(() => undefined)
  }
}

