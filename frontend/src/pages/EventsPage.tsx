/** 事件历史（FR-006）。 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, api, type EventQuery } from '../api/client'
import type { EventListResponse, EventOut, SessionFocus } from '../api/types'
import { EmptyState, Notice, formatDuration, formatLocalTime } from '../components/common'

const EVENT_TYPES = ['fatigue', 'phone_use', 'drinking', 'distraction', 'other']
const RISK_LEVELS = ['SAFE', 'WARNING', 'DANGER']
const STATUSES = ['active', 'ended', 'discarded']

function toIso(value: string): string | undefined {
  if (!value) return undefined
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? undefined : date.toISOString()
}

export function EventsPage({
  token,
  focus,
  onClearFocus,
}: {
  token: string | null
  focus: SessionFocus | null
  onClearFocus: () => void
}) {
  const [filters, setFilters] = useState({
    event_type: '',
    risk_level: '',
    status: '',
    started_from: '',
    started_to: '',
    session_id: '',
  })
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [data, setData] = useState<EventListResponse | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [selected, setSelected] = useState<EventOut | null>(null)
  const [loading, setLoading] = useState(false)

  const query = useMemo<EventQuery>(
    () => ({
      page,
      page_size: pageSize,
      event_type: filters.event_type || undefined,
      risk_level: filters.risk_level || undefined,
      status: filters.status || undefined,
      started_from: toIso(filters.started_from),
      started_to: toIso(filters.started_to),
      session_id: filters.session_id.trim() || undefined,
    }),
    [filters, page, pageSize],
  )

  // 从「离线分析」跳过来时只看这一轮的结果；用户仍可手动改回全部会话。
  useEffect(() => {
    if (!focus) return
    setFilters((previous) => ({ ...previous, session_id: focus.sessionId }))
    setPage(1)
  }, [focus])

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const result = await api.listEvents(token, query)
      setData(result)
      setError(null)
    } catch (caught) {
      setData(null)
      setError(caught as Error)
    } finally {
      setLoading(false)
    }
  }, [query, token])

  useEffect(() => {
    void load()
  }, [load])

  const updateFilter = (key: keyof typeof filters, value: string) => {
    setPage(1)
    setFilters((previous) => ({ ...previous, [key]: value }))
  }

  return (
    <div className="page">
      <div className="page-head">
        <h2>事件历史</h2>
        <div className="page-head-meta muted small">
          排序固定为 started_at DESC；分页上限 100（超出将被后端拒绝）
        </div>
      </div>

      <section className="card">
        <div className="filter-grid">
          <label>
            事件类型
            <select value={filters.event_type} onChange={(event) => updateFilter('event_type', event.target.value)}>
              <option value="">全部</option>
              {EVENT_TYPES.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>
          <label>
            风险等级
            <select value={filters.risk_level} onChange={(event) => updateFilter('risk_level', event.target.value)}>
              <option value="">全部</option>
              {RISK_LEVELS.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </label>
          <label>
            状态
            <select value={filters.status} onChange={(event) => updateFilter('status', event.target.value)}>
              <option value="">全部</option>
              {STATUSES.map((status) => (
                <option key={status} value={status}>
                  {status}
                </option>
              ))}
            </select>
          </label>
          <label>
            开始时间（本地时区，提交时转为 UTC）
            <input
              type="datetime-local"
              value={filters.started_from}
              onChange={(event) => updateFilter('started_from', event.target.value)}
            />
          </label>
          <label>
            结束时间
            <input
              type="datetime-local"
              value={filters.started_to}
              onChange={(event) => updateFilter('started_to', event.target.value)}
            />
          </label>
          <label>
            会话 ID（可选）
            <input
              type="text"
              className="mono"
              placeholder="UUID"
              value={filters.session_id}
              onChange={(event) => updateFilter('session_id', event.target.value)}
            />
          </label>
          <div className="filter-actions">
            <button type="button" onClick={() => void load()} disabled={loading}>
              {loading ? '加载中…' : '刷新'}
            </button>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setFilters({
                  event_type: '',
                  risk_level: '',
                  status: '',
                  started_from: '',
                  started_to: '',
                  session_id: '',
                })
                setPage(1)
                onClearFocus()
              }}
            >
              重置
            </button>
          </div>
        </div>
      </section>

      {filters.session_id.trim() ? (
        <Notice kind="info" title={`仅显示会话 ${filters.session_id.trim().slice(0, 8)}… 的事件`}>
          {focus && focus.sessionId === filters.session_id.trim() ? `来源：${focus.origin}。` : ''}
          每跑一次离线分析都会产生新会话，按会话看才能对上某一轮素材。
          <div style={{ marginTop: 6 }}>
            <button type="button" className="ghost" onClick={() => updateFilter('session_id', '')}>
              显示全部会话
            </button>
          </div>
        </Notice>
      ) : null}

      {error instanceof ApiError ? (
        <Notice kind="error" title={`${error.code}：${error.message}`}>
          {error.details.length > 0 ? (
            <ul>
              {error.details.map((detail, index) => (
                <li key={index}>
                  <code>{detail.field}</code> {detail.reason}
                </li>
              ))}
            </ul>
          ) : null}
          {error.requestId ? <div className="muted small">request_id: {error.requestId}</div> : null}
        </Notice>
      ) : error ? (
        <Notice kind="error" title="请求失败">
          {error.message}
        </Notice>
      ) : null}

      <section className="card">
        {data && data.items.length === 0 ? (
          <EmptyState text="暂无事件。正常运行时会随风险进入可记录状态而创建。" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>类型</th>
                <th>最高风险</th>
                <th>状态</th>
                <th>质量</th>
                <th>开始时间</th>
                <th>时长</th>
                <th>最高置信度</th>
                <th>片段</th>
              </tr>
            </thead>
            <tbody>
              {(data?.items ?? []).map((item) => (
                <tr
                  key={item.event_id}
                  className={selected?.event_id === item.event_id ? 'selected' : undefined}
                  onClick={() => setSelected(item)}
                >
                  <td>{item.event_type}</td>
                  <td>
                    <span className={`badge risk-${item.max_risk_level.toLowerCase()}`}>{item.max_risk_level}</span>
                  </td>
                  <td>{item.status}</td>
                  <td>{item.quality}</td>
                  <td>{formatLocalTime(item.started_at)}</td>
                  <td>{formatDuration(item.duration_ms)}</td>
                  <td>{item.max_confidence.toFixed(2)}</td>
                  <td>{item.clip_available ? '可查看' : '未保存'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {data ? (
          <div className="pagination">
            <span className="muted">
              共 {data.total} 条 · 第 {data.page} 页
            </span>
            <div className="pagination-actions">
              <label>
                每页
                <select
                  value={pageSize}
                  onChange={(event) => {
                    setPage(1)
                    setPageSize(Number(event.target.value))
                  }}
                >
                  {[10, 20, 50, 100].map((size) => (
                    <option key={size} value={size}>
                      {size}
                    </option>
                  ))}
                </select>
              </label>
              <button type="button" className="ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                上一页
              </button>
              <button
                type="button"
                className="ghost"
                disabled={!data.has_next}
                onClick={() => setPage((p) => p + 1)}
              >
                下一页
              </button>
            </div>
          </div>
        ) : null}
      </section>

      {selected ? (
        <section className="card">
          <div className="page-head">
            <h3>事件详情</h3>
            <button type="button" className="ghost" onClick={() => setSelected(null)}>
              关闭
            </button>
          </div>
          <div className="detail-grid">
            <div>
              <span className="muted">event_id</span>
              <div className="mono">{selected.event_id}</div>
            </div>
            <div>
              <span className="muted">session_id</span>
              <div className="mono">{selected.session_id}</div>
            </div>
            <div>
              <span className="muted">开始 / 结束</span>
              <div>
                {formatLocalTime(selected.started_at)} → {formatLocalTime(selected.ended_at)}
              </div>
            </div>
            <div>
              <span className="muted">模型版本</span>
              <div className="mono">{selected.model_version_id ?? '—'}</div>
            </div>
          </div>
          <h4>风险分数</h4>
          <pre className="code">{JSON.stringify(selected.risk_scores, null, 2)}</pre>
          <h4>原因码</h4>
          <div className="chip-row">
            {selected.reason_codes.length === 0 ? (
              <span className="muted">无</span>
            ) : (
              selected.reason_codes.map((code) => (
                <span className="chip" key={code}>
                  {code}
                </span>
              ))
            )}
          </div>
          {!selected.clip_available ? (
            <Notice kind="info" title="默认不保存视频片段">
              原始车内视频默认不上传、不持久化；启用事件片段需显式配置保存目录与保留期（NFR-003）。
            </Notice>
          ) : null}
        </section>
      ) : null}
    </div>
  )
}
