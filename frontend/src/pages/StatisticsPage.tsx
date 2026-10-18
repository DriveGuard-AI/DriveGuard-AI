/** 统计分析（FR-008）。 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import type { EChartsOption } from 'echarts'
import { ApiError, api } from '../api/client'
import type { StatisticsSummary } from '../api/types'
import { Chart, EmptyState, MetricCard, Notice } from '../components/common'

function toInputValue(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000
  return new Date(date.getTime() - offset).toISOString().slice(0, 16)
}

export function StatisticsPage({ token }: { token: string | null }) {
  const [startedFrom, setStartedFrom] = useState(() => toInputValue(new Date(Date.now() - 7 * 86400000)))
  const [startedTo, setStartedTo] = useState(() => toInputValue(new Date()))
  const [summary, setSummary] = useState<StatisticsSummary | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const result = await api.statisticsSummary(
        token,
        new Date(startedFrom).toISOString(),
        new Date(startedTo).toISOString(),
      )
      setSummary(result)
      setError(null)
    } catch (caught) {
      setSummary(null)
      setError(caught as Error)
    } finally {
      setLoading(false)
    }
  }, [startedFrom, startedTo, token])

  useEffect(() => {
    void load()
  }, [load])

  const fatigueCount = useMemo(
    () => summary?.by_type.find((item) => item.event_type === 'fatigue')?.count ?? 0,
    [summary],
  )
  const distractionCount = useMemo(
    () => summary?.by_type.find((item) => item.event_type === 'distraction')?.count ?? 0,
    [summary],
  )
  const totalDuration = useMemo(
    () => summary?.by_type.reduce((acc, item) => acc + item.total_duration_ms, 0) ?? 0,
    [summary],
  )

  const typeOption = useMemo<EChartsOption>(
    () => ({
      tooltip: { trigger: 'axis' },
      grid: { left: 90, right: 24, top: 24, bottom: 32 },
      xAxis: { type: 'value', name: '事件数' },
      yAxis: { type: 'category', data: (summary?.by_type ?? []).map((item) => item.event_type) },
      series: [
        {
          type: 'bar',
          data: (summary?.by_type ?? []).map((item) => item.count),
          itemStyle: { color: '#1663d6', borderRadius: [0, 4, 4, 0] },
        },
      ],
    }),
    [summary],
  )

  const riskOption = useMemo<EChartsOption>(
    () => ({
      tooltip: { trigger: 'item' },
      legend: { bottom: 0 },
      series: [
        {
          type: 'pie',
          radius: ['40%', '68%'],
          data: (summary?.by_risk_level ?? []).map((item) => ({
            name: item.risk_level,
            value: item.count,
            itemStyle: {
              color: item.risk_level === 'DANGER' ? '#dc2626' : item.risk_level === 'WARNING' ? '#d97706' : '#16a34a',
            },
          })),
        },
      ],
    }),
    [summary],
  )

  const trendOption = useMemo<EChartsOption>(
    () => ({
      tooltip: { trigger: 'axis' },
      grid: { left: 48, right: 24, top: 24, bottom: 40 },
      xAxis: {
        type: 'category',
        data: (summary?.trend ?? []).map((item) => item.bucket_start.slice(5, 10)),
        axisLabel: { rotate: 0 },
      },
      yAxis: { type: 'value', name: '事件数' },
      series: [
        {
          type: 'line',
          smooth: true,
          areaStyle: { opacity: 0.12 },
          data: (summary?.trend ?? []).map((item) => item.count),
          itemStyle: { color: '#0b3b8c' },
        },
      ],
    }),
    [summary],
  )

  return (
    <div className="page">
      <div className="page-head">
        <h2>统计分析</h2>
        <div className="page-head-meta muted small">时间跨度上限 90 天；无数据时返回 0 与空数组</div>
      </div>

      <section className="card">
        <div className="filter-grid">
          <label>
            开始时间
            <input type="datetime-local" value={startedFrom} onChange={(event) => setStartedFrom(event.target.value)} />
          </label>
          <label>
            结束时间
            <input type="datetime-local" value={startedTo} onChange={(event) => setStartedTo(event.target.value)} />
          </label>
          <div className="filter-actions">
            <button type="button" onClick={() => void load()} disabled={loading}>
              {loading ? '加载中…' : '查询'}
            </button>
          </div>
        </div>
      </section>

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
        <Notice kind="error" title="查询失败">
          {error.message}
        </Notice>
      ) : null}

      <div className="metric-grid">
        <MetricCard label="危险事件总数" value={summary?.total_events ?? 0} />
        <MetricCard label="疲劳事件" value={fatigueCount} />
        <MetricCard label="分心事件" value={distractionCount} />
        <MetricCard label="累计持续时间" value={(totalDuration / 1000).toFixed(1)} unit="s" />
      </div>

      {summary && summary.total_events === 0 ? (
        <EmptyState text="所选时间范围内没有事件（空态而非异常）" />
      ) : (
        <div className="two-column">
          <section className="card">
            <h3>风险类型分布</h3>
            <Chart option={typeOption} />
          </section>
          <section className="card">
            <h3>风险等级占比</h3>
            <Chart option={riskOption} />
          </section>
          <section className="card wide">
            <h3>风险趋势（按天）</h3>
            <Chart option={trendOption} />
          </section>
        </div>
      )}
    </div>
  )
}
