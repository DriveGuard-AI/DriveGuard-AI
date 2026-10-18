/** 通用展示组件。 */

import { useEffect, useRef, type ReactNode } from 'react'
import * as echarts from 'echarts'
import type { RiskLevel } from '../api/types'

const RISK_LABEL: Record<RiskLevel, string> = {
  SAFE: '安全',
  WARNING: '警告',
  DANGER: '危险',
}

export function RiskBadge({ level }: { level: RiskLevel }) {
  return (
    <span className={`badge risk-${level.toLowerCase()}`}>
      {level} · {RISK_LABEL[level]}
    </span>
  )
}

export function QualityBadge({ quality }: { quality: string }) {
  const label = quality === 'good' ? '输入良好' : quality === 'degraded' ? '输入降级' : '输入未知'
  return <span className={`badge quality-${quality}`}>{label}</span>
}

export function StatusDot({ state }: { state: 'connecting' | 'open' | 'closed' }) {
  const text = state === 'open' ? '实时已连接' : state === 'connecting' ? '连接中…' : '已断开，重连中'
  return (
    <span className={`status-dot status-${state}`}>
      <i />
      {text}
    </span>
  )
}

export function MetricCard({
  label,
  value,
  unit,
  hint,
}: {
  label: string
  value: string | number
  unit?: string
  hint?: string
}) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">
        {value}
        {unit ? <small>{unit}</small> : null}
      </div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  )
}

export function EmptyState({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>
}

export function Notice({
  kind = 'info',
  title,
  children,
}: {
  kind?: 'info' | 'warn' | 'error'
  title: string
  children?: ReactNode
}) {
  return (
    <div className={`notice notice-${kind}`}>
      <strong>{title}</strong>
      {children ? <div className="notice-body">{children}</div> : null}
    </div>
  )
}

export function Chart({ option, height = 280 }: { option: echarts.EChartsOption; height?: number }) {
  const container = useRef<HTMLDivElement>(null)
  const instance = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!container.current) return
    instance.current = echarts.init(container.current)
    const handleResize = () => instance.current?.resize()
    window.addEventListener('resize', handleResize)
    return () => {
      window.removeEventListener('resize', handleResize)
      instance.current?.dispose()
      instance.current = null
    }
  }, [])

  useEffect(() => {
    instance.current?.setOption(option, true)
  }, [option])

  return <div ref={container} style={{ height }} />
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) return `${milliseconds} ms`
  const seconds = milliseconds / 1000
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes} min ${Math.round(seconds % 60)} s`
}

export function formatLocalTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  // 存储与传输为 UTC，展示按浏览器本地时区（业务规则 4）
  return date.toLocaleString()
}
