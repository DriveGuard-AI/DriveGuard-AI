import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { SystemStatus } from './api/types'
import { StatusDot } from './components/common'
import { useMonitoringSocket } from './hooks/useMonitoringSocket'
import { EventsPage } from './pages/EventsPage'
import { MonitoringPage } from './pages/MonitoringPage'
import { SettingsPage } from './pages/SettingsPage'
import { StatisticsPage } from './pages/StatisticsPage'

type TabKey = 'monitoring' | 'events' | 'statistics' | 'settings'

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: 'monitoring', label: '实时监控' },
  { key: 'events', label: '事件历史' },
  { key: 'statistics', label: '统计分析' },
  { key: 'settings', label: '系统设置' },
]

const TOKEN_STORAGE_KEY = 'dgb.token'

export default function App() {
  const [tab, setTab] = useState<TabKey>('monitoring')
  const [token, setToken] = useState<string | null>(() => window.localStorage.getItem(TOKEN_STORAGE_KEY))
  const [status, setStatus] = useState<SystemStatus | null>(null)

  const feed = useMonitoringSocket(token)

  useEffect(() => {
    if (token) {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, token)
    } else {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY)
    }
  }, [token])

  const refreshStatus = useCallback(async () => {
    try {
      setStatus(await api.systemStatus(token))
    } catch {
      setStatus(null)
    }
  }, [token])

  useEffect(() => {
    void refreshStatus()
    const timer = window.setInterval(() => void refreshStatus(), 10000)
    return () => window.clearInterval(timer)
  }, [refreshStatus])

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">DG</span>
          <div>
            <div className="brand-title">DriveGuard AI</div>
            <div className="brand-sub">驾安智护 · 危险驾驶安全检测</div>
          </div>
        </div>

        <nav className="tabs">
          {TABS.map((item) => (
            <button
              key={item.key}
              type="button"
              className={tab === item.key ? 'tab active' : 'tab'}
              onClick={() => setTab(item.key)}
            >
              {item.label}
            </button>
          ))}
        </nav>

        <div className="header-meta">
          <StatusDot state={feed.connection} />
          <label className="token-field">
            管理 Token
            <input
              type="password"
              placeholder="Bearer Token（可选）"
              value={token ?? ''}
              onChange={(event) => setToken(event.target.value || null)}
            />
          </label>
        </div>
      </header>

      {status ? (
        <div className="status-bar">
          <span>
            推理模式 <b className={status.pipeline.inference_mode === 'fake' ? 'warn-text' : ''}>{status.pipeline.inference_mode}</b>
          </span>
          <span>管线 {status.pipeline.state}</span>
          <span>处理 {status.pipeline.processed_fps.toFixed(1)} FPS</span>
          <span>
            延迟 P50 {status.pipeline.latency_p50_ms.toFixed(1)} ms / P95 {status.pipeline.latency_p95_ms.toFixed(1)} ms
          </span>
          <span>WS 客户端 {status.websocket.clients}</span>
          <span>数据库 {status.database.writable ? '可写' : '不可写'}</span>
          <span>模型 {status.model.loaded ? '已加载' : '未加载'}</span>
          <span>服务 v{status.service_version}</span>
        </div>
      ) : (
        <div className="status-bar">
          <span className="warn-text">未能获取 /api/v1/system/status（可能需要 Token，或后端未启动）</span>
        </div>
      )}

      <main className="app-main">
        {tab === 'monitoring' ? <MonitoringPage feed={feed} /> : null}
        {tab === 'events' ? <EventsPage token={token} /> : null}
        {tab === 'statistics' ? <StatisticsPage token={token} /> : null}
        {tab === 'settings' ? <SettingsPage token={token} /> : null}
      </main>

      <footer className="app-footer">
        模型识别与判断暂未实现，风险数据来自脚本化回放（inference_mode = fake），仅用于前后端联调。
      </footer>
    </div>
  )
}
