import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { SessionFocus, SystemStatus } from './api/types'
import { StatusDot } from './components/common'
import { useMonitoringSocket } from './hooks/useMonitoringSocket'
import { AnalysisPage } from './pages/AnalysisPage'
import { EventsPage } from './pages/EventsPage'
import { MonitoringPage } from './pages/MonitoringPage'
import { SettingsPage } from './pages/SettingsPage'

type TabKey = 'monitoring' | 'analysis' | 'events' | 'settings'

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: 'monitoring', label: '实时监控' },
  { key: 'analysis', label: '离线分析' },
  { key: 'events', label: '事件历史' },
  { key: 'settings', label: '系统设置' },
]

const TOKEN_STORAGE_KEY = 'dgb.token'
const DESKTOP_BRIDGE = window.driveguard

export default function App() {
  const [tab, setTab] = useState<TabKey>('monitoring')
  // 桌面版由 Electron preload 注入本地管理 Token；Web 调试仍从 localStorage 读取。
  const [token, setToken] = useState<string | null>(
    () => DESKTOP_BRIDGE?.token || window.localStorage.getItem(TOKEN_STORAGE_KEY),
  )
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [focus, setFocus] = useState<SessionFocus | null>(null)

  const openSession = useCallback(
    (sessionId: string, target: 'events', origin: string) => {
      setFocus({ sessionId, origin })
      setTab(target)
    },
    [],
  )

  const feed = useMonitoringSocket(token)

  useEffect(() => {
    // 桌面版 Token 每次启动由主进程生成，不写浏览器存储，避免过期值残留。
    if (DESKTOP_BRIDGE) return
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

  // 从离线回放切回实时；失败时把原因留在页面上（别静默吞掉）。
  const [resumeError, setResumeError] = useState<string | null>(null)
  const resumeRealtime = useCallback(async () => {
    try {
      await api.resumeMonitoring(token)
      setResumeError(null)
    } catch (caught) {
      setResumeError(caught instanceof Error ? caught.message : '恢复实时失败')
    }
    await refreshStatus()
  }, [token, refreshStatus])

  return (
    <div className={DESKTOP_BRIDGE ? 'app desktop' : 'app'}>
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
          {DESKTOP_BRIDGE ? (
            <span className="desktop-version" title={`桌面版 ${DESKTOP_BRIDGE.version}`}>
              桌面版 v{DESKTOP_BRIDGE.version}
            </span>
          ) : (
            <label className="token-field">
              管理 Token
              <input
                type="password"
                placeholder="Bearer Token（可选）"
                value={token ?? ''}
                onChange={(event) => setToken(event.target.value || null)}
              />
            </label>
          )}
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
          {typeof status.pipeline.warmup_slots === 'number' &&
          status.pipeline.warmup_slots < (status.pipeline.warmup_total ?? 100) ? (
            <span className="warn-text">
              模型预热 {status.pipeline.warmup_slots}/{status.pipeline.warmup_total ?? 100} 帧
            </span>
          ) : null}
          <span>服务 v{status.service_version}</span>
          {status.pipeline.playback ? <span className="warn-text">当前离线回放</span> : null}
        </div>
      ) : (
        <div className="status-bar">
          <span className="warn-text">未能获取 /api/v1/system/status（可能需要 Token，或后端未启动）</span>
        </div>
      )}

      <main className="app-main">
        {tab === 'monitoring' ? (
          <MonitoringPage
            token={token}
            feed={feed}
            playback={status?.pipeline.playback ?? false}
            pipelineState={status?.pipeline.state ?? ''}
            resumeError={resumeError}
            onResume={() => void resumeRealtime()}
          />
        ) : null}
        {tab === 'analysis' ? <AnalysisPage token={token} onOpenSession={openSession} /> : null}
        {tab === 'events' ? (
          <EventsPage token={token} focus={focus} onClearFocus={() => setFocus(null)} />
        ) : null}
        {tab === 'settings' ? <SettingsPage token={token} /> : null}
      </main>

      <footer className="app-footer">
        {status?.pipeline.inference_mode === 'fake'
          ? '当前为演示模式：风险数据来自脚本化回放（inference_mode = fake），仅用于界面与流程验证，不是模型识别结论。'
          : '风险数据来自模型服务（inference_mode = real）；当前候选版本未经真实场景精度验收，不作为告警或安全结论。'}
      </footer>
    </div>
  )
}
