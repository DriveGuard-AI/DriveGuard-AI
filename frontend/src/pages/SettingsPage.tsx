/** 系统设置（FR-009）。v0.1 只开放影响当前实例的基础配置。 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiError, api } from '../api/client'
import type { SettingItem, SettingsResponse } from '../api/types'
import { EmptyState, Notice, formatLocalTime } from '../components/common'

/**
 * v0.1 发布版白名单。
 *
 * 告警/风险/时序阈值目前没有接到 live 模型判决上；模型路径是历史 stub。
 * 把这些键展示出来会让用户以为“调了就能改告警”，因此只保留真正生效的项。
 * 后端接口仍保留完整注册表，v0.2 再按能力开放。
 */
const RELEASE_SETTING_KEYS = new Set([
  'camera.device_index',
  'camera.frame_width',
  'camera.frame_height',
  'preview.enabled',
  'preview.max_fps',
  'preview.max_clients',
  'storage.event_retention_days',
])

function visibleItems(items: SettingItem[]): SettingItem[] {
  return items.filter((item) => RELEASE_SETTING_KEYS.has(item.key))
}

function toDraft(items: SettingItem[]): Record<string, unknown> {
  const draft: Record<string, unknown> = {}
  items.forEach((item) => {
    draft[item.key] = item.is_secret ? '' : item.value
  })
  return draft
}

function valuesEqual(a: unknown, b: unknown): boolean {
  if (typeof a === 'number' || typeof b === 'number') {
    return Number(a) === Number(b)
  }
  return a === b
}

export function SettingsPage({ token }: { token: string | null }) {
  const [settings, setSettings] = useState<SettingsResponse | null>(null)
  const [draft, setDraft] = useState<Record<string, unknown>>({})
  const [error, setError] = useState<Error | null>(null)
  const [saved, setSaved] = useState<{ version: number; restartRequired: string[] } | null>(null)
  const [busy, setBusy] = useState(false)

  const items = useMemo(() => (settings ? visibleItems(settings.items) : []), [settings])

  const load = useCallback(async () => {
    try {
      const result = await api.getSettings(token)
      setSettings(result)
      setDraft(toDraft(visibleItems(result.items)))
      setError(null)
    } catch (caught) {
      setSettings(null)
      setError(caught as Error)
    }
  }, [token])

  useEffect(() => {
    void load()
  }, [load])

  const changes = useMemo(() => {
    if (!settings) return {}
    const diff: Record<string, unknown> = {}
    items.forEach((item) => {
      const next = draft[item.key]
      if (item.is_secret) {
        if (typeof next === 'string' && next.trim().length > 0) {
          diff[item.key] = next
        }
        return
      }
      if (item.value_type === 'integer' || item.value_type === 'number') {
        if (next === '' || next === null || next === undefined) return
        const numeric = Number(next)
        if (!Number.isNaN(numeric) && !valuesEqual(numeric, item.value)) {
          diff[item.key] = numeric
        }
        return
      }
      if (!valuesEqual(next, item.value)) {
        diff[item.key] = next
      }
    })
    return diff
  }, [draft, items, settings])

  const save = async () => {
    if (!settings) return
    setBusy(true)
    setSaved(null)
    try {
      const result = await api.updateSettings(token, settings.version, changes)
      setSaved({ version: result.version, restartRequired: result.restart_required })
      setError(null)
      await load()
    } catch (caught) {
      setError(caught as Error)
      if (caught instanceof ApiError && caught.code === 'CONFIG_VERSION_CONFLICT') {
        // 乐观锁冲突：重新读取最新版本，避免用旧版本覆盖
        await load()
      }
    } finally {
      setBusy(false)
    }
  }

  const renderInput = (item: SettingItem) => {
    const value = draft[item.key]
    if (item.is_secret) {
      return (
        <input
          type="password"
          placeholder={item.is_configured ? '已配置（留空表示不修改）' : '未配置'}
          value={typeof value === 'string' ? value : ''}
          onChange={(event) => setDraft((previous) => ({ ...previous, [item.key]: event.target.value }))}
        />
      )
    }
    if (item.value_type === 'boolean') {
      return (
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => setDraft((previous) => ({ ...previous, [item.key]: event.target.checked }))}
        />
      )
    }
    if (item.value_type === 'integer' || item.value_type === 'number') {
      return (
        <input
          type="number"
          step={item.value_type === 'integer' ? 1 : 0.05}
          value={value === null || value === undefined ? '' : String(value)}
          onChange={(event) => setDraft((previous) => ({ ...previous, [item.key]: event.target.value }))}
        />
      )
    }
    return (
      <input
        type="text"
        value={value === null || value === undefined ? '' : String(value)}
        onChange={(event) => setDraft((previous) => ({ ...previous, [item.key]: event.target.value }))}
      />
    )
  }

  const dirtyCount = Object.keys(changes).length

  return (
    <div className="page">
      <div className="page-head">
        <h2>系统设置</h2>
        <div className="page-head-meta muted small">
          {settings ? `配置版本 v${settings.version}` : '—'} · 更新使用乐观锁，冲突返回 409
        </div>
      </div>

      <Notice kind="info" title="v0.1 只开放基础配置">
        告警延迟、风险阈值、时序窗口和模型路径尚未接入当前 live 判决，发布版不展示。
        摄像头分辨率等标注“需重启”的项在下次启动后生效。
      </Notice>

      {error ? (
        error instanceof ApiError ? (
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
            {error.code === 'CONFIG_VERSION_CONFLICT' ? (
              <div>配置已被其他会话更新，已重新载入最新版本，请确认后再次提交。</div>
            ) : null}
            {error.code === 'AUTH_REQUIRED' ? (
              <div>桌面版会自动注入本地 Token；Web 调试时请在右上角填入管理 Token。</div>
            ) : null}
          </Notice>
        ) : (
          <Notice kind="error" title="加载配置失败">
            {error.message}
          </Notice>
        )
      ) : null}

      {saved ? (
        <Notice kind="info" title={`已保存，当前版本 v${saved.version}`}>
          {saved.restartRequired.length > 0 ? (
            <div>需重启生效：{saved.restartRequired.join('、')}</div>
          ) : (
            <div>无需重启即可生效。</div>
          )}
        </Notice>
      ) : null}

      <section className="card">
        <div className="page-head">
          <h3>配置项</h3>
          <button type="button" onClick={() => void save()} disabled={busy || dirtyCount === 0}>
            {busy ? '保存中…' : `保存更改${dirtyCount > 0 ? `（${dirtyCount}）` : ''}`}
          </button>
        </div>

        {!settings ? (
          <EmptyState text="暂无配置数据" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: '26%' }}>键</th>
                <th style={{ width: '32%' }}>值</th>
                <th>类型</th>
                <th>生效</th>
                <th>更新时间</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.key} className={item.key in changes ? 'selected' : undefined}>
                  <td className="mono">{item.key}</td>
                  <td>{renderInput(item)}</td>
                  <td>
                    {item.value_type}
                    {item.is_secret ? ' · secret' : ''}
                  </td>
                  <td>{item.requires_restart ? '需重启' : '即时'}</td>
                  <td className="muted small">{formatLocalTime(item.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {window.driveguard ? (
        <section className="card">
          <h3>桌面版数据</h3>
          <p className="muted small">
            数据库、上传素材和后端日志保存在当前 Windows 用户的数据目录；卸载默认保留。
          </p>
          <div className="preview-actions">
            <button type="button" onClick={() => void window.driveguard?.openDataDir()}>
              打开数据目录
            </button>
            <button type="button" className="ghost" onClick={() => void window.driveguard?.openLogs()}>
              查看后端日志
            </button>
          </div>
        </section>
      ) : null}
    </div>
  )
}
