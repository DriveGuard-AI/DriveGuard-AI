/**
 * 实时监控 WebSocket Hook。
 *
 * 落实 WS-001 的客户端义务：
 * - 断线使用带抖动的指数退避重连（最大 30 秒）
 * - 序列号跳跃时通过 REST 补最新完整快照（`API-009`）
 * - 消息带 `schema_version`，未知版本给出提示而不是静默吞掉
 * - 心跳超时标记连接陈旧
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, websocketUrl } from '../api/client'
import type { EventChangedPayload, RiskSnapshot, WsMessage } from '../api/types'

export type ConnectionState = 'connecting' | 'open' | 'closed'

export interface MonitoringFeed {
  connection: ConnectionState
  snapshot: RiskSnapshot | null
  events: EventChangedPayload[]
  lastMessageAt: string | null
  stale: boolean
  sequenceGap: boolean
  schemaWarning: string | null
  reconnectAttempts: number
  resync: () => Promise<void>
}

const SUPPORTED_SCHEMA = '1.0'
const MAX_EVENT_FEED = 20

export function useMonitoringSocket(token: string | null, enabled = true): MonitoringFeed {
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [snapshot, setSnapshot] = useState<RiskSnapshot | null>(null)
  const [events, setEvents] = useState<EventChangedPayload[]>([])
  const [lastMessageAt, setLastMessageAt] = useState<string | null>(null)
  const [stale, setStale] = useState(false)
  const [sequenceGap, setSequenceGap] = useState(false)
  const [schemaWarning, setSchemaWarning] = useState<string | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)

  const lastSequence = useRef<number | null>(null)

  const resync = useCallback(async () => {
    try {
      const latest = await api.snapshot(token)
      setSnapshot(latest)
      lastSequence.current = latest.sequence
      setSequenceGap(false)
    } catch {
      // 无活动会话或鉴权失败时保持现状
    }
  }, [token])

  useEffect(() => {
    if (!enabled) return
    let socket: WebSocket | null = null
    let closed = false
    let attempt = 0
    let timer: number | undefined
    let staleTimer: number | undefined

    const scheduleStaleCheck = () => {
      window.clearTimeout(staleTimer)
      staleTimer = window.setTimeout(() => setStale(true), 45000)
    }

    const connect = () => {
      if (closed) return
      setConnection('connecting')
      socket = new WebSocket(websocketUrl(token))

      socket.onopen = () => {
        attempt = 0
        setReconnectAttempts(0)
        setConnection('open')
        setStale(false)
        scheduleStaleCheck()
        // 重连后主动补齐最新状态，避免依赖服务端回放
        void resync()
      }

      socket.onmessage = (raw) => {
        scheduleStaleCheck()
        setLastMessageAt(new Date().toISOString())
        let message: WsMessage
        try {
          message = JSON.parse(raw.data as string) as WsMessage
        } catch {
          return
        }
        if (message.schema_version !== SUPPORTED_SCHEMA) {
          setSchemaWarning(`收到不兼容的消息版本 ${message.schema_version}`)
        }
        switch (message.type) {
          case 'risk.snapshot': {
            const next = message.data as RiskSnapshot
            if (lastSequence.current !== null && next.sequence > lastSequence.current + 1) {
              // 序列跳跃：不要求服务端回放，改为 REST 拉取完整状态
              setSequenceGap(true)
              void resync()
            }
            lastSequence.current = next.sequence
            setSnapshot(next)
            break
          }
          case 'event.changed': {
            // 事件变化消息不得静默丢弃
            setEvents((previous) => [message.data as EventChangedPayload, ...previous].slice(0, MAX_EVENT_FEED))
            break
          }
          case 'error': {
            setSchemaWarning(String((message.data as { message?: string }).message ?? '服务端错误'))
            break
          }
          default:
            break
        }
      }

      socket.onclose = () => {
        setConnection('closed')
        window.clearTimeout(staleTimer)
        if (closed) return
        attempt += 1
        setReconnectAttempts(attempt)
        // 带抖动的指数退避，最大 30 秒
        const base = Math.min(30000, 500 * 2 ** attempt)
        const jitter = Math.random() * 250
        timer = window.setTimeout(connect, base + jitter)
      }

      socket.onerror = () => {
        socket?.close()
      }
    }

    connect()

    return () => {
      closed = true
      window.clearTimeout(timer)
      window.clearTimeout(staleTimer)
      socket?.close()
    }
  }, [enabled, resync, token])

  const value = useMemo<MonitoringFeed>(
    () => ({
      connection,
      snapshot,
      events,
      lastMessageAt,
      stale,
      sequenceGap,
      schemaWarning,
      reconnectAttempts,
      resync,
    }),
    [connection, events, lastMessageAt, reconnectAttempts, resync, schemaWarning, sequenceGap, snapshot, stale],
  )

  return value
}
