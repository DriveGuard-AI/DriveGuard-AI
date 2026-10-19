/**
 * 实时预览（API-010）。
 *
 * 目的只有一个：让用户确认"模型此刻到底看到了什么"。
 *
 * - 画面来自后端 `PreviewTap` 记录的、**已送进推理的同一份 JPEG 字节**；
 *   分片头里的 `X-Seq` / `X-Sha256` 直接显示在界面上，便于核对；
 * - 不能直接用 `<img src>`：契约要求 Token 走 Authorization Header，
 *   所以这里 fetch 流并手工解析 multipart；
 * - 预览默认关闭（`preview.enabled=false`），由管理员显式打开；
 * - 离线回放期间不显示：那时的输入是素材，不是摄像头。
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { ApiError, api, readPreviewStream } from '../api/client'
import type { PreviewState } from '../api/types'
import { Notice } from './common'

interface FrameView {
  url: string
  seq: number
  timestampMs: number
  sha256: string
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 403) return '预览开关未打开，或当前 Token 没有管理员权限。'
    if (error.status === 409) return '当前没有活动监测会话，无法预览。'
    if (error.status === 503) return error.message
    return `${error.message}（${error.code}）`
  }
  if (error instanceof Error) return error.message
  return String(error)
}

export function LivePreview({
  token,
  playback,
  pipelineState,
}: {
  token: string | null
  playback: boolean
  pipelineState: string
}) {
  const [state, setState] = useState<PreviewState | null>(null)
  const [frame, setFrame] = useState<FrameView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [enabling, setEnabling] = useState(false)
  const [retryKey, setRetryKey] = useState(0)
  const [streaming, setStreaming] = useState(false)
  const frameRef = useRef<FrameView | null>(null)

  const refresh = useCallback(async () => {
    try {
      setState(await api.previewState(token))
    } catch {
      // 状态查询失败不覆盖已有的流错误提示，避免刷屏
    }
  }, [token])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 2000)
    return () => window.clearInterval(timer)
  }, [refresh])

  const previewing =
    Boolean(state?.enabled) && state?.pipeline === 'realtime' && !playback && pipelineState !== 'stopped'

  // 建立/重建 MJPEG 流。依赖只用原始值：state 每次轮询都是新对象，不能进依赖。
  useEffect(() => {
    if (!previewing) {
      setStreaming(false)
      return
    }
    const controller = new AbortController()
    let cancelled = false
    setError(null)
    setStreaming(true)

    readPreviewStream(token, controller.signal, (message) => {
      if (cancelled) return
      const url = URL.createObjectURL(message.jpeg)
      // 只保留最新一帧：旧 URL 立即回收，慢渲染不积压
      if (frameRef.current) URL.revokeObjectURL(frameRef.current.url)
      const next = { url, seq: message.seq, timestampMs: message.timestampMs, sha256: message.sha256 }
      frameRef.current = next
      setFrame(next)
    })
      .then(() => {
        if (cancelled) return
        setStreaming(false)
        // 服务端空闲超时会正常收流；稍后自动重连
        window.setTimeout(() => {
          if (!cancelled) setRetryKey((value) => value + 1)
        }, 2000)
      })
      .catch((reason: unknown) => {
        if (cancelled || controller.signal.aborted) return
        setStreaming(false)
        setError(describeError(reason))
        // 摄像头预热/会话未就绪属于暂时状态，自动重试；权限类错误不重试
        if (reason instanceof ApiError && (reason.status === 503 || reason.status === 409)) {
          window.setTimeout(() => {
            if (!cancelled) setRetryKey((value) => value + 1)
          }, 3000)
        }
      })

    return () => {
      cancelled = true
      controller.abort()
      setStreaming(false)
      if (frameRef.current) {
        URL.revokeObjectURL(frameRef.current.url)
        frameRef.current = null
      }
      setFrame(null)
    }
  }, [previewing, token, retryKey])

  const enablePreview = async () => {
    setEnabling(true)
    setError(null)
    try {
      await api.enablePreview(token)
      await refresh()
    } catch (reason) {
      setError(describeError(reason))
    } finally {
      setEnabling(false)
    }
  }

  if (playback) return null

  if (!state) return null

  if (!state.enabled) {
    return (
      <Notice kind="info" title="实时预览默认关闭">
        预览只读取内存中的最新一帧、不落盘，但会占用摄像头与带宽，因此需要显式开启
        （设置项 <code>preview.enabled</code>）。
        <div className="preview-actions">
          <button type="button" onClick={() => void enablePreview()} disabled={enabling}>
            {enabling ? '正在开启…' : '开启预览'}
          </button>
          {streaming ? <span className="muted small">连接中…</span> : null}
        </div>
        {error ? <div className="warn-text small">{error}</div> : null}
      </Notice>
    )
  }

  if (!previewing) {
    return (
      <Notice kind="warn" title="当前无法预览">
        {pipelineState === 'stopped'
          ? '推理管线已停止，没有实时画面。'
          : '离线回放期间不提供实时预览，请在分析回放页查看素材。'}
      </Notice>
    )
  }

  return (
    <section className="card live-preview">
      <div className="preview-head">
        <h3>摄像头实时画面</h3>
        <span className="muted small">
          来源 {state.source === 'camera' ? `摄像头 #${state.device_index ?? 0}` : 'JPEG 序列'} · 输出上限{' '}
          {state.max_fps} FPS · 连接 {state.clients}/{state.max_clients}
        </span>
      </div>

      {frame ? (
        <>
          <img className="preview-image" src={frame.url} alt="实时摄像头画面" />
          <div className="preview-meta">
            <span className="chip">帧 #{frame.seq}</span>
            <span className="chip">t = {frame.timestampMs} ms</span>
            <span className="chip mono" title={frame.sha256}>
              sha256 {frame.sha256.slice(0, 12)}…
            </span>
          </div>
          <p className="muted small">
            画面与推理输入是<strong>同一份 JPEG 字节</strong>（摘要一致）；这里显示的就是模型刚刚处理的那一帧。
          </p>
        </>
      ) : (
        <div className="empty-state">
          {streaming ? '正在等待第一帧…' : '尚未收到画面。'}
          {state.last_error ? <div className="warn-text small">{state.last_error}</div> : null}
        </div>
      )}

      {error ? (
        <Notice kind="warn" title="预览已中断">
          {error}
          <div className="preview-actions">
            <button type="button" className="ghost" onClick={() => setRetryKey((value) => value + 1)}>
              重新连接
            </button>
          </div>
        </Notice>
      ) : null}

      {state.sequence != null && frame && state.sequence !== frame.seq ? (
        <p className="muted small">
          推理已处理到帧 #{state.sequence}，预览显示 #{frame.seq}（输出限速时会落后几帧）。
        </p>
      ) : null}
    </section>
  )
}
