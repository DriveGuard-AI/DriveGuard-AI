/**
 * 回放与证据：把「第几帧出问题」直接摆到画面上。
 *
 * 两种播放源：
 * - 任务保留了原视频 → `<video>`，可拖动、流畅；
 * - 只留了抽帧 → 逐帧取 JPEG，帧号与检测结果一一对应（浏览器缓存 + 预取）。
 *
 * 风险曲线来自服务端逐 tick 记录并降采样后的点；事件区间由服务端按
 * 「会话起点 + 素材偏移」换算成帧号，前端只负责定位。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api, analysisFrameUrl, analysisSourceUrl } from '../api/client'
import type { AnalysisEvidence, AnalysisJob, RiskPoint } from '../api/types'
import { Notice, formatNumber } from '../components/common'

const RISK_COLOR: Record<string, string> = {
  SAFE: '#16a34a',
  WARNING: '#d97706',
  DANGER: '#dc2626',
}
const RISK_ORDER = ['SAFE', 'WARNING', 'DANGER']

function formatClock(ms: number): string {
  const tenths = Math.max(0, Math.floor(ms / 100))
  const minutes = Math.floor(tenths / 600)
  const seconds = Math.floor((tenths % 600) / 10)
  return `${minutes}:${String(seconds).padStart(2, '0')}.${tenths % 10}`
}

function pointAt(points: RiskPoint[], ms: number): RiskPoint | null {
  // 点按 ms 升序；找最后一个 ms <= 目标的点。
  let low = 0
  let high = points.length - 1
  let found: RiskPoint | null = null
  while (low <= high) {
    const middle = (low + high) >> 1
    if (points[middle].ms <= ms) {
      found = points[middle]
      low = middle + 1
    } else {
      high = middle - 1
    }
  }
  return found
}

function RiskCurve({
  points,
  durationMs,
  positionMs,
  events,
  onSeek,
}: {
  points: RiskPoint[]
  durationMs: number
  positionMs: number
  events: Array<{ started_ms: number; ended_ms: number }>
  onSeek: (ms: number) => void
}) {
  const width = 1000
  const height = 56
  const total = Math.max(1, durationMs)

  const shapes = useMemo(() => {
    if (points.length === 0) return null
    const stepWidth = width / Math.max(1, points.length)
    return points.map((point, index) => (
      <rect
        key={index}
        x={(point.ms / total) * width}
        y={height - ((RISK_ORDER.indexOf(point.risk_level) + 1) / 3) * height}
        width={Math.max(stepWidth, 0.6)}
        height={((RISK_ORDER.indexOf(point.risk_level) + 1) / 3) * height}
        fill={RISK_COLOR[point.risk_level] ?? '#94a3b8'}
        opacity={0.75}
      />
    ))
  }, [points, total])

  return (
    <div className="risk-curve">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="逐帧风险曲线（点击跳转）"
        onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect()
          const ratio = (event.clientX - box.left) / Math.max(1, box.width)
          onSeek(Math.round(ratio * total))
        }}
      >
        <rect x={0} y={0} width={width} height={height} fill="#f7f9fd" />
        {events.map((event, index) => (
          <rect
            key={index}
            x={(event.started_ms / total) * width}
            y={0}
            width={Math.max(2, ((event.ended_ms - event.started_ms) / total) * width)}
            height={height}
            fill="#dc2626"
            opacity={0.12}
          />
        ))}
        {shapes}
        <line
          x1={(positionMs / total) * width}
          x2={(positionMs / total) * width}
          y1={0}
          y2={height}
          stroke="#0b3b8c"
          strokeWidth={2}
        />
      </svg>
      <div className="risk-curve-scale">
        <span>0:00</span>
        <span>{formatClock(total)}</span>
      </div>
    </div>
  )
}

export function AnalysisPlayback({ token, job }: { token: string | null; job: AnalysisJob }) {
  const [evidence, setEvidence] = useState<AnalysisEvidence | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [positionMs, setPositionMs] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [videoFailed, setVideoFailed] = useState(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const timerRef = useRef<number | null>(null)
  const step = job.step_ms || evidence?.step_ms || 40
  const frames = evidence?.frames_total ?? job.frames_total

  useEffect(() => {
    let cancelled = false
    setEvidence(null)
    setError(null)
    setPositionMs(0)
    setVideoFailed(false)
    void api
      .analysisEvidence(token, job.job_id)
      .then((result) => {
        if (!cancelled) setEvidence(result)
      })
      .catch((caught: Error) => {
        if (!cancelled) setError(caught)
      })
    return () => {
      cancelled = true
    }
  }, [job.job_id, token])

  const seek = useCallback(
    (ms: number) => {
      const total = Math.max(0, evidence?.duration_ms ?? job.timeline_ms)
      const clamped = Math.max(0, Math.min(total, ms))
      setPositionMs(clamped)
      const video = videoRef.current
      if (video) {
        video.currentTime = clamped / 1000
      }
    },
    [evidence?.duration_ms, job.timeline_ms],
  )

  // 帧序列播放：按 step_ms 前进。原生 video 自己走，不需要这个定时器。
  const useVideo = Boolean(evidence?.source_available) && !videoFailed

  useEffect(() => {
    if (!playing || !evidence || useVideo) return
    timerRef.current = window.setInterval(() => {
      setPositionMs((current) => {
        const next = current + step
        if (next >= evidence.duration_ms) {
          setPlaying(false)
          return evidence.duration_ms
        }
        return next
      })
    }, step)
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [playing, evidence, step, useVideo])

  // 预取后续几帧，避免逐帧回放时一路白屏。
  useEffect(() => {
    if (!evidence || useVideo) return
    const current = Math.floor(positionMs / step)
    for (let offset = 1; offset <= 6; offset += 1) {
      const index = current + offset
      if (index >= frames) break
      const image = new Image()
      image.src = analysisFrameUrl(token, job.job_id, index)
    }
  }, [evidence, positionMs, step, token, job.job_id, useVideo, frames])

  const currentFrame = Math.min(Math.max(0, Math.floor(positionMs / step)), Math.max(0, frames - 1))
  const point = evidence ? pointAt(evidence.timeline, positionMs) : null
  const inEvent = evidence?.events.find(
    (item) => positionMs >= item.started_ms && positionMs <= item.ended_ms,
  )

  if (error) {
    return (
      <Notice kind="warn" title="回放数据不可用">
        {(error as Error).message}
      </Notice>
    )
  }

  if (!evidence) {
    return <div className="muted small">正在加载回放数据…</div>
  }

  return (
    <div className="playback">
      <div className="playback-head">
        <strong>回放</strong>
        <span className="muted small">
          第 {currentFrame} 帧 / 共 {frames} 帧 · {formatClock(positionMs)} / {formatClock(evidence.duration_ms)}
          {' · '}步长 {step} ms
        </span>
        <span className="muted small">
          {videoFailed
            ? '浏览器无法解码原视频（编码不支持），已切换为抽帧回放'
            : useVideo
              ? '播放源：原视频（保留中）'
              : '播放源：抽帧序列（原视频未保留）'}
        </span>
      </div>

      <div className="playback-stage">
        {useVideo ? (
          <video
            ref={videoRef}
            src={analysisSourceUrl(token, job.job_id)}
            controls
            preload="metadata"
            onTimeUpdate={(event) => setPositionMs(event.currentTarget.currentTime * 1000)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            // 编码不支持时别给用户一个黑框：直接退回抽帧回放。
            onError={() => setVideoFailed(true)}
          />
        ) : (
          <img
            src={analysisFrameUrl(token, job.job_id, currentFrame)}
            alt={`第 ${currentFrame} 帧`}
            draggable={false}
          />
        )}
        <div className="playback-badge">
          {point ? (
            <>
              <span className={`badge risk-${point.risk_level.toLowerCase()}`}>{point.risk_level}</span>
              <span className="muted small">
                {point.primary_risk ?? '无行为结论'} · 置信度 {formatNumber(point.confidence, 2)}
              </span>
            </>
          ) : (
            <span className="muted small">该位置没有风险采样</span>
          )}
          {inEvent ? <span className="chip">{inEvent.event_type}</span> : null}
        </div>
      </div>

      {!useVideo ? (
        <div className="playback-controls">
          <button type="button" className="ghost" onClick={() => seek(positionMs - step * 10)}>
            −10 帧
          </button>
          <button type="button" className="ghost" onClick={() => seek(positionMs - step)}>
            −1 帧
          </button>
          <button type="button" onClick={() => setPlaying((value) => !value)}>
            {playing ? '暂停' : '播放'}
          </button>
          <button type="button" className="ghost" onClick={() => seek(positionMs + step)}>
            +1 帧
          </button>
          <button type="button" className="ghost" onClick={() => seek(positionMs + step * 10)}>
            +10 帧
          </button>
        </div>
      ) : null}

      <RiskCurve
        points={evidence.timeline}
        durationMs={evidence.duration_ms}
        positionMs={positionMs}
        events={evidence.events}
        onSeek={seek}
      />

      {evidence.events.length === 0 ? (
        <div className="muted small">这一轮没有形成事件（可能是全 SAFE，或不足最短持续时间）。</div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>事件</th>
              <th>最高风险</th>
              <th>帧区间</th>
              <th>时间区间</th>
              <th>时长</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {evidence.events.map((item) => (
              <tr key={item.event_id} onClick={() => seek(item.started_ms)}>
                <td>{item.event_type}</td>
                <td>
                  <span className={`badge risk-${item.risk_level.toLowerCase()}`}>{item.risk_level}</span>
                </td>
                <td className="mono">
                  {item.start_frame} – {item.end_frame}
                </td>
                <td className="mono">
                  {formatClock(item.started_ms)} – {formatClock(item.ended_ms)}
                </td>
                <td>{(item.duration_ms / 1000).toFixed(1)} s</td>
                <td>
                  <button
                    type="button"
                    className="ghost"
                    onClick={(event) => {
                      event.stopPropagation()
                      seek(item.started_ms)
                    }}
                  >
                    定位
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
