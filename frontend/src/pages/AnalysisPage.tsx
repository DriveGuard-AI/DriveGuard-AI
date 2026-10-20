/**
 * 离线分析（API-010 ~ API-014）。
 *
 * 上传一段视频 → 服务端抽帧 → 用这份素材重跑整条感知链路。
 * 进度由服务端跟踪写库，这里只负责轮询，所以刷新页面后状态不会丢。
 */

import { useCallback, useEffect, useState } from 'react'
import { ApiError, api, isJobActive, uploadAnalysisVideo } from '../api/client'
import type { AnalysisJob, AnalysisJobStatus } from '../api/types'
import { EmptyState, MetricCard, Notice, formatLocalTime } from '../components/common'
import { AnalysisPlayback } from './AnalysisPlayback'

const STEP_CHOICES = [40, 80, 120, 160, 200]
const JOB_STORAGE_KEY = 'dgb.analysis.job'

const STATUS_LABEL: Record<AnalysisJobStatus, string> = {
  queued: '排队中',
  running: '分析中',
  completed: '已完成',
  failed: '失败',
  cancelled: '已取消',
}

function formatTimeline(ms: number): string {
  const tenthsTotal = Math.max(0, Math.floor(ms / 100))
  const minutes = Math.floor(tenthsTotal / 600)
  const seconds = Math.floor((tenthsTotal % 600) / 10)
  const tenths = tenthsTotal % 10
  return `${minutes}:${String(seconds).padStart(2, '0')}.${tenths}`
}

function ProgressBar({ percent, label }: { percent: number; label: string }) {
  const width = Math.max(0, Math.min(100, percent))
  return (
    <div className="progress">
      <div className="progress-track">
        <i className="progress-fill" style={{ width: `${width}%` }} />
      </div>
      <div className="progress-meta">
        <span>{label}</span>
        <span className="mono">{width}%</span>
      </div>
    </div>
  )
}

export function AnalysisPage({
  token,
  onOpenSession,
}: {
  token: string | null
  onOpenSession: (sessionId: string, target: 'events', origin: string) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [stepMs, setStepMs] = useState(40)
  const [retainSource, setRetainSource] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadPercent, setUploadPercent] = useState(0)
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const [jobs, setJobs] = useState<AnalysisJob[]>([])
  const [error, setError] = useState<Error | null>(null)
  const [conflict, setConflict] = useState(false)
  const [listing, setListing] = useState(false)

  const loadJobs = useCallback(async () => {
    setListing(true)
    try {
      const result = await api.listAnalysisJobs(token, { page: 1, page_size: 20 })
      setJobs(result.items)
    } catch (caught) {
      setError(caught as Error)
    } finally {
      setListing(false)
    }
  }, [token])

  useEffect(() => {
    void loadJobs()
  }, [loadJobs])

  // 切标签会卸载本页；记住上一次看的任务，回来时把卡片还原。
  useEffect(() => {
    const stored = window.localStorage.getItem(JOB_STORAGE_KEY)
    if (!stored) return
    let cancelled = false
    void api
      .getAnalysisJob(token, stored)
      .then((restored) => {
        if (!cancelled) setJob(restored)
      })
      .catch(() => window.localStorage.removeItem(JOB_STORAGE_KEY))
    return () => {
      cancelled = true
    }
  }, [token])

  useEffect(() => {
    if (job) window.localStorage.setItem(JOB_STORAGE_KEY, job.job_id)
  }, [job])

  const active = job ? isJobActive(job.status) : false
  const activeJobId = job?.job_id

  // 只盯当前任务；终态由服务端写入，客户端不做本地推断。
  useEffect(() => {
    if (!activeJobId || !active) return
    let cancelled = false
    const timer = window.setInterval(() => {
      void api
        .getAnalysisJob(token, activeJobId)
        .then((next) => {
          if (cancelled) return
          setJob(next)
          // 同步列表里同一行，否则"最近任务"的进度会停在 0 直到任务结束。
          setJobs((previous) => previous.map((item) => (item.job_id === next.job_id ? next : item)))
          if (!isJobActive(next.status)) void loadJobs()
        })
        .catch((caught: Error) => {
          if (!cancelled) setError(caught)
        })
    }, 1000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [active, activeJobId, token, loadJobs])

  const submit = async (force = false) => {
    if (!file) return
    setUploading(true)
    setUploadPercent(0)
    setError(null)
    setConflict(false)
    try {
      const created = await uploadAnalysisVideo(
        token,
        file,
        { step_ms: stepMs, retain_source: retainSource, force },
        setUploadPercent,
      )
      setUploadPercent(100)
      setJob(created)
      await loadJobs()
    } catch (caught) {
      // 实时在跑：不静默顶掉，先问一句再带 force 重试。
      if (caught instanceof ApiError && caught.code === 'MONITORING_ACTIVE') {
        setConflict(true)
      } else {
        setError(caught as Error)
      }
    } finally {
      setUploading(false)
    }
  }

  const cancel = async () => {
    if (!job) return
    try {
      setJob(await api.cancelAnalysisJob(token, job.job_id))
      await loadJobs()
    } catch (caught) {
      setError(caught as Error)
    }
  }

  const framePercent =
    job && job.frames_total > 0
      ? Math.round((Math.min(job.frames_done, job.frames_total) / job.frames_total) * 100)
      : 0

  return (
    <div className="page">
      <div className="page-head">
        <h2>离线分析</h2>
        <div className="page-head-meta muted small">
          上传一段视频，服务端按 25 Hz 栅格抽帧后重新分析；每次分析会开启一个新会话
        </div>
      </div>

      <section className="card">
        <h3>1. 选择素材</h3>
        <div className="upload-row">
          <label className="upload-picker">
            <input
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mkv,.m4v"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              disabled={uploading}
            />
            <span>{file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB` : '选择视频文件（mp4 / mov / mkv / webm）'}</span>
          </label>
          <label className="inline-field">
            抽帧步长
            <select value={stepMs} onChange={(event) => setStepMs(Number(event.target.value))} disabled={uploading}>
              {STEP_CHOICES.map((value) => (
                <option key={value} value={value}>
                  {value} ms（{(1000 / value).toFixed(1)} Hz）
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => void submit()} disabled={!file || uploading}>
            {uploading ? '上传并抽帧…' : '开始分析'}
          </button>
        </div>

        <label className="inline-check">
          <input
            type="checkbox"
            checked={retainSource}
            onChange={(event) => setRetainSource(event.target.checked)}
            disabled={uploading}
          />
          保留原视频（回放更流畅；默认不保留，遵守 NFR-003）
        </label>

        {uploading ? (
          <ProgressBar percent={uploadPercent} label="上传进度（抽帧在服务端完成，期间请勿关闭页面）" />
        ) : null}

        <div className="muted small" style={{ marginTop: 10 }}>
          抽帧失败（非视频、编码不支持、超过帧数/磁盘上限）会直接拒绝，不会留下半成品任务；
          原始视频默认在抽帧后删除。
        </div>
      </section>

      {conflict ? (
        <Notice kind="warn" title="实时监控正在运行">
          切换素材会停止当前实时采集并结束该会话（事件按会话隔离，旧数据不会丢）。
          确认要切换到离线分析吗？
          <div style={{ marginTop: 8 }}>
            <button type="button" onClick={() => void submit(true)} disabled={uploading}>
              {uploading ? '切换中…' : '停止实时并开始分析'}
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

      {job ? (
        <section className="card">
          <div className="page-head">
            <h3>2. 当前任务</h3>
            <span className={`badge job-${job.status}`}>{STATUS_LABEL[job.status]}</span>
          </div>

          <ProgressBar
            percent={framePercent}
            label={`已处理 ${job.frames_done} / ${job.frames_total} 帧 · 素材位置 ${formatTimeline(job.timeline_ms)}`}
          />

          <div className="metric-grid">
            <MetricCard label="帧数" value={`${job.frames_done} / ${job.frames_total}`} />
            <MetricCard label="素材时间轴" value={formatTimeline(job.timeline_ms)} />
            <MetricCard label="创建时间" value={formatLocalTime(job.created_at)} />
            <MetricCard
              label="结束时间"
              value={job.ended_at ? formatLocalTime(job.ended_at) : '—'}
              hint={job.ended_at ? '终态时间' : '仍在进行'}
            />
          </div>

          <div className="job-actions">
            <span className="muted small mono">会话 {job.session_id}</span>
            {isJobActive(job.status) ? (
              <button type="button" className="ghost" onClick={() => void cancel()}>
                取消任务
              </button>
            ) : null}
            <button
              type="button"
              className="ghost"
              onClick={() => onOpenSession(job.session_id, 'events', '离线分析')}
            >
              查看该会话事件
            </button>
          </div>

          {job.error ? (
            <Notice kind="error" title="任务失败">
              <span className="mono">{job.error}</span>
            </Notice>
          ) : null}

          <div className="muted small" style={{ marginTop: 10 }}>
            素材文件：<span className="mono">{job.source_path.split(/[\\/]/).filter(Boolean).pop() ?? job.source_path}</span>
            {job.source_sha256 ? (
              <>
                {' '}
                · sha256 <span className="mono">{job.source_sha256.slice(0, 12)}…</span>
              </>
            ) : null}
          </div>
        </section>
      ) : null}

      {job && !isJobActive(job.status) ? (
        <section className="card">
          <AnalysisPlayback token={token} job={job} />
        </section>
      ) : null}

      <section className="card">
        <div className="page-head">
          <h3>最近任务</h3>
          <button type="button" className="ghost" onClick={() => void loadJobs()} disabled={listing}>
            {listing ? '刷新中…' : '刷新'}
          </button>
        </div>
        {(jobs.length ?? 0) === 0 ? (
          <EmptyState text="还没有分析任务。选择上方视频即可开始。" />
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>任务</th>
                <th>状态</th>
                <th>进度</th>
                <th>创建时间</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {jobs.map((item) => (
                <tr
                  key={item.job_id}
                  className={job?.job_id === item.job_id ? 'selected' : undefined}
                  onClick={() => setJob(item)}
                >
                  <td className="mono">{item.job_id.slice(0, 8)}…</td>
                  <td>
                    <span className={`badge job-${item.status}`}>{STATUS_LABEL[item.status]}</span>
                  </td>
                  <td>
                    {item.frames_done} / {item.frames_total}
                  </td>
                  <td>{formatLocalTime(item.created_at)}</td>
                  <td>
                    <button
                      type="button"
                      className="ghost"
                      onClick={(event) => {
                        event.stopPropagation()
                        onOpenSession(item.session_id, 'events', '离线分析')
                      }}
                    >
                      事件
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
