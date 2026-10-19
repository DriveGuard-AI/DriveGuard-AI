/** 实时监控（FR-007）。 */

import {
  EmptyState,
  MetricCard,
  Notice,
  QualityBadge,
  RiskBadge,
  StatusDot,
  formatLocalTime,
  formatNumber,
} from '../components/common'
import type { MonitoringFeed } from '../hooks/useMonitoringSocket'
import type { RiskSnapshot } from '../api/types'

/** 毫秒时间轴 → m:ss.s。离线分析时表示素材播放位置，不是墙钟。 */
function formatTimeline(ms: number): string {
  const tenthsTotal = Math.max(0, Math.floor(ms / 100))
  const minutes = Math.floor(tenthsTotal / 600)
  const seconds = Math.floor((tenthsTotal % 600) / 10)
  const tenths = tenthsTotal % 10
  return `${minutes}:${String(seconds).padStart(2, '0')}.${tenths}`
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100)
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <span className="score-track">
        <i style={{ width: `${percent}%` }} />
      </span>
      <span className="score-value">{value.toFixed(2)}</span>
    </div>
  )
}

export function MonitoringPage({ feed }: { feed: MonitoringFeed }) {
  const snapshot: RiskSnapshot | null = feed.snapshot

  return (
    <div className="page">
      <div className="page-head">
        <h2>实时监控</h2>
        <div className="page-head-meta">
          <StatusDot state={feed.connection} />
          {snapshot?.source_timestamp_ms != null ? (
            <span className="muted">素材时间：{formatTimeline(snapshot.source_timestamp_ms)}</span>
          ) : null}
          <span className="muted">最近消息：{formatLocalTime(feed.lastMessageAt)}</span>
        </div>
      </div>

      {snapshot?.inference_mode === 'fake' ? (
        <Notice kind="warn" title="当前推理为脚本化占位（inference_mode = fake）">
          模型识别与判断尚未实现。以下风险等级、特征与检测结果由确定性脚本生成，仅用于前后端联调，
          <b>不构成真实行为结论</b>。接入真实模型后该字段将变为 <code>real</code>。
        </Notice>
      ) : null}

      {feed.stale ? (
        <Notice kind="warn" title="心跳超时">
          已超过 45 秒未收到服务端消息，连接可能已陈旧。
        </Notice>
      ) : null}

      {feed.schemaWarning ? <Notice kind="warn" title="协议版本提示">{feed.schemaWarning}</Notice> : null}

      {feed.sequenceGap ? (
        <Notice kind="info" title="检测到序列跳跃">
          已通过 REST 重新拉取完整快照（不要求服务端回放每一帧消息）。
        </Notice>
      ) : null}

      {!snapshot ? (
        <EmptyState text="等待风险快照…（若无活动会话将返回 409 MONITORING_NOT_ACTIVE）" />
      ) : (
        <>
          <section className="card">
            <div className="risk-head">
              <div>
                <div className="risk-level">
                  <RiskBadge level={snapshot.risk_level} />
                  <QualityBadge quality={snapshot.quality} />
                </div>
                <div className="risk-summary">
                  主要风险：<b>{snapshot.primary_risk ?? '无行为结论'}</b>
                  <span className="muted"> · 置信度 {formatNumber(snapshot.confidence, 2)}</span>
                </div>
                <div className="muted small">
                  会话 {snapshot.session_id.slice(0, 8)}… · 序列 {snapshot.sequence} · 采集{' '}
                  {formatLocalTime(snapshot.captured_at)} · schema {snapshot.schema_version}
                </div>
              </div>
              <div className="risk-actions">
                <button type="button" className="ghost" onClick={() => void feed.resync()}>
                  重新同步快照
                </button>
                {feed.reconnectAttempts > 0 ? (
                  <span className="muted small">重连尝试 {feed.reconnectAttempts} 次</span>
                ) : null}
              </div>
            </div>

            {snapshot.quality !== 'good' ? (
              <Notice kind="warn" title="输入质量不足（degraded / unknown）">
                系统不会把缺失或降级的输入解释为安全（业务规则 2）。此时不应据此判断驾驶员处于安全状态。
              </Notice>
            ) : null}

            <div className="metric-grid">
              <MetricCard label="EAR 眼睛纵横比" value={formatNumber(snapshot.features.ear)} />
              <MetricCard label="MAR 嘴部纵横比" value={formatNumber(snapshot.features.mar)} />
              <MetricCard label="PERCLOS" value={formatNumber(snapshot.features.perclos)} />
              <MetricCard
                label="头部姿态 yaw/pitch/roll"
                value={
                  snapshot.features.head_pose
                    ? `${formatNumber(snapshot.features.head_pose.yaw, 1)} / ${formatNumber(
                        snapshot.features.head_pose.pitch,
                        1,
                      )} / ${formatNumber(snapshot.features.head_pose.roll, 1)}`
                    : '—'
                }
                unit="°"
              />
            </div>
          </section>

          <div className="two-column">
            <section className="card">
              <h3>风险分数</h3>
              {Object.keys(snapshot.risk_scores).length === 0 ? (
                <EmptyState text="当前无风险分数" />
              ) : (
                Object.entries(snapshot.risk_scores).map(([key, value]) => (
                  <ScoreBar key={key} label={key} value={value} />
                ))
              )}
              <h4>原因码</h4>
              {snapshot.reason_codes.length === 0 ? (
                <span className="muted">无</span>
              ) : (
                <div className="chip-row">
                  {snapshot.reason_codes.map((code) => (
                    <span className="chip" key={code}>
                      {code}
                    </span>
                  ))}
                </div>
              )}
            </section>

            <section className="card">
              <h3>检测结果（帧级观测，非行为结论）</h3>
              {snapshot.detections.length === 0 ? (
                <EmptyState text="无检测目标（输入降级时不会伪造结果）" />
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>标签</th>
                      <th>置信度</th>
                      <th>边界框</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.detections.map((detection) => (
                      <tr key={detection.label}>
                        <td>{detection.label}</td>
                        <td>{formatNumber(detection.confidence, 2)}</td>
                        <td className="mono">{detection.bbox.map((v) => v.toFixed(2)).join(', ')}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
          </div>

          <section className="card">
            <h3>事件通知（event.changed）</h3>
            {feed.events.length === 0 ? (
              <EmptyState text="尚未收到事件变化通知" />
            ) : (
              <ul className="feed">
                {feed.events.map((change, index) => (
                  <li key={`${change.event.event_id}-${index}`}>
                    <span className={`badge action-${change.action}`}>{change.action}</span>
                    <b>{change.event.event_type}</b>
                    <span className="muted">
                      最高等级 {change.event.max_risk_level} · 时长 {change.event.duration_ms} ms ·{' '}
                      {formatLocalTime(change.event.started_at)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </div>
  )
}
