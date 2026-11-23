import { useEffect, useState } from 'react'
import { api, runPath, STATIC } from '../api'
import { CopyButton, Delta, Pill, Stat, useToast } from '../components/ui'
import { fmtAgo, fmtDuration, fmtMs, fmtPct, fmtScore } from '../format'
import { href, navigate } from '../router'
import { failedCount, runStats, type RunState, type RunStats } from '../runState'
import type { RunSummary } from '../types'

interface Props {
  run: RunState
  info: RunSummary | null
  base: { id: string; stats: RunStats } | null
  metricFilter: string | null
  onMetricFilter: (name: string | null) => void
  onShowLog: () => void
}

function useNow(active: boolean): number {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [active])
  return now
}

export function RunHeader({ run, info, base, metricFilter, onMetricFilter, onShowLog }: Props) {
  const toast = useToast()
  const stats = runStats(run)
  const running = run.status === 'running'
  const now = useNow(running)
  const meta = info?.meta ?? {}
  const failed = failedCount(run)
  const started = run.firstTs ?? (meta.started_at ? new Date(meta.started_at).getTime() / 1000 : null)
  const elapsed = started ? (running ? now / 1000 : run.lastTs ?? now / 1000) - started : null
  const gates = (run.end?.gates ?? meta.gates ?? {}) as { min_score?: number | null; min_pass_rate?: number | null }
  const gateFailures: string[] = run.end?.gate_failures ?? []
  const judges: string[] = meta.request?.judges ?? []
  const status = info?.meta?.stopped && !running ? 'stopped' : run.status

  async function rerun(failedOnly: boolean) {
    try {
      const r = await api.post<RunSummary>(runPath(run.id, '/rerun'), { failed_only: failedOnly })
      navigate(href.run(r.id))
    } catch (e) { toast((e as Error).message, true) }
  }
  async function stop() {
    try { await api.post(runPath(run.id, '/stop')); toast('Stopping… pytest will finish the current test') }
    catch (e) { toast((e as Error).message, true) }
  }

  const pct = (n: number) => `${stats.total ? (n / stats.total) * 100 : 0}%`
  const c = stats.counts

  return (
    <div className="panel run-head">
      <div className="row">
        <Pill status={status} />
        <h1 className="mono" style={{ fontSize: 17 }}>{run.id}</h1>
        {info?.profile ? <span className="chip">profile {info.profile}</span> : null}
        {info?.origin ? <span className="chip">{info.origin}</span> : null}
        <span className="muted">{fmtAgo(meta.started_at)} · {fmtDuration(elapsed)}</span>
        <span className="spacer" />
        {!STATIC ? (
          <>
            {info?.has_log ? <button className="btn small" onClick={onShowLog}>Console</button> : null}
            <a className="btn small" href={runPath(run.id, '/report.html')} title="Self-contained HTML report (CI artifact)">Export HTML</a>
            <button className="btn small" onClick={() => rerun(false)} title="Same profile, tags, paths and flags">↻ Re-run</button>
            <button className="btn small" disabled={!failed || running} onClick={() => rerun(true)} title="Only the cases that failed in this run">↻ Re-run failed{failed ? ` (${failed})` : ''}</button>
            {running && info?.origin === 'ui' ? <button className="btn small danger" onClick={stop}>■ Stop</button> : null}
          </>
        ) : null}
      </div>

      {judges.length ? (
        <div className="banner">Judge swap: {judges.map((j) => j.replace('=', ' → ')).join(', ')}</div>
      ) : null}
      {base ? (
        <div className="banner">
          Compared with <a href={href.run(base.id)} className="mono"><u>{base.id}</u></a>:{' '}
          pass rate {fmtPct(base.stats.passRate)} → <b>{fmtPct(stats.passRate)}</b>{' '}
          · avg score {fmtScore(base.stats.avg)} → <b>{fmtScore(stats.avg)}</b> <Delta from={base.stats.avg} to={stats.avg} />
        </div>
      ) : null}
      {status === 'interrupted' ? <div className="banner warn">This run ended without finishing (process exited early or was killed).</div> : null}

      {meta.fasteval_command || meta.command ? (
        <div>
          <div className="cmd" title="Run this to reproduce the run from a terminal">
            <code>$ {meta.fasteval_command ?? meta.command}</code><span className="spacer" />
            <CopyButton text={meta.fasteval_command ?? meta.command} />
          </div>
          {meta.fasteval_command && meta.command ? (
            <details className="cmd-pytest">
              <summary>Underlying pytest command</summary>
              <div className="cmd"><code>$ {meta.command}</code><span className="spacer" /><CopyButton text={meta.command} /></div>
            </details>
          ) : null}
        </div>
      ) : null}

      <div>
        <div className="progress" title={`${stats.done}/${stats.total}`}>
          <span className="passed" style={{ width: pct(c.passed) }} />
          <span className="failed" style={{ width: pct(c.failed) }} />
          <span className="error" style={{ width: pct(c.error) }} />
          <span className="skipped" style={{ width: pct(c.skipped) }} />
          <span className="running" style={{ width: pct(c.running) }} />
        </div>
        <div className="row muted" style={{ fontSize: 12.5, marginTop: 6 }}>
          <span><b style={{ color: 'var(--text)' }}>{stats.done}</b> / {stats.total} cases</span>
          <span style={{ color: 'var(--pass)' }}>{c.passed} passed</span>
          {c.failed ? <span style={{ color: 'var(--fail)' }}>{c.failed} failed</span> : null}
          {c.error ? <span style={{ color: 'var(--error)' }}>{c.error} errors</span> : null}
          {c.skipped ? <span>{c.skipped} skipped</span> : null}
          {c.running ? <span style={{ color: 'var(--run)' }}>{c.running} running</span> : null}
          {c.queued && running ? <span>{c.queued} queued</span> : null}
        </div>
      </div>

      <div className="stats">
        <Stat label="Eval pass rate" value={fmtPct(stats.passRate)} sub={`${stats.results} scored results`} />
        <Stat label="Avg score" value={fmtScore(stats.avg)} sub={base ? <Delta from={base.stats.avg} to={stats.avg} /> : null} />
        <Stat label="Failed cases" value={<span style={{ color: failed ? 'var(--fail)' : undefined }}>{failed}</span>} />
        <Stat label="Judge calls" value={stats.judgeCalls} sub={stats.judgeCalls ? `avg ${fmtMs(stats.judgeMs / stats.judgeCalls)}` : 'no LLM judges'} />
        <Stat label="Duration" value={fmtDuration(elapsed)} sub={run.start?.workers ? `${run.start.workers} workers` : null} />
      </div>

      {gates.min_score != null || gates.min_pass_rate != null ? (
        <div className="row" style={{ fontSize: 13 }}>
          <span className="muted">Quality gates:</span>
          {gates.min_pass_rate != null ? <GateChip label={`pass rate ≥ ${fmtPct(gates.min_pass_rate)}`} failed={gateFailures.some((g) => g.startsWith('pass rate'))} pending={!run.end} /> : null}
          {gates.min_score != null ? <GateChip label={`avg score ≥ ${gates.min_score}`} failed={gateFailures.some((g) => g.startsWith('average score'))} pending={!run.end} /> : null}
        </div>
      ) : null}

      {stats.metrics.length ? (
        <div className="metric-chips">
          {stats.metrics.map((m) => {
            const avg = m.sum / m.n
            return (
              <div key={m.name} className={`metric-chip${metricFilter === m.name ? ' on' : ''}`} title="Click to show only cases where this metric failed"
                onClick={() => onMetricFilter(metricFilter === m.name ? null : m.name)}>
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <span className="name">{m.name}</span>
                  <span className={`score ${m.pass === m.n ? 'good' : 'bad'}`}>{avg.toFixed(2)}</span>
                </div>
                <div className={`bar${m.pass === m.n ? '' : ' bad'}`}><span style={{ width: `${avg * 100}%` }} /></div>
                <span className="muted" style={{ fontSize: 12 }}>{m.pass}/{m.n} passed · {fmtPct(m.pass / m.n)}</span>
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}

function GateChip({ label, failed, pending }: { label: string; failed: boolean; pending: boolean }) {
  const status = pending ? 'queued' : failed ? 'failed' : 'passed'
  return <Pill status={status}>{pending ? '…' : failed ? '✗' : '✓'} {label}</Pill>
}
