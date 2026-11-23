import { useEffect, useState } from 'react'
import { api } from '../api'
import { Bar, Dot, Empty, Score, Spinner, useToast } from '../components/ui'
import { fmtAgo, fmtDuration, fmtPct } from '../format'
import { useProject } from '../project'
import { href, navigate } from '../router'
import type { RunRequest, RunSummary } from '../types'

export function describeSelection(req?: RunRequest): string {
  if (!req) return ''
  const bits: string[] = []
  if (req.paths?.length) bits.push(req.paths.length === 1 ? req.paths[0] : `${req.paths.length} paths`)
  if (req.tags?.length) bits.push(`-t ${req.tags.join(' | ')}`)
  if (req.keyword) bits.push(`-k ${req.keyword}`)
  if (req.last_failed) bits.push('--lf')
  if (req.judges?.length) bits.push(`judge ${req.judges.join(', ')}`)
  return bits.join(' · ')
}

function runSeconds(r: RunSummary): number | null {
  if (!r.started_at) return null
  const end = r.finished_at ? new Date(r.finished_at).getTime() : r.status === 'running' ? Date.now() : null
  return end ? (end - new Date(r.started_at).getTime()) / 1000 : null
}

export function RunsPage() {
  const { openNewRun } = useProject()
  const toast = useToast()
  const [runs, setRuns] = useState<RunSummary[] | null>(null)

  useEffect(() => {
    let timer = 0
    let alive = true
    const load = () =>
      api.get<{ runs: RunSummary[] }>('/api/runs').then((d) => {
        if (!alive) return
        setRuns(d.runs)
        // Keep the list live while anything is running.
        if (d.runs.some((r) => r.status === 'running')) timer = window.setTimeout(load, 2000)
      }, (e: Error) => toast(e.message, true))
    load()
    return () => { alive = false; clearTimeout(timer) }
  }, [toast])

  async function rerun(r: RunSummary, failedOnly: boolean) {
    try {
      const run = await api.post<RunSummary>(`/api/runs/${encodeURIComponent(r.id)}/rerun`, { failed_only: failedOnly })
      navigate(href.run(run.id))
    } catch (e) {
      toast((e as Error).message, true)
    }
  }

  if (runs === null) return <Empty title="Loading runs…"><Spinner /></Empty>

  return (
    <>
      <div className="page-head">
        <h1>Runs</h1>
        <span className="muted">{runs.length} recorded · newest first</span>
        <span className="spacer" />
        <button className="btn primary" onClick={() => openNewRun()}>▶ New run</button>
      </div>
      {runs.length === 0 ? (
        <div className="panel">
          <Empty title="No runs yet">
            Start one with <span className="kbd">fasteval run</span> or the <b>New run</b> button.
          </Empty>
        </div>
      ) : (
        <div className="panel">
          <table className="list">
            <thead>
              <tr>
                <th>Status</th><th>Run</th><th>Profile</th><th>Selection</th><th>Evals</th>
                <th>Pass rate</th><th>Avg score</th><th>Duration</th><th />
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="clickable" onClick={() => navigate(href.run(r.id))}>
                  <td><span className="row"><Dot status={r.status} /> {r.status}</span></td>
                  <td>
                    <div className="mono">{r.id}</div>
                    <div className="faint" style={{ fontSize: 12 }}>
                      {fmtAgo(r.started_at)} · <span className="chip">{r.origin ?? 'cli'}</span>
                    </div>
                  </td>
                  <td><span className="chip">{r.profile ?? '—'}</span></td>
                  <td className="muted" style={{ maxWidth: 280 }}>{describeSelection(r.request) || '—'}</td>
                  <td>{r.total != null ? <><b>{r.passed}</b><span className="muted">/{r.total}</span></> : '—'}</td>
                  <td>
                    {r.pass_rate != null ? (
                      <div className="row" style={{ gap: 8 }}>
                        <Bar value={r.pass_rate} bad={r.pass_rate < 0.9} /> {fmtPct(r.pass_rate)}
                      </div>
                    ) : '—'}
                  </td>
                  <td>{r.avg_score != null ? <Score value={r.avg_score} passed={r.avg_score >= 0.7} /> : '—'}</td>
                  <td className="mono muted">{fmtDuration(runSeconds(r))}</td>
                  <td onClick={(e) => e.stopPropagation()}>
                    <div className="row" style={{ gap: 4 }}>
                      <button className="btn small" title="Re-run with the same configuration" onClick={() => rerun(r, false)}>↻ Re-run</button>
                      {r.failed ? <button className="btn small" title="Re-run only failed cases" onClick={() => rerun(r, true)}>↻ Failed</button> : null}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}
