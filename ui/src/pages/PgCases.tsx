import { useEffect, useState } from 'react'
import { api } from '../api'
import { Delta, Dot, Modal, Pill, Score, Spinner } from '../components/ui'
import { fmtMs } from '../format'
import type { PgCase } from '../playgroundStore'
import { href } from '../router'
import { loadRun, type RunState } from '../runState'
import type { RunSummary } from '../types'

function Verdict({ passed }: { passed?: boolean }) {
  if (passed === undefined) return null
  return <Pill status={passed ? 'passed' : 'failed'} />
}

export function PgCaseCard({ c, onChange, onRemove }: { c: PgCase; onChange: (c: PgCase) => void; onRemove: () => void }) {
  const r = c.result
  const ei = c.eval_input
  const agrees = (passed?: boolean) =>
    c.expected && passed !== undefined ? (passed === (c.expected === 'pass') ? '✓ agrees' : '✗ disagrees') : null
  return (
    <div className="panel pcase">
      <div className="row">
        <b className="grow" title={c.nodeid}>{c.label}</b>
        {c.runId && c.nodeid ? <a className="btn small ghost" href={href.run(c.runId, c.nodeid)}>open in run</a> : null}
        <button className="btn small ghost" onClick={onRemove} title="Remove from playground">✕</button>
      </div>
      <details>
        <summary>Input · expected · actual</summary>
        <div className="io">
          <div className="box"><div className="bh">Input</div><div className="bb"><pre>{ei.input ?? '—'}</pre></div></div>
          <div className="box expected"><div className="bh">Expected</div><div className="bb"><pre>{ei.expected_output ?? '—'}</pre></div></div>
          <div className="box actual"><div className="bh">Actual</div><div className="bb"><pre>{ei.actual_output ?? '—'}</pre></div></div>
        </div>
      </details>
      <div className="row" style={{ fontSize: 12.5 }}>
        <span className="muted">Should the judge pass this?</span>
        <div className="seg">
          {([null, 'pass', 'fail'] as const).map((v) => (
            <button key={String(v)} className={c.expected === v ? 'on' : ''} onClick={() => onChange({ ...c, expected: v })}>
              {v === null ? 'unlabeled' : v === 'pass' ? 'should pass' : 'should fail'}
            </button>
          ))}
        </div>
      </div>
      <div className="cmp">
        <div className="side">
          <span className="lbl">Original judge{c.original?.judge?.model ? ` · ${c.original.judge.model}` : ''}</span>
          {c.original ? (
            <>
              <div className="row"><Score value={c.original.score} passed={c.original.passed} /><Verdict passed={c.original.passed} /><span className="muted">{agrees(c.original.passed)}</span></div>
              {c.original.reasoning ? <div className="reasoning">{c.original.reasoning}</div> : null}
            </>
          ) : <span className="faint">No original score (added without one)</span>}
        </div>
        <div className="side new">
          <span className="lbl">Edited judge{r?.latency_ms != null ? ` · ${fmtMs(r.latency_ms)}` : ''}</span>
          {!r ? <span className="faint">Not tested yet. Press <b>Test</b>.</span> : r.error && r.score === undefined ? (
            <span style={{ color: 'var(--error)' }}>{r.error}</span>
          ) : (
            <>
              <div className="row">
                <Score value={r.score} passed={r.passed} /><Verdict passed={r.passed} />
                <Delta from={c.original?.score} to={r.score} /><span className="muted">{agrees(r.passed)}</span>
              </div>
              {r.reasoning ? <div className="reasoning">{r.reasoning}</div> : null}
              {r.error ? <div style={{ color: 'var(--error)', fontSize: 12.5 }}>{r.error}</div> : null}
              {r.prompt ? <details><summary>Rendered prompt</summary><pre className="trace">{r.prompt}</pre></details> : null}
              {r.response ? <details><summary>Raw response</summary><pre className="trace">{r.response}</pre></details> : null}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/** Pick cases from any run to test the judge against. */
export function AddCasesDialog({ metricName, defaultRunId, existing, onAdd, onClose }: {
  metricName: string | null; defaultRunId?: string; existing: Set<string>; onAdd: (cases: PgCase[]) => void; onClose: () => void
}) {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [runId, setRunId] = useState(defaultRunId ?? '')
  const [run, setRun] = useState<RunState | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  // Compare against one metric's verdicts. Fixed when opened from a metric;
  // otherwise the user picks which metric's original scores to compare with.
  const [chosenMetric, setChosenMetric] = useState<string | null>(metricName)

  useEffect(() => {
    api.get<{ runs: RunSummary[] }>('/api/runs').then((d) => {
      setRuns(d.runs)
      if (!runId && d.runs[0]) setRunId(d.runs[0].id)
    })
  }, []) // load the run list once; runId default is only applied the first time
  useEffect(() => {
    if (!runId) return
    setRun(null); setPicked(new Set())
    if (!metricName) setChosenMetric(null) // re-pick for the new run's metrics
    let alive = true
    loadRun(runId).then((r) => { if (alive) setRun(r) })
    return () => { alive = false }
  }, [runId, metricName])

  const metricNames = [...new Set([...(run?.tests.values() ?? [])].flatMap((t) => t.evals.flatMap((e) => e?.metric_results.map((m) => m.metric_name) ?? [])))].sort()
  const llmMetrics = new Set([...(run?.tests.values() ?? [])].flatMap((t) => t.evals.flatMap((e) => e?.metric_results.filter((m) => m.details?.judge_calls?.length).map((m) => m.metric_name) ?? [])))
  // Default to an LLM-judged metric (what a judge playground is for).
  useEffect(() => {
    if (!run || metricName) return
    if (!chosenMetric || !metricNames.includes(chosenMetric)) {
      setChosenMetric(metricNames.find((m) => llmMetrics.has(m)) ?? metricNames[0] ?? null)
    }
  }, [run, chosenMetric, metricName, metricNames.join('|')])

  // One candidate per (case, score) scored by the chosen metric.
  const candidates: PgCase[] = []
  for (const t of run?.tests.values() ?? []) {
    t.evals.forEach((e, i) => {
      if (!e) return
      const m = chosenMetric ? e.metric_results.find((x) => x.metric_name === chosenMetric) : undefined
      if (!m) return
      const key = `${t.nodeid}#${i}`
      if (existing.has(key)) return
      const judge = m?.details?.judge_calls?.slice(-1)[0]?.judge
      candidates.push({
        key, label: t.evals.length > 1 ? `${t.name} #${i + 1}` : t.name, runId, nodeid: t.nodeid, eval_input: e.eval_input,
        original: m ? { score: m.score, passed: m.passed, reasoning: m.reasoning, judge } : undefined,
      })
    })
  }
  const toggle = (key: string) => setPicked((s) => { const n = new Set(s); n.has(key) ? n.delete(key) : n.add(key); return n })
  const pick = (pred: (c: PgCase) => boolean) => setPicked(new Set(candidates.filter(pred).map((c) => c.key)))

  return (
    <Modal title="Add cases from a run" onClose={onClose}
      actions={<button className="btn primary" disabled={!picked.size} onClick={() => { onAdd(candidates.filter((c) => picked.has(c.key))); onClose() }}>Add {picked.size || ''} cases</button>}>
      <label className="field">Run
        <select value={runId} onChange={(e) => setRunId(e.target.value)}>
          {runs.map((r) => <option key={r.id} value={r.id}>{r.id} · {r.profile} · {r.status}</option>)}
        </select>
      </label>
      {!metricName && metricNames.length ? (
        <label className="field">Compare with the original verdicts of
          <select value={chosenMetric ?? ''} onChange={(e) => { setChosenMetric(e.target.value); setPicked(new Set()) }}>
            {metricNames.map((m) => <option key={m} value={m}>{m}{llmMetrics.has(m) ? ' (LLM judge)' : ''}</option>)}
          </select>
        </label>
      ) : null}
      <div className="row">
        <span className="muted">Cases scored by <b>{chosenMetric ?? '…'}</b></span>
        <span className="spacer" />
        <button className="btn small" onClick={() => pick((c) => c.original?.passed === false)}>Select failed</button>
        <button className="btn small" onClick={() => pick(() => true)}>Select all</button>
      </div>
      {!run ? <Spinner /> : candidates.length === 0 ? <div className="muted">No matching cases in this run.</div> : (
        <div style={{ maxHeight: '45vh', overflow: 'auto', border: '1px solid var(--border)', borderRadius: 8 }}>
          {candidates.map((c) => (
            <label key={c.key} className="case" style={{ cursor: 'pointer' }}>
              <input type="checkbox" checked={picked.has(c.key)} onChange={() => toggle(c.key)} />
              <Dot status={c.original ? (c.original.passed ? 'passed' : 'failed') : 'queued'} />
              <span className="nm">{c.label}</span>
              {c.original ? <Score value={c.original.score} passed={c.original.passed} /> : null}
            </label>
          ))}
        </div>
      )}
    </Modal>
  )
}
