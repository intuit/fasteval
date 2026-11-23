import { useEffect, useState } from 'react'
import { api, runPath } from '../api'
import { Modal, Spinner, useToast } from '../components/ui'
import { useProject } from '../project'
import type { PgState } from '../playgroundStore'
import { href, navigate } from '../router'
import type { RunSummary } from '../types'

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_-]{0,63}$/

/** The built-in metric a judge descends from (kept through re-tunes). */
function basedOn(pg: PgState): string | null {
  const m = pg.metric
  if (!m) return null
  if (m.type === 'prompt_judge') return m.based_on ?? m.name
  return m.type || m.name
}

/** Save the edited judge to judges/<name>.toml. Resolves with the saved name. */
export function SaveJudgeDialog({ pg, onSaved, onClose }: { pg: PgState; onSaved: (name: string) => void; onClose: () => void }) {
  const toast = useToast()
  const judgesDir = useProject().project?.judges_dir ?? 'judges'
  const [name, setName] = useState(pg.savedAs ?? pg.name ?? '')
  const [description, setDescription] = useState(pg.description)
  const [overwrite, setOverwrite] = useState(!!pg.savedAs)
  // Off by default: a swapped-in judge then keeps each eval's own threshold.
  const [pinThreshold, setPinThreshold] = useState(false)
  const [busy, setBusy] = useState(false)
  const valid = NAME_RE.test(name)

  async function save() {
    setBusy(true)
    try {
      const res = await api.post<{ name: string; path: string }>('/api/judges', {
        name, description, prompt: pg.template, based_on: basedOn(pg),
        model: pg.model || null, temperature: pg.model ? pg.temperature : null,
        threshold: pinThreshold ? pg.threshold : null, scoring: pg.scoring, overwrite,
      })
      toast(`Saved ${res.path}`)
      onSaved(res.name)
      onClose()
    } catch (e) {
      toast((e as Error).message, true)
      setBusy(false)
    }
  }

  return (
    <Modal title="Save judge" onClose={onClose}
      actions={<button className="btn primary" disabled={!valid || busy} onClick={save}>{busy ? <Spinner /> : null} Save</button>}>
      <label className="field">Name
        <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="refund_correctness_strict" />
      </label>
      {!valid && name ? <span style={{ color: 'var(--fail)', fontSize: 12.5 }}>Use letters, digits, “_” or “-”, starting with a letter.</span> : null}
      <label className="field">Description
        <input type="text" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What this judge checks and why it differs" />
      </label>
      <label className="check"><input type="checkbox" checked={pinThreshold} onChange={(e) => setPinThreshold(e.target.checked)} /> Save threshold {pg.threshold} in the judge</label>
      <div className="muted" style={{ fontSize: 12.5, marginTop: -8 }}>
        {pinThreshold ? `Every eval using this judge passes at ≥ ${pg.threshold}.` : "Each eval keeps its own threshold (0.5 when used via @fe.prompt_judge without one)."}
      </div>
      <label className="check"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} /> Overwrite if it exists</label>
      <div className="muted" style={{ fontSize: 12.5 }}>
        Written to <span className="mono">{judgesDir}/{name || '<name>'}.toml</span> in your project. Commit it, then use it with
        {' '}<span className="kbd">@fe.prompt_judge("{name || 'name'}")</span> or <span className="kbd">fasteval run --judge {pg.metric?.name ?? 'metric'}={name || 'name'}</span>.
      </div>
    </Modal>
  )
}

const NO_BASE = '__new__'

/**
 * Run evals with a metric swapped for a saved judge: either re-run an
 * earlier run's exact configuration (then compared case by case) or start
 * a fresh run with the default profile.
 */
export function RunWithJudgeDialog({ judgeName, defaultMetric, defaultRunId, onClose }: {
  judgeName: string; defaultMetric?: string | null; defaultRunId?: string | null; onClose: () => void
}) {
  const toast = useToast()
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [baseRun, setBaseRun] = useState(defaultRunId ?? '')
  const [metric, setMetric] = useState(defaultMetric ?? '')
  const [failedOnly, setFailedOnly] = useState(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.get<{ runs: RunSummary[] }>('/api/runs').then((d) => {
      setRuns(d.runs)
      setBaseRun((b) => b || d.runs[0]?.id || NO_BASE)
    }, () => { setRuns([]); setBaseRun(NO_BASE) })
  }, [])

  async function go() {
    setBusy(true)
    const judges = [`${metric.trim()}=${judgeName}`]
    try {
      const r = baseRun === NO_BASE
        ? await api.post<RunSummary>('/api/runs', { judges })
        : await api.post<RunSummary>(runPath(baseRun, '/rerun'), { judges, failed_only: failedOnly })
      onClose()
      navigate(href.run(r.id))
    } catch (e) {
      toast((e as Error).message, true)
      setBusy(false)
    }
  }

  const metricOk = /^[A-Za-z_][\w.-]*$/.test(metric.trim())
  return (
    <Modal title={`Run evals with judge “${judgeName}”`} onClose={onClose}
      actions={<button className="btn primary" disabled={!baseRun || !metricOk || busy} onClick={go}>{busy ? <Spinner /> : '▶'} Run</button>}>
      <label className="field">Configuration
        <select value={baseRun} onChange={(e) => setBaseRun(e.target.value)} disabled={runs === null}>
          {(runs ?? []).map((r) => <option key={r.id} value={r.id}>Same as run {r.id} · {r.profile} · {r.status}{r.total != null ? ` · ${r.passed}/${r.total}` : ''}</option>)}
          <option value={NO_BASE}>New run with the default profile (no comparison)</option>
        </select>
      </label>
      <label className="field">Metric to replace
        <input type="text" value={metric} onChange={(e) => setMetric(e.target.value)} placeholder="e.g. correctness" />
      </label>
      {!metricOk && metric ? <span style={{ color: 'var(--fail)', fontSize: 12.5 }}>Use the metric's name as shown in a run (e.g. correctness).</span> : null}
      {baseRun !== NO_BASE ? (
        <label className="check"><input type="checkbox" checked={failedOnly} onChange={(e) => setFailedOnly(e.target.checked)} /> Only cases that failed in that run</label>
      ) : null}
      <div className="muted" style={{ fontSize: 12.5 }}>
        Every eval scored by <b>{metric || 'that metric'}</b> is judged by <b>{judgeName}</b> instead.
        {baseRun !== NO_BASE ? ' The new run is compared case by case with the one you pick, so you can see exactly which verdicts changed.' : ''}
      </div>
    </Modal>
  )
}
