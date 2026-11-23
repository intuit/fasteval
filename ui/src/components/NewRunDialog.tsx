import { useState } from 'react'
import { api } from '../api'
import { useProject } from '../project'
import { href, navigate } from '../router'
import type { RunRequest, RunSummary } from '../types'
import { Modal, Spinner, useToast } from './ui'

const splitList = (s: string) => s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean)

/** Start a run from the UI. Fields mirror `fasteval run` flags. */
export function NewRunDialog({ prefill, onClose }: { prefill: RunRequest; onClose: () => void }) {
  const { project } = useProject()
  const toast = useToast()
  const [profile, setProfile] = useState(prefill.profile ?? project?.default_profile ?? '')
  const [tags, setTags] = useState((prefill.tags ?? []).join(', '))
  const [keyword, setKeyword] = useState(prefill.keyword ?? '')
  const [paths, setPaths] = useState((prefill.paths ?? []).join('\n'))
  const [judges, setJudges] = useState((prefill.judges ?? []).join(', '))
  const [workers, setWorkers] = useState(prefill.workers ?? '')
  const [exitfirst, setExitfirst] = useState(!!prefill.exitfirst)
  const [lastFailed, setLastFailed] = useState(!!prefill.last_failed)
  const [includeAll, setIncludeAll] = useState(!!prefill.include_all)
  const [busy, setBusy] = useState(false)

  const selected = project?.profiles.find((p) => p.name === profile)
  const tagNames = Object.keys(project?.tags ?? {})

  async function start() {
    setBusy(true)
    const body: RunRequest = {
      profile: profile || null,
      tags: tags.trim() ? [tags.trim()] : [],
      keyword: keyword.trim() || null,
      paths: splitList(paths),
      judges: splitList(judges),
      workers: workers.trim() || null,
      exitfirst,
      last_failed: lastFailed,
      include_all: includeAll,
    }
    try {
      const run = await api.post<RunSummary>('/api/runs', body)
      onClose()
      navigate(href.run(run.id))
    } catch (e) {
      toast((e as Error).message, true)
      setBusy(false)
    }
  }

  return (
    <Modal
      title="New run"
      onClose={onClose}
      actions={<button className="btn primary" disabled={busy} onClick={start}>{busy ? <Spinner /> : '▶'} Run</button>}
    >
      <div className="form-grid">
        <label className="field full">
          Profile
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {project?.profiles.map((p) => (
              <option key={p.name} value={p.name}>
                {p.name}{p.name === project.default_profile ? ' (default)' : ''}{p.description ? ` — ${p.description}` : ''}
              </option>
            ))}
          </select>
        </label>
        {selected ? (
          <div className="full muted" style={{ fontSize: 12.5 }}>
            paths: {selected.paths.join(' ') || 'pytest testpaths'} · tags: {selected.tags || 'all'}
            {selected.min_pass_rate != null ? ` · pass rate ≥ ${Math.round(selected.min_pass_rate * 100)}%` : ''}
            {selected.min_score != null ? ` · score ≥ ${selected.min_score}` : ''}
          </div>
        ) : null}
        <label className="field">
          Tags (narrows the profile)
          <input type="text" value={tags} placeholder="e.g. smoke and not slow" onChange={(e) => setTags(e.target.value)} />
        </label>
        <label className="field">
          Name filter (-k)
          <input type="text" value={keyword} placeholder="e.g. refund" onChange={(e) => setKeyword(e.target.value)} />
        </label>
        {tagNames.length ? (
          <div className="full row">
            {tagNames.map((t) => (
              <span key={t} className="chip tag clickable" title={project?.tags[t]} onClick={() => setTags(tags ? `${tags} or ${t}` : t)}>{t}</span>
            ))}
          </div>
        ) : null}
        <label className="field full">
          Paths or node ids (one per line, optional)
          <textarea rows={2} value={paths} onChange={(e) => setPaths(e.target.value)} />
        </label>
        <label className="field">
          Judge swaps (metric=judge)
          <input type="text" value={judges} placeholder="correctness=refund_strict" onChange={(e) => setJudges(e.target.value)} />
        </label>
        <label className="field">
          Workers (-n)
          <input type="text" value={workers} placeholder="auto" onChange={(e) => setWorkers(e.target.value)} />
        </label>
        <div className="full row">
          <label className="check"><input type="checkbox" checked={exitfirst} onChange={(e) => setExitfirst(e.target.checked)} /> Stop on first failure</label>
          <label className="check"><input type="checkbox" checked={lastFailed} onChange={(e) => setLastFailed(e.target.checked)} /> Last failed only</label>
          <label className="check"><input type="checkbox" checked={includeAll} onChange={(e) => setIncludeAll(e.target.checked)} /> Include plain tests</label>
        </div>
      </div>
    </Modal>
  )
}
