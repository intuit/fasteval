import { Delta, Dot, Score } from '../components/ui'
import { caseScore, type RunState, type TestState } from '../runState'

export type StatusFilter = 'all' | 'failed' | 'passed' | 'active'

export interface CaseFilters {
  search: string
  status: StatusFilter
  tag: string | null
  metric: string | null
}

const matches = (t: TestState, f: CaseFilters): boolean => {
  if (f.search) {
    const q = f.search.toLowerCase()
    if (!t.nodeid.toLowerCase().includes(q) && !t.tags.some((x) => x.includes(q))) return false
  }
  if (f.status === 'failed' && !['failed', 'error'].includes(t.status)) return false
  if (f.status === 'passed' && t.status !== 'passed') return false
  if (f.status === 'active' && !['running', 'queued'].includes(t.status)) return false
  if (f.tag && !t.tags.includes(f.tag)) return false
  if (f.metric && !t.evals.some((e) => e?.metric_results.some((m) => m.metric_name === f.metric && !m.passed))) return false
  return true
}

/** Split "test_x[case id]" into name and case id for display. */
function splitName(name: string): [string, string] {
  const i = name.indexOf('[')
  return i < 0 ? [name, ''] : [name.slice(0, i), name.slice(i)]
}

interface Props {
  run: RunState
  filters: CaseFilters
  onFilters: (f: CaseFilters) => void
  selected: string | null
  onSelect: (nodeid: string) => void
  baseScores: Map<string, number> | null
}

export function CaseList({ run, filters, onFilters, selected, onSelect, baseScores }: Props) {
  const tests = [...run.tests.values()]
  const allTags = [...new Set(tests.flatMap((t) => t.tags))].sort()
  const visible = tests.filter((t) => matches(t, filters))
  const groups = new Map<string, TestState[]>()
  for (const t of visible) groups.set(t.file, [...(groups.get(t.file) ?? []), t])
  const set = (patch: Partial<CaseFilters>) => onFilters({ ...filters, ...patch })

  return (
    <div className="panel cases">
      <div className="toolbar">
        <input type="text" placeholder="Search cases, ids, tags…" value={filters.search} onChange={(e) => set({ search: e.target.value })} />
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <div className="seg">
            {(['all', 'failed', 'passed', 'active'] as StatusFilter[]).map((s) => (
              <button key={s} className={filters.status === s ? 'on' : ''} onClick={() => set({ status: s })}>
                {s === 'active' ? 'running' : s}
              </button>
            ))}
          </div>
          <span className="muted" style={{ fontSize: 12 }}>{visible.length}/{tests.length}</span>
        </div>
        {allTags.length ? (
          <div className="row" style={{ gap: 4 }}>
            {allTags.map((tag) => (
              <span key={tag} className={`chip tag clickable${filters.tag === tag ? ' on' : ''}`} onClick={() => set({ tag: filters.tag === tag ? null : tag })}>{tag}</span>
            ))}
          </div>
        ) : null}
        {filters.metric ? (
          <div className="row" style={{ fontSize: 12.5 }}>
            <span className="muted">Only cases failing</span><span className="chip">{filters.metric}</span>
            <button className="btn small ghost" onClick={() => set({ metric: null })}>clear</button>
          </div>
        ) : null}
      </div>
      <div className="case-list">
        {visible.length === 0 ? <div className="empty">{tests.length ? 'No cases match these filters.' : 'Waiting for collection…'}</div> : null}
        {[...groups.entries()].map(([file, items]) => (
          <div className="file-group" key={file}>
            <div className="file"><span className="grow">{file}</span><span>{items.length}</span></div>
            {items.map((t) => {
              const s = caseScore(t)
              const [fn, cid] = splitName(t.name)
              return (
                <div key={t.nodeid} className={`case${selected === t.nodeid ? ' sel' : ''}`} onClick={() => onSelect(t.nodeid)}>
                  <Dot status={t.status} />
                  <span className="nm" title={t.nodeid}>{fn}<span className="cid">{cid}</span></span>
                  {baseScores && s ? <Delta from={baseScores.get(t.nodeid)} to={s.score} /> : null}
                  {s ? <Score value={s.score} passed={s.passed} /> : null}
                  <span className="dur">{t.duration != null ? `${t.duration.toFixed(2)}s` : ''}</span>
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}
