import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { Empty, Spinner, useToast } from '../components/ui'
import { useProject } from '../project'
import { href, navigate } from '../router'
import type { InventoryItem, RunRequest, RunSummary } from '../types'

interface EvalRow { file: string; func: string; nodeid: string; cases: InventoryItem[] }

function dataLabel(d: InventoryItem['data']): string | null {
  if (!d) return null
  if (d.type === 'csv' || d.type === 'jsonl') return `${d.type} ${d.path ?? ''}`
  if (d.type === 'dataset') return `dataset ${d.name ?? ''}`
  if (d.type === 'conversation') return `conversation ${String(d.turns ?? '')} turns`
  if (d.type.startsWith('langfuse')) return `${d.type.replace('_', ' ')} ${d.name ?? d.project ?? ''}`
  return d.type
}

export function EvalsPage() {
  const { project } = useProject()
  const toast = useToast()
  const [profile, setProfile] = useState('')
  const [tags, setTags] = useState('')
  const [keyword, setKeyword] = useState('')
  const [items, setItems] = useState<InventoryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  // The filters the list was collected with; Run uses these, never half-typed input.
  const [applied, setApplied] = useState<{ profile: string; tags: string; keyword: string } | null>(null)

  useEffect(() => { if (project && !profile) setProfile(project.default_profile) }, [project, profile])

  const load = useCallback((f: { profile: string; tags: string; keyword: string }) => {
    if (!f.profile) return
    setLoading(true)
    api.get<{ items: InventoryItem[] }>(`/api/evals?${new URLSearchParams(f)}`).then(
      (d) => { setItems(d.items); setError(null); setApplied(f) },
      (e: Error) => setError(e.message),
    ).finally(() => setLoading(false))
  }, [])
  // Profile changes and tag-chip clicks apply immediately; typed filters on Enter/Apply.
  useEffect(() => { if (profile) load({ profile, tags, keyword }) }, [profile])
  const applyTag = (t: string) => { setTags(t); load({ profile, tags: t, keyword }) }
  const pending = !!applied && (applied.tags !== tags || applied.keyword !== keyword)

  async function run(extra: Partial<RunRequest>) {
    if (!applied) return
    try {
      const r = await api.post<RunSummary>('/api/runs', {
        profile: applied.profile, tags: applied.tags ? [applied.tags] : [], keyword: applied.keyword || null, ...extra,
      })
      navigate(href.run(r.id))
    } catch (e) { toast((e as Error).message, true) }
  }

  // Group cases (parametrized / dataset rows) under their eval function.
  const files = new Map<string, EvalRow[]>()
  for (const it of items ?? []) {
    const rows = files.get(it.file) ?? []
    let row = rows.find((r) => r.func === it.function)
    if (!row) { row = { file: it.file, func: it.function, nodeid: `${it.nodeid.split('::').slice(0, -1).join('::')}::${it.function}`, cases: [] }; rows.push(row) }
    row.cases.push(it)
    files.set(it.file, rows)
  }
  const evalCount = [...files.values()].reduce((n, rows) => n + rows.length, 0)

  return (
    <>
      <div className="page-head">
        <h1>Evals</h1>
        <span className="muted">{items ? `${evalCount} evals · ${items.length} cases · ${files.size} files` : ''}</span>
        <span className="spacer" />
        <button className="btn primary" disabled={!items?.length || loading} onClick={() => run({})}
          title="Runs exactly the evals listed below">▶ Run {items ? `${items.length} cases` : 'all shown'}</button>
      </div>
      <form className="panel filters" onSubmit={(e) => { e.preventDefault(); load({ profile, tags, keyword }) }}>
        <label className="field">Profile
          <select value={profile} onChange={(e) => setProfile(e.target.value)}>
            {project?.profiles.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
          </select>
        </label>
        <label className="field">Tags<input type="text" value={tags} placeholder="rag and not slow" onChange={(e) => setTags(e.target.value)} /></label>
        <label className="field">Name (-k)<input type="text" value={keyword} onChange={(e) => setKeyword(e.target.value)} /></label>
        <button className={`btn${pending ? ' primary' : ''}`} type="submit">{loading ? <Spinner /> : null} Apply</button>
        {tags || keyword ? <button className="btn ghost" type="button" onClick={() => { setTags(''); setKeyword(''); load({ profile, tags: '', keyword: '' }) }}>Clear</button> : null}
        {Object.entries(project?.tags ?? {}).map(([t, d]) => (
          <span key={t} className={`chip tag clickable${applied?.tags === t ? ' on' : ''}`} title={d} onClick={() => applyTag(t)}>{t}</span>
        ))}
        {pending ? <span className="muted" style={{ fontSize: 12.5 }}>Press Enter or Apply to filter</span> : null}
      </form>
      {error ? <div className="panel"><Empty title="Couldn't collect evals"><pre className="trace" style={{ textAlign: 'left' }}>{error}</pre></Empty></div> : null}
      {items === null && !error ? <Empty title="Collecting evals…"><Spinner /></Empty> : null}
      {items && !items.length ? (
        <div className="panel"><Empty title="No evals match">Evals are tests that use fasteval decorators, e.g. <span className="kbd">@fe.correctness()</span>.</Empty></div>
      ) : null}
      {[...files.entries()].map(([file, rows]) => (
        <div className="panel" key={file} style={{ marginBottom: 12 }}>
          <div className="row" style={{ padding: '10px 12px', borderBottom: '1px solid var(--border)' }}>
            <b className="mono">{file}</b><span className="muted">{rows.length} evals</span><span className="spacer" />
            <button className="btn small" onClick={() => run({ paths: [file] })}>▶ Run file</button>
          </div>
          <table className="list">
            <tbody>
              {rows.map((r) => {
                const first = r.cases[0]
                const caseTags = [...new Set(r.cases.flatMap((c) => c.tags))]
                return (
                  <tr key={r.func}>
                    <td style={{ width: '30%' }}><b>{r.func}</b>{r.cases.length > 1 ? <span className="muted"> ×{r.cases.length}</span> : null}</td>
                    <td>{caseTags.map((t) => <span key={t} className="chip tag" style={{ marginRight: 4 }}>{t}</span>)}</td>
                    <td>{first.metrics.map((m) => <span key={m.name} className="chip" style={{ marginRight: 4 }} title={m.type}>{m.name} ≥ {m.threshold}</span>)}</td>
                    <td className="muted">{dataLabel(first.data)}{first.human_review ? ' · human review' : ''}</td>
                    <td style={{ textAlign: 'right' }}><button className="btn small" onClick={() => run({ paths: [r.nodeid] })}>▶ Run</button></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}
    </>
  )
}
