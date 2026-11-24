import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Delta, Dot, Score } from '../components/ui'
import { caseScore, type RunState, type TestState } from '../runState'

export type StatusFilter = 'all' | 'failed' | 'passed' | 'active'

export interface CaseFilters {
  search: string
  status: StatusFilter
  tag: string | null
  metric: string | null
}

/** Every list row (file header or case) has this height, so the list can be windowed. */
const ROW_HEIGHT = 34
/** Extra rows rendered above/below the viewport to avoid flicker while scrolling. */
const OVERSCAN = 12
/** Tags shown before "+N more" collapses the rest. */
const TAGS_COLLAPSED = 8

const matches = (t: TestState, f: CaseFilters, q: string): boolean => {
  if (q && !t.nodeid.toLowerCase().includes(q) && !t.tags.some((x) => x.includes(q))) return false
  if (f.status === 'failed' && !['failed', 'error'].includes(t.status)) return false
  if (f.status === 'passed' && t.status !== 'passed') return false
  if (f.status === 'active' && !['running', 'queued'].includes(t.status)) return false
  if (f.tag && !t.tags.includes(f.tag)) return false
  if (f.metric && !t.evals.some((e) => e?.metric_results.some((m) => m.metric_name === f.metric && !m.passed))) return false
  return true
}

/** Split "test_x[case id]" into function name and case id (without brackets). */
function splitName(name: string): [string, string] {
  const i = name.indexOf('[')
  return i < 0 ? [name, ''] : [name.slice(0, i), name.slice(i + 1, name.endsWith(']') ? -1 : undefined)]
}

/** Characters kept visible at the end of a long label (case ids differ at the end). */
const TAIL_CHARS = 10

/** Middle-truncating label: the head shrinks with an ellipsis, the tail always shows. */
function MiddleEllipsis({ text }: { text: string }) {
  if (text.length <= TAIL_CHARS + 4) return <span className="mid-head">{text}</span>
  return (
    <>
      <span className="mid-head">{text.slice(0, -TAIL_CHARS)}</span>
      <span className="mid-tail">{text.slice(-TAIL_CHARS)}</span>
    </>
  )
}

type Row =
  | { kind: 'file'; file: string; count: number; func: string | null }
  | { kind: 'case'; test: TestState; showFunc: boolean }

interface Props {
  run: RunState
  filters: CaseFilters
  onFilters: (f: CaseFilters) => void
  selected: string | null
  onSelect: (nodeid: string) => void
  baseScores: Map<string, number> | null
}

function TagChips({ counts, selected, onToggle }: { counts: Array<[string, number]>; selected: string | null; onToggle: (tag: string) => void }) {
  const [expanded, setExpanded] = useState(false)
  if (!counts.length) return null
  const collapsible = counts.length > TAGS_COLLAPSED
  let shown = expanded || !collapsible ? counts : counts.slice(0, TAGS_COLLAPSED)
  // A selected tag stays visible even when collapsed.
  if (selected && !shown.some(([t]) => t === selected)) {
    const sel = counts.find(([t]) => t === selected)
    if (sel) shown = [sel, ...shown]
  }
  return (
    <div className={`tag-chips${expanded ? ' expanded' : ''}`}>
      {shown.map(([tag, n]) => (
        <span key={tag} className={`chip tag clickable${selected === tag ? ' on' : ''}`} title={`${n} cases`} onClick={() => onToggle(tag)}>
          {tag} <span className="n">{n}</span>
        </span>
      ))}
      {collapsible ? (
        <button className="btn small ghost" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'show fewer' : `+${counts.length - TAGS_COLLAPSED} more`}
        </button>
      ) : null}
    </div>
  )
}

export function CaseList({ run, filters, onFilters, selected, onSelect, baseScores }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewport, setViewport] = useState(600)
  const set = (patch: Partial<CaseFilters>) => onFilters({ ...filters, ...patch })

  // Recomputed when the run changes (run.version) or filters change; O(n).
  const { total, visibleCount, rows, tagCounts, multiFile } = useMemo(() => {
    const tests = [...run.tests.values()]
    const q = filters.search.toLowerCase()
    const counts = new Map<string, number>()
    const byFile = new Map<string, TestState[]>()
    let visibleCount = 0
    for (const t of tests) {
      for (const tag of t.tags) counts.set(tag, (counts.get(tag) ?? 0) + 1)
      if (!matches(t, filters, q)) continue
      visibleCount++
      let group = byFile.get(t.file)
      if (!group) byFile.set(t.file, (group = []))
      group.push(t)
    }
    const rows: Row[] = []
    for (const [file, items] of byFile) {
      // With one test function per file, rows show just the case id and the
      // function name moves to the file header.
      const funcs = new Set(items.map((t) => t.func))
      const single = funcs.size === 1 ? items[0].func : null
      rows.push({ kind: 'file', file, count: items.length, func: single })
      for (const test of items) rows.push({ kind: 'case', test, showFunc: single === null })
    }
    const tagCounts = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    return { total: tests.length, visibleCount, rows, tagCounts, multiFile: byFile.size > 1 }
  }, [run, run.version, filters])

  // Track the scroll viewport height.
  useLayoutEffect(() => {
    const el = scroller.current
    if (!el) return
    const update = () => setViewport(el.clientHeight || 600)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // Back to the top when the filters change the list.
  useEffect(() => {
    scroller.current?.scrollTo({ top: 0 })
    setScrollTop(0)
  }, [filters.search, filters.status, filters.tag, filters.metric])

  // Keep the selected case in view when it's chosen from outside the list
  // (URL, back/forward, "open in run" links).
  useEffect(() => {
    const el = scroller.current
    if (!el || !selected) return
    const index = rows.findIndex((r) => r.kind === 'case' && r.test.nodeid === selected)
    if (index < 0) return
    const top = index * ROW_HEIGHT
    if (top < el.scrollTop || top + ROW_HEIGHT > el.scrollTop + el.clientHeight) {
      el.scrollTo({ top: Math.max(0, top - el.clientHeight / 3) })
    }
  }, [selected]) // only when the selection changes, not on every live update

  const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const end = Math.min(rows.length, Math.ceil((scrollTop + viewport) / ROW_HEIGHT) + OVERSCAN)
  // The file the viewport is in, shown pinned when there are several files.
  let currentFile: Row | undefined
  for (let i = Math.min(Math.floor(scrollTop / ROW_HEIGHT), rows.length - 1); i >= 0; i--) {
    if (rows[i].kind === 'file') { currentFile = rows[i]; break }
  }

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
          <span className="muted" style={{ fontSize: 12 }}>{visibleCount.toLocaleString()}/{total.toLocaleString()}</span>
        </div>
        <TagChips counts={tagCounts} selected={filters.tag} onToggle={(tag) => set({ tag: filters.tag === tag ? null : tag })} />
        {filters.metric ? (
          <div className="row" style={{ fontSize: 12.5 }}>
            <span className="muted">Only cases failing</span><span className="chip">{filters.metric}</span>
            <button className="btn small ghost" onClick={() => set({ metric: null })}>clear</button>
          </div>
        ) : null}
      </div>
      {multiFile && currentFile && currentFile.kind === 'file' && scrollTop > 0 ? (
        <div className="file pinned" title={currentFile.file}>
          <span className="grow ellipsis">{currentFile.file}{currentFile.func ? <span className="func"> · {currentFile.func}</span> : null}</span>
          <span>{currentFile.count.toLocaleString()}</span>
        </div>
      ) : null}
      <div className="case-list" ref={scroller} onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}>
        {rows.length === 0 ? <div className="empty">{total ? 'No cases match these filters.' : 'Waiting for collection…'}</div> : null}
        <div style={{ height: rows.length * ROW_HEIGHT, position: 'relative' }}>
          {rows.slice(start, end).map((row, i) => {
            const top = (start + i) * ROW_HEIGHT
            if (row.kind === 'file') {
              return (
                <div key={`file:${row.file}`} className="file" style={{ top }} title={row.file}>
                  <span className="grow ellipsis">{row.file}{row.func ? <span className="func"> · {row.func}</span> : null}</span>
                  <span>{row.count.toLocaleString()}</span>
                </div>
              )
            }
            const t = row.test
            const s = caseScore(t)
            const [fn, cid] = splitName(t.name)
            return (
              <div key={t.nodeid} className={`case${selected === t.nodeid ? ' sel' : ''}`} style={{ top }} onClick={() => onSelect(t.nodeid)}>
                <Dot status={t.status} />
                <span className="nm" title={t.nodeid}>
                  {row.showFunc || !cid ? <span className="fn">{fn}</span> : null}
                  {cid ? <span className="cid"><MiddleEllipsis text={cid} /></span> : null}
                </span>
                {baseScores && s ? <Delta from={baseScores.get(t.nodeid)} to={s.score} /> : null}
                {s ? <Score value={s.score} passed={s.passed} /> : null}
                <span className="dur">{t.duration != null ? `${t.duration.toFixed(2)}s` : ''}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
