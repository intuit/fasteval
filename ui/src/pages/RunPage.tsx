import { useEffect, useMemo, useState } from 'react'
import { api, runPath, STATIC } from '../api'
import { Empty, Modal, Spinner } from '../components/ui'
import { href, navigate } from '../router'
import { caseScore, loadRun, runStats, useRunStream, type RunStats } from '../runState'
import type { RunSummary } from '../types'
import { CaseDetail } from './CaseDetail'
import { CaseList, type CaseFilters } from './CaseList'
import { RunHeader } from './RunHeader'

function ConsoleLog({ runId, live, onClose }: { runId: string; live: boolean; onClose: () => void }) {
  const [text, setText] = useState<string | null>(null)
  useEffect(() => {
    let timer = 0
    const load = () => api.text(runPath(runId, '/log')).then((t) => {
      setText(t)
      if (live) timer = window.setTimeout(load, 1500)
    })
    load()
    return () => clearTimeout(timer)
  }, [runId, live])
  return (
    <Modal title="pytest console" onClose={onClose}>
      <pre className="trace" style={{ maxHeight: '60vh' }}>{text ?? 'Loading…'}</pre>
    </Modal>
  )
}

export function RunPage({ runId, nodeid }: { runId: string; nodeid: string | null }) {
  const run = useRunStream(runId)
  const [info, setInfo] = useState<RunSummary | null>(STATIC ? STATIC.run : null)
  const [filters, setFilters] = useState<CaseFilters>({ search: '', status: 'all', tag: null, metric: null })
  const [base, setBase] = useState<{ id: string; stats: RunStats; scores: Map<string, number> } | null>(null)
  const [showLog, setShowLog] = useState(false)
  const status = run?.status

  // Run metadata (origin, request, command). Refetch when the run finishes.
  useEffect(() => {
    if (STATIC) return
    api.get<RunSummary>(runPath(runId)).then(setInfo, () => setInfo(null))
  }, [runId, status])

  // Re-runs (e.g. with a swapped judge) compare against the run they came from.
  const rerunOf: string | undefined = info?.meta?.rerun_of
  useEffect(() => {
    setBase(null)
    if (!rerunOf || STATIC) return
    let alive = true
    loadRun(rerunOf).then((r) => {
      if (!alive) return
      const scores = new Map<string, number>()
      for (const t of r.tests.values()) {
        const s = caseScore(t)
        if (s) scores.set(t.nodeid, s.score)
      }
      setBase({ id: rerunOf, stats: runStats(r), scores })
    })
    return () => { alive = false }
  }, [rerunOf])

  const selected = useMemo(() => (run && nodeid ? run.tests.get(nodeid) ?? null : null), [run, nodeid, run?.version])

  if (!run) return <Empty title="Loading run…"><Spinner /></Empty>

  return (
    <>
      <RunHeader
        run={run}
        info={info}
        base={base}
        metricFilter={filters.metric}
        onMetricFilter={(metric) => setFilters({ ...filters, metric, status: metric ? 'all' : filters.status })}
        onShowLog={() => setShowLog(true)}
      />
      <div className="split">
        <CaseList
          run={run}
          filters={filters}
          onFilters={setFilters}
          selected={nodeid}
          onSelect={(id) => navigate(href.run(runId, id))}
          baseScores={base?.scores ?? null}
        />
        <CaseDetail key={nodeid ?? ''} run={run} test={selected} />
      </div>
      {showLog ? <ConsoleLog runId={runId} live={run.status === 'running'} onClose={() => setShowLog(false)} /> : null}
    </>
  )
}
