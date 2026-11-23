import { useEffect, useRef, useState } from 'react'
import { runPath, STATIC } from './api'
import type { EvalResult, InventoryItem, RunEvent, RunStatus, Status } from './types'

export interface TestState {
  nodeid: string
  file: string
  func: string
  name: string
  tags: string[]
  metrics: InventoryItem['metrics']
  data: InventoryItem['data']
  status: Status
  evals: EvalResult[]
  duration: number | null
  longrepr: string | null
  stdout: string | null
  stderr: string | null
  startTs: number | null
  endTs: number | null
  worker?: string | null
}

export interface RunState {
  id: string
  start: RunEvent | null
  end: RunEvent | null
  total: number | null
  tests: Map<string, TestState>
  status: RunStatus
  firstTs: number | null
  lastTs: number | null
  version: number
}

export function newRun(id: string): RunState {
  return { id, start: null, end: null, total: null, tests: new Map(), status: 'running', firstTs: null, lastTs: null, version: 0 }
}

function testFor(r: RunState, nodeid: string): TestState {
  let t = r.tests.get(nodeid)
  if (!t) {
    const parts = nodeid.split('::')
    const name = parts[parts.length - 1]
    t = {
      nodeid, file: parts[0], func: name.split('[')[0], name, tags: [], metrics: [], data: null,
      status: 'queued', evals: [], duration: null, longrepr: null, stdout: null, stderr: null,
      startTs: null, endTs: null,
    }
    r.tests.set(nodeid, t)
  }
  return t
}

const BAD: Status[] = ['failed', 'error']

/** Fold one event into the run state (mutates; bump version for React). */
export function applyEvent(r: RunState, ev: RunEvent): void {
  r.version++
  if (ev.ts) {
    r.firstTs = r.firstTs ?? ev.ts
    r.lastTs = ev.ts
  }
  switch (ev.type) {
    case 'run_start':
      r.start = ev
      break
    case 'collected':
      r.total = ev.total
      for (const item of (ev.items ?? []) as InventoryItem[]) {
        const t = testFor(r, item.nodeid)
        Object.assign(t, { file: item.file, func: item.function, name: item.name, tags: item.tags ?? [], metrics: item.metrics ?? [], data: item.data })
      }
      break
    case 'test_start': {
      const t = testFor(r, ev.nodeid)
      if (t.status === 'queued') t.status = 'running'
      t.startTs = ev.ts
      break
    }
    case 'eval': {
      const t = testFor(r, ev.nodeid)
      t.evals[ev.index ?? 0] = ev.result as EvalResult
      break
    }
    case 'test_end': {
      const t = testFor(r, ev.nodeid)
      // A teardown error turns a pass into an error; never downgrade a failure.
      if (ev.when !== 'teardown' || ev.outcome === 'error') {
        if (!(BAD.includes(t.status) && ev.when === 'teardown')) t.status = ev.outcome as Status
      }
      if (ev.when !== 'teardown' || t.duration === null) t.duration = ev.duration
      if (ev.longrepr) t.longrepr = t.longrepr ? `${t.longrepr}\n\n${ev.longrepr}` : ev.longrepr
      t.stdout = ev.stdout ?? t.stdout
      t.stderr = ev.stderr ?? t.stderr
      t.endTs = ev.ts
      t.worker = ev.worker_id ?? t.worker
      break
    }
    case 'run_end':
      r.end = ev
      break
  }
}

export function finishRun(r: RunState, status?: RunStatus): void {
  r.status = status ?? (r.end ? (r.end.exit_status === 0 ? 'passed' : 'failed') : 'interrupted')
  r.version++
}

/** Live state of a run: replays its event log, then follows it until the end. */
export function useRunStream(runId: string | null): RunState | null {
  const [, setTick] = useState(0)
  const ref = useRef<RunState | null>(null)

  useEffect(() => {
    if (!runId) return
    const r = newRun(runId)
    ref.current = r
    let frame = 0
    const kick = () => {
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; setTick((n) => n + 1) })
    }
    if (STATIC) {
      for (const ev of STATIC.events) applyEvent(r, ev)
      finishRun(r, STATIC.run.status)
      kick()
      return () => cancelAnimationFrame(frame)
    }
    const es = new EventSource(runPath(runId, '/stream'))
    es.onmessage = (m) => {
      try { applyEvent(r, JSON.parse(m.data)) } catch { /* ignore a malformed line */ }
      kick()
    }
    es.addEventListener('end', (m) => {
      es.close()
      const data = JSON.parse((m as MessageEvent).data || '{}') as { status?: RunStatus; stopped?: boolean }
      finishRun(r, data.stopped ? 'stopped' : data.status)
      kick()
    })
    return () => { es.close(); cancelAnimationFrame(frame) }
  }, [runId])

  return ref.current && ref.current.id === runId ? ref.current : null
}

/** Load a whole (finished) run once, e.g. to compare against. */
export function loadRun(runId: string): Promise<RunState> {
  return new Promise((resolve) => {
    const r = newRun(runId)
    const es = new EventSource(runPath(runId, '/stream'))
    es.onmessage = (m) => { try { applyEvent(r, JSON.parse(m.data)) } catch { /* skip */ } }
    es.addEventListener('end', () => { es.close(); finishRun(r); resolve(r) })
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) { finishRun(r); resolve(r) } }
  })
}

// ── aggregates ────────────────────────────────────────────────────────────

export function caseScore(t: TestState): { score: number; passed: boolean } | null {
  const evals = t.evals.filter(Boolean)
  if (!evals.length) return null
  return { score: evals.reduce((a, e) => a + e.aggregate_score, 0) / evals.length, passed: evals.every((e) => e.passed) }
}

export interface MetricAgg { name: string; n: number; pass: number; sum: number }

export interface RunStats {
  counts: Record<'passed' | 'failed' | 'error' | 'skipped' | 'running' | 'queued', number>
  total: number
  done: number
  results: number
  passRate: number | null
  avg: number | null
  judgeCalls: number
  judgeMs: number
  metrics: MetricAgg[]
}

export function runStats(r: RunState): RunStats {
  const counts = { passed: 0, failed: 0, error: 0, skipped: 0, running: 0, queued: 0 }
  const metrics = new Map<string, MetricAgg>()
  let results = 0, passed = 0, sum = 0, judgeCalls = 0, judgeMs = 0
  for (const t of r.tests.values()) {
    const key = (t.status in counts ? t.status : t.status.startsWith('x') ? 'skipped' : 'queued') as keyof typeof counts
    counts[key]++
    for (const e of t.evals) {
      if (!e) continue
      results++
      if (e.passed) passed++
      sum += e.aggregate_score
      for (const m of e.metric_results ?? []) {
        const a = metrics.get(m.metric_name) ?? { name: m.metric_name, n: 0, pass: 0, sum: 0 }
        a.n++
        if (m.passed) a.pass++
        a.sum += m.score
        metrics.set(m.metric_name, a)
        for (const c of m.details?.judge_calls ?? []) { judgeCalls++; judgeMs += c.latency_ms ?? 0 }
      }
    }
  }
  const total = r.total ?? r.tests.size
  return {
    counts, total, done: counts.passed + counts.failed + counts.error + counts.skipped,
    results, passRate: results ? passed / results : null, avg: results ? sum / results : null,
    judgeCalls, judgeMs, metrics: [...metrics.values()],
  }
}

export function failedCount(r: RunState): number {
  let n = 0
  for (const t of r.tests.values()) if (BAD.includes(t.status)) n++
  return n
}
