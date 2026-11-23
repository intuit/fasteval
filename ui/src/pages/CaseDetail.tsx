import { useState } from 'react'
import { STATIC } from '../api'
import { Box, Empty, Gauge, Pill, Score } from '../components/ui'
import { asText } from '../format'
import { seedPg } from '../playgroundStore'
import { href, navigate } from '../router'
import { caseScore, type RunState, type TestState } from '../runState'
import type { EvalInput, EvalResult, MetricResult } from '../types'
import { JudgesTab } from './JudgeCalls'

type Tab = 'overview' | 'judges' | 'output'

function Inputs({ ei }: { ei: EvalInput }) {
  return (
    <>
      <div className="io">
        <Box title="Input">{ei.input ? <pre>{ei.input}</pre> : <span className="faint">—</span>}</Box>
        <Box title="Expected" kind="expected">{ei.expected_output ? <pre>{ei.expected_output}</pre> : <span className="faint">—</span>}</Box>
        <Box title="Actual" kind="actual">{ei.actual_output ? <pre>{ei.actual_output}</pre> : <span className="faint">—</span>}</Box>
      </div>
      {ei.context?.length ? <DocList title="Context" docs={ei.context} /> : null}
      {ei.retrieval_context?.length ? <DocList title="Retrieval context" docs={ei.retrieval_context} /> : null}
      {ei.history?.length ? (
        <Box title={`History · ${ei.history.length} turns`}>
          <div className="chat">
            {ei.history.map((turn, i) => <div key={i} className={`turn ${turn.role}`}><pre>{turn.content}</pre></div>)}
          </div>
        </Box>
      ) : null}
      {ei.tool_calls?.length || ei.expected_tools?.length ? (
        <div className="io">
          <Box title="Tool calls"><pre>{asText(ei.tool_calls)}</pre></Box>
          <Box title="Expected tools"><pre>{asText(ei.expected_tools)}</pre></Box>
        </div>
      ) : null}
      {ei.metadata && Object.keys(ei.metadata).length ? <Box title="Metadata"><pre>{asText(ei.metadata)}</pre></Box> : null}
    </>
  )
}

function DocList({ title, docs }: { title: string; docs: string[] }) {
  return (
    <Box title={`${title} · ${docs.length}`}>
      {docs.map((d, i) => <div key={i} className="msg"><div className="role">[{i + 1}]</div><pre>{d}</pre></div>)}
    </Box>
  )
}

function MetricCard({ m, onTune }: { m: MetricResult; onTune?: () => void }) {
  const calls = m.details?.judge_calls ?? []
  const judge = calls[calls.length - 1]?.judge
  const { judge_calls: _calls, raw_response: _raw, prompt_judge: pj, ...rest } = m.details ?? {}
  return (
    <div className={`mcard${m.passed ? '' : ' fail'}`}>
      <div className="row">
        <b>{m.metric_name}</b>
        <Pill status={m.passed ? 'passed' : 'failed'} />
        <Score value={m.score} passed={m.passed} />
        <span className="muted">threshold {m.threshold}</span>
        {judge ? <span className="chip" title={judge.provider}>judged by {judge.model ?? judge.provider}</span> : null}
        {pj ? <span className="chip tag">prompt judge {pj.name}</span> : null}
        <span className="spacer" />
        {onTune && calls.length ? <button className="btn small" onClick={onTune}>✎ Tune judge</button> : null}
      </div>
      <Gauge score={m.score} threshold={m.threshold} passed={m.passed} />
      {m.reasoning ? <div className="reasoning">{m.reasoning}</div> : null}
      {Object.keys(rest).length ? (
        <details><summary>Details</summary><pre className="trace">{asText(rest)}</pre></details>
      ) : null}
    </div>
  )
}

export function CaseDetail({ run, test }: { run: RunState; test: TestState | null }) {
  const [tab, setTab] = useState<Tab>('overview')
  if (!test) {
    return (
      <div className="panel detail">
        <Empty title="Select a case">See its input, expected and actual output, every metric's score and reasoning, and the judge prompts.</Empty>
      </div>
    )
  }
  const evals = test.evals.filter(Boolean) as EvalResult[]
  const s = caseScore(test)
  const judgeCalls = evals.reduce((n, e) => n + e.metric_results.reduce((k, m) => k + (m.details?.judge_calls?.length ?? 0), 0), 0)

  const tune = (evalIndex: number, m: MetricResult) => {
    const e = evals[evalIndex]
    const calls = m.details?.judge_calls ?? []
    const last = calls[calls.length - 1]
    // Start from the prompt that actually produced this verdict: a swapped-in
    // or @fe.prompt_judge judge, otherwise the eval's own metric.
    const pj = m.details?.prompt_judge
    const metric = pj
      ? { name: m.metric_name, type: 'prompt_judge', threshold: m.threshold, config: { judge: pj.name }, based_on: pj.based_on }
      : test.metrics.find((x) => x.name === m.metric_name) ?? { name: m.metric_name, type: '', threshold: m.threshold }
    seedPg({
      source: { runId: run.id, nodeid: test.nodeid, metricName: m.metric_name },
      metric,
      originalJudge: last?.judge ?? null,
      renderedPrompt: last ? last.messages.map((x) => (typeof x.content === 'string' ? x.content : '')).join('\n') : null,
      name: pj ? `${pj.name}_v2` : `${m.metric_name}_tuned`,
      model: '',
      threshold: m.threshold,
      cases: [{
        key: `${test.nodeid}#${evalIndex}`, label: test.name, runId: run.id, nodeid: test.nodeid, eval_input: e.eval_input,
        original: { score: m.score, passed: m.passed, reasoning: m.reasoning, judge: last?.judge },
      }],
    })
    navigate(href.playground())
  }

  return (
    <div className="panel detail">
      <div className="dhead">
        <div className="row">
          <Pill status={test.status} />
          <b style={{ fontSize: 15 }}>{test.name}</b>
          {s ? <Score value={s.score} passed={s.passed} /> : null}
          {test.duration != null ? <span className="muted mono">{test.duration.toFixed(2)}s</span> : null}
          {test.worker ? <span className="chip">{test.worker}</span> : null}
        </div>
        <div className="row" style={{ gap: 6 }}>
          <span className="mono faint" style={{ fontSize: 12 }}>{test.nodeid}</span>
          {test.tags.map((t) => <span key={t} className="chip tag">{t}</span>)}
          {test.data ? <span className="chip">{test.data.type}{test.data.name ? ` ${test.data.name}` : test.data.path ? ` ${test.data.path}` : ''}</span> : null}
        </div>
      </div>
      <div className="tabs">
        <button className={tab === 'overview' ? 'on' : ''} onClick={() => setTab('overview')}>Overview</button>
        <button className={tab === 'judges' ? 'on' : ''} onClick={() => setTab('judges')}>Judges{judgeCalls ? ` (${judgeCalls})` : ''}</button>
        <button className={tab === 'output' ? 'on' : ''} onClick={() => setTab('output')}>Output{test.longrepr ? ' ⚠' : ''}</button>
      </div>
      <div className="tabbody">
        {tab === 'overview' ? (
          evals.length ? (
            evals.map((e, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {evals.length > 1 ? <h3>Score #{i + 1} · {e.passed ? 'passed' : 'failed'} · {e.aggregate_score.toFixed(3)}</h3> : null}
                <Inputs ei={e.eval_input} />
                {e.metric_results.map((m) => <MetricCard key={m.metric_name} m={m} onTune={STATIC ? undefined : () => tune(i, m)} />)}
                {e.human_review ? <div className="banner">Human review: {e.human_review.skipped ? 'skipped' : `score ${e.human_review.score} (“${e.human_review.raw_input}”)`}</div> : null}
              </div>
            ))
          ) : (
            <div className="muted">
              {test.status === 'running' || test.status === 'queued' ? 'Waiting for results…' : test.longrepr ? 'No score recorded: the test failed before fe.score() ran. See the Output tab.' : 'No fasteval scores recorded for this case.'}
            </div>
          )
        ) : null}
        {tab === 'judges' ? <JudgesTab evals={evals} onTune={STATIC ? () => undefined : tune} /> : null}
        {tab === 'output' ? (
          <>
            {test.longrepr ? <Box title="Failure"><pre>{test.longrepr}</pre></Box> : <div className="muted">No failure output.</div>}
            {test.stdout ? <Box title="stdout"><pre>{test.stdout}</pre></Box> : null}
            {test.stderr ? <Box title="stderr"><pre>{test.stderr}</pre></Box> : null}
          </>
        ) : null}
      </div>
    </div>
  )
}
