import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { Delta, Empty, Spinner, useToast } from '../components/ui'
import { fmtPct, fmtScore } from '../format'
import { emptyPg, loadPg, savePg, type PgCase, type PgResult, type PgState } from '../playgroundStore'
import { useProject } from '../project'
import { href } from '../router'
import { AddCasesDialog, PgCaseCard } from './PgCases'
import { RunWithJudgeDialog, SaveJudgeDialog } from './PgDialogs'

const VARS = ['input', 'actual_output', 'expected_output', 'context', 'retrieval_context', 'history', 'tool_calls', 'expected_tools', 'metadata']

function usedVars(template: string): string[] {
  return VARS.filter((v) => new RegExp(`\\{\\{\\s*${v}\\s*\\}\\}`).test(template))
}

function avg(xs: Array<number | undefined>): number | null {
  const ns = xs.filter((x): x is number => typeof x === 'number')
  return ns.length ? ns.reduce((a, b) => a + b, 0) / ns.length : null
}

function agreement(cases: PgCase[], pick: (c: PgCase) => boolean | undefined): string | null {
  const labeled = cases.filter((c) => c.expected && pick(c) !== undefined)
  if (!labeled.length) return null
  const ok = labeled.filter((c) => pick(c) === (c.expected === 'pass')).length
  return `${ok}/${labeled.length}`
}

export function Playground() {
  const { project } = useProject()
  const toast = useToast()
  const [pg, setPgRaw] = useState<PgState>(loadPg)
  const [testing, setTesting] = useState(false)
  const [loadingTpl, setLoadingTpl] = useState(false)
  // 'save-run': "Run evals with this judge" on an unsaved judge saves first, then runs.
  const [dialog, setDialog] = useState<'add' | 'save' | 'save-run' | 'run' | null>(null)
  const textarea = useRef<HTMLTextAreaElement>(null)

  const setPg = (patch: Partial<PgState> | ((s: PgState) => PgState)) =>
    setPgRaw((s) => { const next = typeof patch === 'function' ? patch(s) : { ...s, ...patch }; savePg(next); return next })

  // Seeded from a run: derive an editable template from the metric's real prompt.
  useEffect(() => {
    if (pg.template || (!pg.metric && !pg.renderedPrompt)) return
    setLoadingTpl(true)
    api.post<{ template: string; method: 'metric' | 'inferred' }>('/api/judges/template', {
      metric_type: pg.metric?.type, metric_name: pg.metric?.name, threshold: pg.metric?.threshold,
      metric_config: pg.metric?.config, rendered_prompt: pg.renderedPrompt, eval_input: pg.cases[0]?.eval_input,
    }).then(
      (d) => setPg({ template: d.template, originalTemplate: d.template, templateMethod: d.method }),
      (e: Error) => toast(`Couldn't load the judge prompt: ${e.message}`, true),
    ).finally(() => setLoadingTpl(false))
  }, []) // run once for the seeded state

  function insertVar(v: string) {
    const el = textarea.current
    const token = `{{${v}}}`
    if (!el) { setPg({ template: pg.template + token }); return }
    const { selectionStart: a, selectionEnd: b } = el
    const next = pg.template.slice(0, a) + token + pg.template.slice(b)
    setPg({ template: next })
    requestAnimationFrame(() => { el.focus(); el.selectionStart = el.selectionEnd = a + token.length })
  }

  async function test() {
    if (!pg.cases.length) { setDialog('add'); return }
    setTesting(true)
    setPg((s) => ({ ...s, cases: s.cases.map((c) => ({ ...c, result: null })) }))
    try {
      const res = await api.post<{ judge: { model?: string; provider?: string }; results: Array<PgResult & { key: string }> }>('/api/judges/test', {
        template: pg.template, model: pg.model || null, temperature: pg.temperature, scoring: pg.scoring, threshold: pg.threshold,
        cases: pg.cases.map((c) => ({ key: c.key, eval_input: c.eval_input })),
      })
      const byKey = new Map(res.results.map((r) => [r.key, r]))
      setPg((s) => ({ ...s, cases: s.cases.map((c) => ({ ...c, result: byKey.get(c.key) ?? { error: 'No result' } })) }))
      toast(`Tested ${res.results.length} cases with ${res.judge.model ?? res.judge.provider}`)
    } catch (e) {
      toast((e as Error).message, true)
    } finally {
      setTesting(false)
    }
  }

  const tested = pg.cases.filter((c) => c.result && c.result.score !== undefined)
  const flips = tested.filter((c) => c.original && c.original.passed !== c.result!.passed).length
  const used = usedVars(pg.template)
  const dirty = pg.template !== pg.originalTemplate
  const unsaved = !pg.savedAs || pg.savedTemplate !== pg.template
  const defaultJudge = project?.judge_client ?? 'default provider (env keys / fe.set_default_provider)'

  if (loadingTpl) return <Empty title="Loading the judge's prompt…"><Spinner /></Empty>

  return (
    <>
      <div className="page-head">
        <h1>Judge playground</h1>
        {pg.source ? (
          <span className="muted">
            From <a href={href.run(pg.source.runId, pg.source.nodeid)}><u className="mono">{pg.source.runId}</u></a> · metric <b>{pg.source.metricName}</b>
            {pg.originalJudge?.model ? <> · originally judged by <b>{pg.originalJudge.model}</b></> : null}
          </span>
        ) : <span className="muted">Write or tune an LLM judge prompt and test it on real cases.</span>}
        <span className="spacer" />
        {pg.template || pg.cases.length ? (
          <button className="btn small ghost" onClick={() => {
            if (window.confirm('Clear the prompt, settings and test cases?')) setPg(() => emptyPg())
          }}>Start over</button>
        ) : null}
      </div>

      <div className="pg">
        <div className="panel editor">
          <div className="grid">
            <label className="field">Judge model
              <input type="text" value={pg.model} placeholder="default" onChange={(e) => setPg({ model: e.target.value })} list="fe-models" />
              <datalist id="fe-models">
                {['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'o3-mini', 'claude-sonnet-4-6', 'claude-haiku-4-5'].map((m) => <option key={m} value={m} />)}
              </datalist>
            </label>
            <label className="field">Temperature
              <input type="number" min={0} max={2} step={0.1} value={pg.temperature} onChange={(e) => setPg({ temperature: Number(e.target.value) })} />
            </label>
            <label className="field">Scoring
              <select value={pg.scoring} onChange={(e) => setPg({ scoring: e.target.value as PgState['scoring'] })}>
                <option value="continuous">continuous (0–1)</option>
                <option value="binary">binary (pass/fail)</option>
              </select>
            </label>
            <label className="field">Threshold
              <input type="number" min={0} max={1} step={0.05} value={pg.threshold} onChange={(e) => setPg({ threshold: Number(e.target.value) })} />
            </label>
          </div>
          <div className="muted" style={{ fontSize: 12.5 }}>
            {pg.model ? <>Calls <b>{pg.model}</b> directly (needs its API key in the server's environment).</> : <>Empty model uses <span className="mono">{defaultJudge}</span>.</>}
          </div>

          <div className="row">
            <h3 style={{ margin: 0 }}>Prompt template</h3>
            {pg.templateMethod === 'inferred' ? <span className="chip" title="Rebuilt from the rendered prompt by replacing this case's values">inferred</span> : null}
            {dirty ? <span className="chip tag">edited</span> : null}
            <span className="spacer" />
            <button className="btn small ghost" disabled={!dirty} onClick={() => setPg({ template: pg.originalTemplate })}>Reset to original</button>
          </div>
          <div className="vars">
            {VARS.map((v) => (
              <span key={v} className={`chip clickable${used.includes(v) ? ' tag' : ''}`} title="Insert at cursor" onClick={() => insertVar(v)}>{`{{${v}}}`}</span>
            ))}
          </div>
          <textarea ref={textarea} className="tpl" spellCheck={false} value={pg.template}
            placeholder={'Grade the OUTPUT against the EXPECTED answer.\n\nINPUT: {{input}}\nOUTPUT: {{actual_output}}\nEXPECTED: {{expected_output}}\n\nRespond with JSON: {"score": <0.0-1.0>, "reasoning": "<why>"}'}
            onChange={(e) => setPg({ template: e.target.value })} />
          {pg.template && !used.includes('actual_output') ? <div className="banner warn">The template never shows the judge <span className="mono">{'{{actual_output}}'}</span>, the output being graded.</div> : null}
          {pg.template && !/score/i.test(pg.template) ? <div className="banner warn">Ask for JSON with a <span className="mono">score</span> (and <span className="mono">reasoning</span>) so fasteval can parse the verdict.</div> : null}

          <div className="row">
            <button className="btn primary" disabled={testing || !pg.template.trim()} onClick={test}>
              {testing ? <Spinner /> : '▶'} Test on {pg.cases.length || ''} case{pg.cases.length === 1 ? '' : 's'}
            </button>
            <button className="btn" disabled={!pg.template.trim()} onClick={() => setDialog('save')}>💾 Save judge{pg.savedAs && !unsaved ? ` (${pg.savedAs})` : ''}</button>
            <button className="btn" disabled={!pg.template.trim()} title={unsaved ? 'Saves the judge, then runs evals with it' : ''}
              onClick={() => setDialog(unsaved ? 'save-run' : 'run')}>▶ Run evals with this judge</button>
          </div>
        </div>

        <div className="cases-col">
          {tested.length ? (
            <div className="panel summary-bar">
              <div><div className="k">Avg score</div><div className="v">{fmtScore(avg(pg.cases.map((c) => c.original?.score)))} → {fmtScore(avg(tested.map((c) => c.result!.score)))} <Delta from={avg(pg.cases.map((c) => c.original?.score))} to={avg(tested.map((c) => c.result!.score))} /></div></div>
              <div><div className="k">Pass rate</div><div className="v">{fmtPct(tested.filter((c) => c.original?.passed).length / tested.length)} → {fmtPct(tested.filter((c) => c.result!.passed).length / tested.length)}</div></div>
              <div><div className="k">Verdicts flipped</div><div className="v">{flips}</div></div>
              {agreement(pg.cases, (c) => c.result?.passed) ? (
                <div><div className="k">Agrees with your labels</div><div className="v">{agreement(pg.cases, (c) => c.original?.passed) ?? '—'} → {agreement(pg.cases, (c) => c.result?.passed)}</div></div>
              ) : <div className="muted" style={{ fontSize: 12.5 }}>Label cases “should pass/fail” to measure how often each judge agrees with you.</div>}
            </div>
          ) : null}
          <div className="row">
            <h3 style={{ margin: 0 }}>Test cases ({pg.cases.length})</h3>
            <span className="spacer" />
            <button className="btn small" onClick={() => setDialog('add')}>+ Add cases from a run</button>
            {pg.cases.length ? <button className="btn small ghost" onClick={() => setPg({ cases: [] })}>Clear</button> : null}
          </div>
          {pg.cases.length === 0 ? (
            <div className="panel"><Empty title="No test cases">Add cases from a run: failures are the most useful, plus a few passes to make sure the judge doesn't get harsher everywhere.</Empty></div>
          ) : pg.cases.map((c) => (
            <PgCaseCard key={c.key} c={c}
              onChange={(next) => setPg((s) => ({ ...s, cases: s.cases.map((x) => (x.key === c.key ? next : x)) }))}
              onRemove={() => setPg((s) => ({ ...s, cases: s.cases.filter((x) => x.key !== c.key) }))} />
          ))}
        </div>
      </div>

      {dialog === 'add' ? (
        <AddCasesDialog metricName={pg.source?.metricName ?? pg.metric?.name ?? null} defaultRunId={pg.source?.runId}
          existing={new Set(pg.cases.map((c) => c.key))} onClose={() => setDialog(null)}
          onAdd={(cases) => setPg((s) => ({ ...s, cases: [...s.cases, ...cases] }))} />
      ) : null}
      {dialog === 'save' || dialog === 'save-run' ? (
        <SaveJudgeDialog pg={pg} onClose={() => setDialog(null)}
          onSaved={(name) => {
            setPg((s) => ({ ...s, savedAs: name, name, savedTemplate: s.template }))
            if (dialog === 'save-run') setTimeout(() => setDialog('run'), 0)
          }} />
      ) : null}
      {dialog === 'run' && pg.savedAs ? (
        <RunWithJudgeDialog judgeName={pg.savedAs} defaultMetric={pg.source?.metricName ?? pg.metric?.name}
          defaultRunId={pg.source?.runId} onClose={() => setDialog(null)} />
      ) : null}
    </>
  )
}
