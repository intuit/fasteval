import { useEffect, useState } from 'react'
import { api } from '../api'
import { Empty, Spinner, useToast } from '../components/ui'
import { RunWithJudgeDialog } from './PgDialogs'
import { emptyPg, seedPg } from '../playgroundStore'
import { useProject } from '../project'
import { href, navigate } from '../router'
import type { JudgeSpec } from '../types'

export function JudgesPage() {
  const { project } = useProject()
  const [runWith, setRunWith] = useState<JudgeSpec | null>(null)
  const toast = useToast()
  const [data, setData] = useState<{ dir: string; judges: JudgeSpec[] } | null>(null)

  useEffect(() => {
    api.get<{ dir: string; judges: JudgeSpec[] }>('/api/judges').then(setData, (e: Error) => toast(e.message, true))
  }, [toast])

  const open = (j: JudgeSpec) => {
    seedPg({
      ...emptyPg(),
      template: j.prompt, originalTemplate: j.prompt, templateMethod: 'saved',
      name: j.name, description: j.description ?? '', model: j.model ?? '',
      temperature: j.temperature ?? 0, scoring: j.scoring ?? 'continuous', threshold: j.threshold ?? 0.5,
      metric: j.based_on ? { name: j.based_on, type: j.based_on, threshold: j.threshold ?? 0.5 } : null,
      savedAs: j.name, savedTemplate: j.prompt,
    })
    navigate(href.playground())
  }

  if (!data) return <Empty title="Loading judges…"><Spinner /></Empty>

  return (
    <>
      <div className="page-head">
        <h1>Judges</h1>
        <span className="muted">Prompt judges saved in <span className="mono">{data.dir}/</span></span>
        <span className="spacer" />
        <button className="btn" onClick={() => { seedPg({ ...emptyPg(), templateMethod: 'blank' }); navigate(href.playground()) }}>+ New judge</button>
      </div>
      {project?.judge_client ? (
        <div className="banner" style={{ marginBottom: 12 }}>Playground calls use <span className="mono">{project.judge_client}</span> unless a judge sets a model.</div>
      ) : null}
      {data.judges.length === 0 ? (
        <div className="panel">
          <Empty title="No saved judges yet">
            Open any LLM-judged case in a run, choose <b>Tune judge</b>, adjust the prompt, test it on real cases, then <b>Save judge</b>.
          </Empty>
        </div>
      ) : (
        <div className="cards">
          {data.judges.map((j) => (
            <div key={j.name} className="panel card">
              <div className="row"><b>{j.name}</b>{j.based_on ? <span className="chip">based on {j.based_on}</span> : null}</div>
              {j.description ? <div className="muted">{j.description}</div> : null}
              <div className="row" style={{ gap: 6 }}>
                <span className="chip">{j.model || 'default judge'}</span>
                <span className="chip">{j.scoring ?? 'continuous'}</span>
                {j.threshold != null ? <span className="chip">threshold {j.threshold}</span> : <span className="chip" title="Uses the threshold of the eval it replaces">eval's threshold</span>}
              </div>
              <pre className="trace" style={{ maxHeight: 120, fontSize: 11.5 }}>{j.prompt.slice(0, 600)}{j.prompt.length > 600 ? '…' : ''}</pre>
              <div className="row">
                <button className="btn small" onClick={() => open(j)}>✎ Open in playground</button>
                <button className="btn small" onClick={() => setRunWith(j)}>▶ Run evals with it</button>
                <span className="spacer" />
                <span className="kbd" title="Use in code">@fe.prompt_judge("{j.name}")</span>
              </div>
            </div>
          ))}
        </div>
      )}
      {runWith ? <RunWithJudgeDialog judgeName={runWith.name} defaultMetric={runWith.based_on} onClose={() => setRunWith(null)} /> : null}
    </>
  )
}
