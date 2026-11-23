import { fmtMs, messageText } from '../format'
import type { EvalResult, MetricResult } from '../types'
import { CopyButton } from '../components/ui'

/** Every LLM call a metric's judge made: who judged, the prompt, the answer. */
export function MetricJudgeCalls({ metric, onTune }: { metric: MetricResult; onTune?: () => void }) {
  const calls = metric.details?.judge_calls ?? []
  return (
    <div className="mcard">
      <div className="row">
        <b>{metric.metric_name}</b>
        {metric.details?.prompt_judge ? <span className="chip tag">judge {metric.details.prompt_judge.name}</span> : null}
        <span className="muted">{calls.length} call{calls.length === 1 ? '' : 's'}</span>
        <span className="spacer" />
        {onTune ? <button className="btn small" onClick={onTune}>✎ Tune judge in playground</button> : null}
      </div>
      {calls.map((call, i) => {
        const prompt = call.messages.map((m) => `${m.role ?? 'user'}:\n${messageText(m.content)}`).join('\n\n')
        return (
          <div key={i} className={`call${call.error ? ' err' : ''}`}>
            <div className="row" style={{ fontSize: 13 }}>
              <b>Attempt {i + 1}</b>
              <span className="chip">{call.judge?.model ?? 'unknown model'}</span>
              <span className="chip">{call.judge?.provider ?? 'client'}</span>
              {call.judge?.temperature != null ? <span className="chip">temp {call.judge.temperature}</span> : null}
              <span className="muted mono">{fmtMs(call.latency_ms)}</span>
              {call.error ? <span style={{ color: 'var(--error)' }}>{call.error}</span> : null}
            </div>
            <details>
              <summary>Prompt · {call.messages.length} message{call.messages.length === 1 ? '' : 's'} · {prompt.length.toLocaleString()} chars</summary>
              <div className="row" style={{ justifyContent: 'flex-end' }}><CopyButton text={prompt} label="Copy prompt" /></div>
              {call.messages.map((m, j) => (
                <div key={j} className="msg">
                  <div className="role">{m.role ?? 'user'}</div>
                  <pre>{messageText(m.content)}</pre>
                </div>
              ))}
            </details>
            {call.response != null ? (
              <details open>
                <summary>Raw response</summary>
                <pre className="trace">{call.response}</pre>
              </details>
            ) : null}
          </div>
        )
      })}
    </div>
  )
}

export function JudgesTab({ evals, onTune }: { evals: EvalResult[]; onTune: (evalIndex: number, metric: MetricResult) => void }) {
  const withCalls = evals.flatMap((e, ei) =>
    (e?.metric_results ?? []).filter((m) => m.details?.judge_calls?.length).map((m) => ({ ei, m })),
  )
  if (!withCalls.length) {
    return (
      <div className="muted">
        No LLM judge calls for this case. Deterministic metrics (exact match, ROUGE, regex…) score without a judge;
        their reasoning is on the Overview tab.
      </div>
    )
  }
  return (
    <>
      {withCalls.map(({ ei, m }) => (
        <div key={`${ei}-${m.metric_name}`}>
          {evals.length > 1 ? <h3>Score #{ei + 1}</h3> : null}
          <MetricJudgeCalls metric={m} onTune={() => onTune(ei, m)} />
        </div>
      ))}
    </>
  )
}
