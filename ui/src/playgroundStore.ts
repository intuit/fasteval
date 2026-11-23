import type { EvalInput, InventoryMetric, JudgeCall } from './types'

/** A case in the playground: inputs plus how the original judge scored it. */
export interface PgCase {
  key: string
  label: string
  runId?: string
  nodeid?: string
  eval_input: EvalInput
  original?: { score: number; passed: boolean; reasoning?: string | null; judge?: JudgeCall['judge'] }
  /** Optional human label: what the judge *should* decide. */
  expected?: 'pass' | 'fail' | null
  result?: PgResult | null
}

export interface PgResult {
  score?: number
  passed?: boolean
  reasoning?: string | null
  prompt?: string
  response?: string | null
  error?: string | null
  latency_ms?: number
}

export interface PgState {
  /** Where the playground was opened from. */
  source?: { runId: string; nodeid: string; metricName: string } | null
  metric?: InventoryMetric | null
  originalJudge?: JudgeCall['judge'] | null
  renderedPrompt?: string | null
  template: string
  originalTemplate: string
  templateMethod?: 'metric' | 'inferred' | 'saved' | 'blank'
  name: string
  description: string
  model: string
  temperature: number
  scoring: 'continuous' | 'binary'
  threshold: number
  cases: PgCase[]
  /** Name the current template was last saved under, and the text saved. */
  savedAs?: string | null
  savedTemplate?: string | null
}

const KEY = 'fasteval.playground.v1'

export function emptyPg(): PgState {
  return {
    template: '', originalTemplate: '', name: '', description: '', model: '',
    temperature: 0, scoring: 'continuous', threshold: 0.5, cases: [],
  }
}

export function loadPg(): PgState {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? { ...emptyPg(), ...(JSON.parse(raw) as PgState) } : emptyPg()
  } catch {
    return emptyPg()
  }
}

export function savePg(state: PgState): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(state))
  } catch {
    /* storage full or disabled: the playground still works in memory */
  }
}

/** Replace the playground with a fresh seed (from "Tune judge"). */
export function seedPg(seed: Partial<PgState>): void {
  savePg({ ...emptyPg(), ...seed })
}
