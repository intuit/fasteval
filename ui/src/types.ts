// Shapes produced by the Python side (fasteval/testing/events.py, fasteval/ui/server.py).

export type Status =
  | 'queued' | 'running' | 'passed' | 'failed' | 'error' | 'skipped' | 'xfailed' | 'xpassed'

export type RunStatus = 'running' | 'passed' | 'failed' | 'interrupted' | 'stopped'

export interface JudgeCall {
  judge: { provider?: string; model?: string; temperature?: number }
  messages: Array<{ role?: string; content?: unknown }>
  response?: string
  error?: string
  latency_ms?: number
}

export interface MetricResult {
  metric_name: string
  score: number
  passed: boolean
  threshold: number
  reasoning?: string | null
  details?: {
    judge_calls?: JudgeCall[]
    raw_response?: string
    error?: string
    prompt_judge?: { name: string; based_on?: string | null }
    [key: string]: unknown
  }
}

export interface EvalInput {
  input?: string | null
  actual_output?: string | null
  expected_output?: string | null
  context?: string[] | null
  retrieval_context?: string[] | null
  history?: Array<{ role: string; content: string }> | null
  tool_calls?: unknown[]
  expected_tools?: unknown[]
  metadata?: Record<string, unknown>
  [key: string]: unknown
}

export interface EvalResult {
  eval_input: EvalInput
  metric_results: MetricResult[]
  passed: boolean
  aggregate_score: number
  execution_time_ms: number
  error?: string | null
  human_review?: { score: number; raw_input: string; skipped: boolean } | null
}

export interface InventoryMetric {
  name: string
  type: string
  threshold: number
  config?: Record<string, unknown>
  model?: string | null
  /** For prompt judges: the metric the judge was derived from. */
  based_on?: string | null
}

export interface InventoryItem {
  nodeid: string
  file: string
  function: string
  name: string
  tags: string[]
  metrics: InventoryMetric[]
  data?: { type: string; name?: string; path?: string; [k: string]: unknown } | null
  human_review?: boolean
  skipped?: boolean
}

export interface RunEvent {
  type: 'run_start' | 'collected' | 'test_start' | 'eval' | 'test_end' | 'run_end'
  ts: number
  [key: string]: any
}

export interface RunRequest {
  profile?: string | null
  tags?: string[]
  keyword?: string | null
  paths?: string[]
  last_failed?: boolean
  exitfirst?: boolean
  include_all?: boolean
  workers?: string | null
  judges?: string[]
  extra_args?: string[]
}

export interface RunSummary {
  id: string
  origin?: 'cli' | 'ui'
  profile?: string
  command?: string
  started_at?: string
  finished_at?: string
  status: RunStatus
  exit_status?: number
  gate_failures?: string[]
  request?: RunRequest
  total?: number
  passed?: number
  failed?: number
  pass_rate?: number
  avg_score?: number
  meta?: Record<string, any>
  has_log?: boolean
}

export interface ProfileInfo {
  name: string
  description?: string
  extends?: string | null
  paths: string[]
  tags?: string | null
  keyword?: string | null
  workers?: string | number | null
  min_score?: number | null
  min_pass_rate?: number | null
}

export interface Project {
  name: string
  root?: string
  config_source: string
  default_profile: string
  profiles: ProfileInfo[]
  tags: Record<string, string>
  judge_client?: string | null
  judges_dir?: string
  judges: string[]
}

export interface JudgeSpec {
  name: string
  prompt: string
  description?: string
  based_on?: string | null
  model?: string | null
  temperature?: number | null
  threshold?: number | null
  scoring?: 'continuous' | 'binary'
}

export interface StaticData {
  generated_at: string
  project: Project
  run: RunSummary
  events: RunEvent[]
}
