import type { StaticData } from './types'

declare global {
  interface Window {
    FE_STATIC?: StaticData
  }
}

/** Embedded data when this is a static (headless) report; null when served live. */
export const STATIC: StaticData | null = window.FE_STATIC ?? null

let token: string | null = null

/** The per-server session token required on POSTs (fetched once). */
async function sessionToken(): Promise<string> {
  if (token === null) {
    const res = await fetch('/api/session')
    if (!res.ok) throw new Error('Could not start a UI session')
    token = ((await res.json()) as { token: string }).token
  }
  return token
}

export class ApiError extends Error {}

async function request<T>(path: string, method: 'GET' | 'POST', body?: unknown): Promise<T> {
  if (STATIC) throw new ApiError('Not available in a static report. Open it with `fasteval ui`.')
  const init: RequestInit = { method, headers: {} }
  if (method === 'POST') {
    init.headers = { 'Content-Type': 'application/json', 'X-Fasteval-Token': await sessionToken() }
    init.body = JSON.stringify(body ?? {})
  }
  const res = await fetch(path, init)
  const text = await res.text()
  let data: unknown = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!res.ok) {
    const message = (data as { error?: string } | null)?.error ?? `${res.status} ${res.statusText}`
    throw new ApiError(message)
  }
  return data as T
}

export const api = {
  get: <T,>(path: string) => request<T>(path, 'GET'),
  post: <T,>(path: string, body?: unknown) => request<T>(path, 'POST', body),
  text: async (path: string): Promise<string> => {
    const res = await fetch(path)
    return res.ok ? res.text() : ''
  },
}

export const runPath = (id: string, suffix = '') => `/api/runs/${encodeURIComponent(id)}${suffix}`
