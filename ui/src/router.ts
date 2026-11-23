import { useEffect, useState } from 'react'
import { STATIC } from './api'

export type Route =
  | { page: 'runs' }
  | { page: 'run'; runId: string; nodeid: string | null }
  | { page: 'evals' }
  | { page: 'judges' }
  | { page: 'playground' }

export function parseHash(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent)
  if (STATIC) {
    // A static report shows exactly one run; a case may be selected.
    return { page: 'run', runId: STATIC.run.id, nodeid: parts[0] === 'runs' ? parts[2] ?? null : null }
  }
  switch (parts[0]) {
    case 'runs':
      return parts[1] ? { page: 'run', runId: parts[1], nodeid: parts[2] ?? null } : { page: 'runs' }
    case 'evals':
      return { page: 'evals' }
    case 'judges':
      return { page: 'judges' }
    case 'playground':
      return { page: 'playground' }
    default:
      return { page: 'runs' }
  }
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseHash(location.hash))
  useEffect(() => {
    const onChange = () => setRoute(parseHash(location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export const href = {
  runs: () => '#/runs',
  run: (id: string, nodeid?: string | null) =>
    `#/runs/${encodeURIComponent(id)}${nodeid ? `/${encodeURIComponent(nodeid)}` : ''}`,
  evals: () => '#/evals',
  judges: () => '#/judges',
  playground: () => '#/playground',
}

export const navigate = (hash: string) => {
  if (location.hash !== hash) location.hash = hash
}
