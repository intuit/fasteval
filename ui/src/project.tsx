import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, STATIC } from './api'
import type { Project, RunRequest } from './types'

interface ProjectCtx {
  project: Project | null
  error: string | null
  reload: () => void
  /** Open the New run dialog, optionally prefilled. */
  openNewRun: (prefill?: RunRequest) => void
  newRunPrefill: RunRequest | null
  closeNewRun: () => void
}

const Ctx = createContext<ProjectCtx | null>(null)

export function useProject(): ProjectCtx {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useProject outside ProjectProvider')
  return ctx
}

export function ProjectProvider({ children }: { children: ReactNode }) {
  const [project, setProject] = useState<Project | null>(STATIC ? STATIC.project : null)
  const [error, setError] = useState<string | null>(null)
  const [newRunPrefill, setPrefill] = useState<RunRequest | null>(null)

  const reload = useCallback(() => {
    if (STATIC) return
    api.get<Project>('/api/project').then(
      (p) => { setProject(p); setError(null) },
      (e: Error) => setError(e.message),
    )
  }, [])
  useEffect(reload, [reload])

  const value: ProjectCtx = {
    project,
    error,
    reload,
    newRunPrefill,
    openNewRun: (prefill) => setPrefill(prefill ?? {}),
    closeNewRun: () => setPrefill(null),
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
