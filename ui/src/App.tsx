import { useState } from 'react'
import { STATIC } from './api'
import { NewRunDialog } from './components/NewRunDialog'
import { ToastProvider } from './components/ui'
import { EvalsPage } from './pages/EvalsPage'
import { JudgesPage } from './pages/JudgesPage'
import { Playground } from './pages/Playground'
import { RunPage } from './pages/RunPage'
import { RunsPage } from './pages/RunsPage'
import { ProjectProvider, useProject } from './project'
import { href, useRoute, type Route } from './router'
import { saveTheme, storedTheme, type Theme } from './theme'

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(storedTheme)
  const next: Theme = theme === 'light' ? 'dark' : 'light'
  return (
    <button className="btn ghost theme-toggle" title={`Switch to ${next} theme`} aria-label={`Switch to ${next} theme`}
      onClick={() => { saveTheme(next); setTheme(next) }}>
      {theme === 'light' ? '☾' : '☀'}
    </button>
  )
}

function TopBar({ route }: { route: Route }) {
  const { project, openNewRun } = useProject()
  const active = route.page === 'run' ? 'runs' : route.page
  const nav: Array<[Route['page'], string, string]> = [
    ['runs', 'Runs', href.runs()],
    ['evals', 'Evals', href.evals()],
    ['judges', 'Judges', href.judges()],
    ['playground', 'Playground', href.playground()],
  ]
  return (
    <header className="topbar">
      <a className="brand" href={STATIC ? undefined : href.runs()}><span className="logo">◆</span> fasteval</a>
      <span className="project">
        {STATIC
          ? `Static report · ${project?.name ?? ''} · generated ${new Date(STATIC.generated_at).toLocaleString()}`
          : project ? `${project.name} · ${project.config_source}` : ''}
      </span>
      {!STATIC ? (
        <nav>
          {nav.map(([page, label, link]) => <a key={page} href={link} className={active === page ? 'active' : ''}>{label}</a>)}
        </nav>
      ) : null}
      <span className="spacer" />
      <ThemeToggle />
      {!STATIC ? <button className="btn primary" onClick={() => openNewRun()}>▶ New run</button> : (
        <span className="muted" style={{ fontSize: 12.5 }}>Open with <span className="kbd">fasteval ui</span> to re-run or tune judges</span>
      )}
    </header>
  )
}

function Page({ route }: { route: Route }) {
  switch (route.page) {
    case 'run': return <RunPage runId={route.runId} nodeid={route.nodeid} />
    case 'evals': return <EvalsPage />
    case 'judges': return <JudgesPage />
    case 'playground': return <Playground />
    default: return <RunsPage />
  }
}

function Shell() {
  const route = useRoute()
  const { error, newRunPrefill, closeNewRun } = useProject()
  return (
    <>
      <TopBar route={route} />
      <main>
        {error ? <div className="banner warn" style={{ marginBottom: 12 }}>Couldn't load project: {error}</div> : null}
        <Page route={route} />
      </main>
      {newRunPrefill ? <NewRunDialog prefill={newRunPrefill} onClose={closeNewRun} /> : null}
    </>
  )
}

export function App() {
  return (
    <ToastProvider>
      <ProjectProvider>
        <Shell />
      </ProjectProvider>
    </ToastProvider>
  )
}
