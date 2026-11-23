import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import './styles.css'
import { applyTheme, storedTheme } from './theme'

applyTheme(storedTheme()) // before first paint: no flash of the wrong theme

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
