export type Theme = 'light' | 'dark'

const KEY = 'fasteval.theme'

/** Light by default; dark only when the user picked it. */
export function storedTheme(): Theme {
  try {
    return localStorage.getItem(KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme
}

export function saveTheme(theme: Theme): void {
  applyTheme(theme)
  try {
    localStorage.setItem(KEY, theme)
  } catch {
    /* private mode: theme still applies for this page */
  }
}
