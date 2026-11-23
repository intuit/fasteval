export const fmtScore = (v: unknown): string => (typeof v === 'number' ? v.toFixed(3) : '—')
export const fmtPct = (v: unknown): string => (typeof v === 'number' ? `${Math.round(v * 100)}%` : '—')

export function fmtMs(v: unknown): string {
  if (typeof v !== 'number') return '—'
  return v < 1000 ? `${Math.round(v)}ms` : `${(v / 1000).toFixed(1)}s`
}

export function fmtDuration(sec: unknown): string {
  if (typeof sec !== 'number' || sec < 0 || !isFinite(sec)) return '—'
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m}m ${String(s).padStart(2, '0')}s`
}

export function fmtAgo(iso?: string | null): string {
  if (!iso) return ''
  const secs = (Date.now() - new Date(iso).getTime()) / 1000
  if (secs < 45) return 'just now'
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`
  return `${Math.round(secs / 86400)}d ago`
}

export const asText = (v: unknown): string =>
  v === null || v === undefined ? '' : typeof v === 'string' ? v : JSON.stringify(v, null, 2)

export function messageText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (part && typeof part === 'object') {
          const p = part as { type?: string; text?: string; omitted?: string }
          if (p.text) return p.text
          if (p.omitted) return `[${p.type ?? 'part'}: ${p.omitted}]`
        }
        return asText(part)
      })
      .join('\n')
  }
  return asText(content)
}
