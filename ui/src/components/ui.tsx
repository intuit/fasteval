import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { fmtScore } from '../format'

export function Pill({ status, children }: { status: string; children?: ReactNode }) {
  return <span className={`pill ${status}`}>{children ?? status}</span>
}

export function Dot({ status }: { status: string }) {
  return <span className={`dot ${status}`} title={status} />
}

export function Score({ value, passed }: { value: number | null | undefined; passed?: boolean }) {
  const cls = passed === undefined ? '' : passed ? 'good' : 'bad'
  return <span className={`score ${cls}`}>{fmtScore(value)}</span>
}

export function Delta({ from, to }: { from?: number | null; to?: number | null }) {
  if (typeof from !== 'number' || typeof to !== 'number') return null
  const d = to - from
  if (Math.abs(d) < 0.0005) return <span className="delta">±0</span>
  return <span className={`delta ${d > 0 ? 'up' : 'down'}`}>{d > 0 ? `▲ +${d.toFixed(2)}` : `▼ ${d.toFixed(2)}`}</span>
}

/** Score bar with a tick at the pass threshold. */
export function Gauge({ score, threshold, passed }: { score: number; threshold: number; passed: boolean }) {
  return (
    <div className={`gauge${passed ? '' : ' bad'}`} title={`score ${score.toFixed(3)} · threshold ${threshold}`}>
      <div className="fill" style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }} />
      <div className="thr" style={{ left: `calc(${threshold * 100}% - 1px)` }} />
    </div>
  )
}

export function Bar({ value, bad }: { value: number | null; bad?: boolean }) {
  return (
    <div className={`bar${bad ? ' bad' : ''}`}>
      <span style={{ width: `${Math.round((value ?? 0) * 100)}%` }} />
    </div>
  )
}

export function Box({ title, kind, children, extra }: { title: string; kind?: string; children: ReactNode; extra?: ReactNode }) {
  return (
    <div className={`box ${kind ?? ''}`}>
      <div className="bh">{title}<span className="spacer" />{extra}</div>
      <div className="bb">{children}</div>
    </div>
  )
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  )
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="big">{title}</div>
      {children}
    </div>
  )
}

export function Spinner() {
  return <span className="spin" aria-label="loading" />
}

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const toast = useToast()
  return (
    <button
      className="btn small ghost"
      onClick={() => navigator.clipboard.writeText(text).then(() => toast('Copied'), () => toast('Copy failed', true))}
    >
      {label}
    </button>
  )
}

export function Modal({ title, onClose, children, actions }: { title: string; onClose: () => void; children: ReactNode; actions?: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="modal-back" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose() }}>
      <div className="modal panel" role="dialog" aria-label={title}>
        <h2>{title}</h2>
        {children}
        <div className="actions">
          <button className="btn" onClick={onClose}>Cancel</button>
          {actions}
        </div>
      </div>
    </div>
  )
}

// ── toasts ──────────────────────────────────────────────────────────────

type Toast = (message: string, isError?: boolean) => void
const ToastContext = createContext<Toast>(() => undefined)
export const useToast = () => useContext(ToastContext)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Array<{ id: number; message: string; isError: boolean }>>([])
  const show = useCallback<Toast>((message, isError = false) => {
    const id = Date.now() + Math.random()
    setItems((xs) => [...xs, { id, message, isError }])
    setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), isError ? 7000 : 3500)
  }, [])
  return (
    <ToastContext.Provider value={show}>
      {children}
      <div id="toast-root">
        {items.map((t) => (
          <div key={t.id} className={`toast${t.isError ? ' err' : ''}`}>{t.message}</div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}
