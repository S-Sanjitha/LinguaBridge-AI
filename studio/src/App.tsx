import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  ArrowRight, ArrowUpRight, Check, CircleAlert, CircleHelp, Clipboard, Languages,
  LoaderCircle, Settings2, SlidersHorizontal, Sparkles, Trash2,
} from 'lucide-react'
import { getHealth, translate, type Context, type Health, type Mode, type TranslationRequest, type TranslationResult } from './api'

const TEXT_LIMIT = 3000
const CONTEXT_LIMIT = 1000
type Gender = NonNullable<Context['speaker_gender']> | ''

function GenderSelect({ id, label, value, onChange }: {
  id: string; label: string; value: Gender; onChange: (value: Gender) => void
}) {
  return <label className="field" htmlFor={id}>
    <span>{label}</span>
    <select id={id} value={value} onChange={event => onChange(event.target.value as Gender)}>
      <option value="">Unspecified</option>
      <option value="feminine">Feminine</option>
      <option value="masculine">Masculine</option>
    </select>
  </label>
}

export default function App() {
  const [text, setText] = useState('')
  const [context, setContext] = useState<Context>({})
  const [mode, setMode] = useState<Mode>('agent')
  const [result, setResult] = useState<TranslationResult | null>(null)
  const [answer, setAnswer] = useState('')
  const [health, setHealth] = useState<Health | null | undefined>(undefined)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [copyStatus, setCopyStatus] = useState('')
  const submitted = useRef<TranslationRequest | null>(null)
  const inFlight = useRef(false)

  useEffect(() => {
    const controller = new AbortController()
    async function checkHealth() {
      try {
        const value = await getHealth(controller.signal)
        if (!controller.signal.aborted) setHealth(value)
      } catch {
        if (!controller.signal.aborted) setHealth(null)
      }
    }
    void checkHealth()
    const timer = window.setInterval(checkHealth, busy ? 1000 : 8000)
    return () => { controller.abort(); window.clearInterval(timer) }
  }, [busy])

  function resetResult() {
    setResult(null)
    setAnswer('')
    setError('')
    setCopyStatus('')
    submitted.current = null
  }

  function updateContext(update: Partial<Context>) {
    setContext(current => ({ ...current, ...update }))
    resetResult()
  }

  async function run(request: TranslationRequest, followUp = false) {
    if (inFlight.current) return
    inFlight.current = true
    setBusy(true)
    setError('')
    setCopyStatus('')
    if (!followUp) {
      setResult(null)
      setAnswer('')
      submitted.current = null
    }
    try {
      const response = await translate(request)
      submitted.current = request
      setAnswer('')
      setResult(response)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Translation request failed.')
    } finally {
      inFlight.current = false
      setBusy(false)
    }
  }

  function submit(event?: FormEvent) {
    event?.preventDefault()
    if (!text.trim()) return
    const meaning = context.meaning?.trim()
    void run({
      text: text.trim(), mode,
      ...(mode === 'agent' ? { context: { ...context, meaning: meaning || undefined } } : {}),
    })
  }

  function clarify(event: FormEvent) {
    event.preventDefault()
    if (!submitted.current || !answer.trim()) return
    const clarification = [submitted.current.clarification, answer.trim()].filter(Boolean).join('\n')
    if (clarification.length > CONTEXT_LIMIT) {
      setError('The combined clarification is too long. Add a shorter summary to the context field and translate again.')
      return
    }
    void run({ ...submitted.current, clarification }, true)
  }

  const french = result?.status === 'resolved' ? result.final_translation : null
  const loadingModels = health?.marianmt === 'loading' || (mode === 'agent' && health?.qwen === 'loading')
  const modelError = health?.marianmt === 'error' || (mode === 'agent' && health?.qwen === 'error')
  const ready = health?.marianmt === 'ready' && (mode === 'mt_only' || health?.qwen === 'ready')
  const healthLabel = health === undefined ? 'Connecting' : health === null ? 'Backend offline'
    : loadingModels ? 'Loading models' : modelError ? 'Model load failed' : ready ? 'Models ready' : 'CPU backend online'

  async function copy() {
    if (!french) return
    try {
      await navigator.clipboard.writeText(french)
      setCopyStatus('Copied')
    } catch {
      setCopyStatus('Copy unavailable. Select the French text to copy it.')
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand"><span className="brand-icon"><Languages size={21} /></span> LinguaBridge <span>AI</span></div>
          <div className="direction" aria-label="English to French">English <ArrowRight size={14} /> French</div>
          <label className="mode-control">
            <Settings2 size={17} aria-hidden="true" />
            <span className="sr-only">Translation mode</span>
            <select aria-label="Translation mode" value={mode} disabled={busy} onChange={event => {
              setMode(event.target.value as Mode)
              resetResult()
            }}>
              <option value="agent">AI agent</option>
              <option value="mt_only">MarianMT only</option>
            </select>
          </label>
        </div>
      </header>

      <main className="studio mx-auto w-full max-w-[1240px]">
        <div className="title-row">
          <h1>Translation <span>Studio</span></h1>
          <span className="health" role="status">
            <span className={`status-dot ${health === null || modelError ? 'offline' : loadingModels ? 'working' : ''}`} />{healthLabel}
          </span>
        </div>

        <form onSubmit={submit}>
          <div className="workspace grid md:grid-cols-2">
            <section className="editor" aria-labelledby="english-label">
              <div className="panel-header">
                <label id="english-label" htmlFor="source">English</label>
                <button type="button" className="icon-button" aria-label="Clear English input" disabled={!text || busy}
                  onClick={() => { setText(''); resetResult() }}><Trash2 size={17} /></button>
              </div>
              <textarea id="source" value={text} maxLength={TEXT_LIMIT} disabled={busy}
                placeholder="Enter English text…"
                onChange={event => { setText(event.target.value); resetResult() }}
                onKeyDown={event => {
                  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                    event.preventDefault()
                    submit()
                  }
                }} />
              <footer><span>{text.length.toLocaleString()} / {TEXT_LIMIT.toLocaleString()}</span></footer>
            </section>
            <section className="editor output" aria-labelledby="french-label" aria-busy={busy}>
              <div className="panel-header">
                <h2 id="french-label">French</h2>
                <button className="copy-button" type="button" disabled={!french || busy} onClick={() => void copy()}>
                  {copyStatus === 'Copied' ? <Check size={16} /> : <Clipboard size={16} />} {copyStatus === 'Copied' ? 'Copied' : 'Copy'}
                </button>
              </div>
              <div className="output-body" aria-live="polite">
                {busy ? <div className="empty"><LoaderCircle className="spin" size={26} />
                  <p>{loadingModels ? 'Loading translation models…' : 'Translating…'}</p></div>
                  : french ? <p className="translation" lang="fr">{french}</p>
                  : <div className="empty">
                    {result ? <CircleAlert size={25} /> : <Languages size={27} />}
                    <p>{result?.status === 'clarification' ? 'Clarification required'
                      : result?.status === 'flagged' ? 'Translation withheld'
                      : error ? 'Translation unavailable' : 'Your French translation will appear here.'}</p>
                  </div>}
              </div>
              <footer aria-live="polite">{copyStatus !== 'Copied' ? copyStatus : ''}</footer>
            </section>
          </div>

          <div className="controls">
            {mode === 'agent' && <fieldset disabled={busy} className="context">
              <legend><SlidersHorizontal size={16} /> Context <span>Optional</span></legend>
              <div className="context-grid grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <label className="field">Formality
                  <select value={context.formality ?? ''} onChange={event =>
                    updateContext({ formality: (event.target.value || undefined) as Context['formality'] })}>
                    <option value="">Unspecified</option><option value="formal">Formal · vous</option>
                    <option value="informal">Informal · tu</option>
                  </select>
                </label>
                <GenderSelect id="speaker" label="Speaker gender" value={context.speaker_gender ?? ''}
                  onChange={value => updateContext({ speaker_gender: value || undefined })} />
                <GenderSelect id="listener" label="Listener gender" value={context.listener_gender ?? ''}
                  onChange={value => updateContext({ listener_gender: value || undefined })} />
                <label className="field">Intended meaning
                  <input value={context.meaning ?? ''} maxLength={CONTEXT_LIMIT} placeholder="e.g. the edge of a river"
                    onChange={event => updateContext({ meaning: event.target.value })} />
                </label>
              </div>
            </fieldset>}
            <div className="action-row">
              <span className="shortcut">Ctrl / ⌘ + Enter</span>
              <button className="primary" type="submit" disabled={!text.trim() || busy}>
                {busy ? <LoaderCircle size={18} className="spin" /> : <Sparkles size={18} />}
                {mode === 'agent' ? 'AI Translate' : 'Translate'}
              </button>
            </div>
          </div>
        </form>

        {error && <div className="notice error" role="alert"><CircleAlert size={19} />{error}</div>}
        {result?.status === 'clarification' && <section className="notice clarification" aria-labelledby="clarification-title">
          <CircleHelp size={21} />
          <div>
            <h2 id="clarification-title">Clarification needed</h2>
            <p>{result.question}</p>
            <form onSubmit={clarify} className="answer-row">
              <input aria-label="Clarification answer" maxLength={CONTEXT_LIMIT} placeholder="Add your clarification…"
                value={answer} disabled={busy} onChange={event => setAnswer(event.target.value)} />
              <button type="submit" className="primary" disabled={!answer.trim() || busy}>Apply clarification <ArrowUpRight size={16} /></button>
            </form>
          </div>
        </section>}

        {result?.decision && <section className="insights" aria-labelledby="insights-title">
          <div className="insight-title">
            <h2 id="insights-title"><Sparkles size={16} /> Agent Insights</h2>
            <span className={`decision ${result.status}`}>{result.decision}</span>
          </div>
          <p>{result.explanation}</p>
          {result.reason && <p className="detail">{result.reason}</p>}
          <div className="comparison grid gap-4 md:grid-cols-2">
            <div><h3>MarianMT draft</h3><p lang="fr">{result.initial_translation}</p></div>
            {french && <div><h3>Final French</h3><p lang="fr">{french}</p></div>}
          </div>
        </section>}
      </main>
    </>
  )
}
