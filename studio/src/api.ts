export type Mode = 'agent' | 'mt_only'
export type Decision = 'ACCEPT_MT' | 'REFINE' | 'CLARIFY' | 'FLAG'
export type Context = {
  formality?: 'formal' | 'informal'
  speaker_gender?: 'feminine' | 'masculine'
  listener_gender?: 'feminine' | 'masculine'
  meaning?: string
}
export type TranslationRequest = {
  text: string
  mode: Mode
  context?: Context
  clarification?: string
}
export type TranslationResult = {
  mode: Mode
  initial_translation: string
  decision: Decision | null
  status: 'resolved' | 'clarification' | 'flagged'
  validation: 'passed' | 'withheld' | 'not_run'
  explanation: string
  final_translation: string | null
  question: string | null
  reason: string | null
}
type ModelStatus = 'idle' | 'loading' | 'ready' | 'error'
export type Health = { status: 'ok'; device: 'cpu'; marianmt: ModelStatus; qwen: ModelStatus }

export async function getHealth(signal: AbortSignal): Promise<Health> {
  const response = await fetch('/api/health', { signal })
  if (!response.ok) throw new Error('Backend unavailable')
  return response.json()
}

export async function translate(request: TranslationRequest): Promise<TranslationResult> {
  let response: Response
  try {
    response = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  } catch {
    throw new Error('Could not reach the backend. Check that FastAPI is running.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' && body.detail
      ? body.detail : 'Translation request failed. Check the backend and try again.')
  }
  return response.json()
}
