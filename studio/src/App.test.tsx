import '@testing-library/jest-dom/vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import type { TranslationResult } from './api'

const health = { status: 'ok', device: 'cpu', marianmt: 'idle', qwen: 'idle' }
const resolved: TranslationResult = {
  mode: 'agent', initial_translation: 'Tu peux ouvrir la fenêtre ?', decision: 'REFINE',
  status: 'resolved', validation: 'passed', explanation: 'The agent selected a revised French translation.',
  final_translation: 'Vous pouvez ouvrir la fenêtre ?', question: null, reason: null,
}
const clarification: TranslationResult = {
  ...resolved, initial_translation: 'Je suis allé à la banque.', decision: 'CLARIFY',
  status: 'clarification', validation: 'withheld', final_translation: null,
  explanation: 'More context is needed.', question: 'Riverbank or financial institution?',
}
const flagged: TranslationResult = {
  ...clarification, decision: 'FLAG', status: 'flagged', question: null, reason: 'Conflicting number.',
}

let request: ReturnType<typeof vi.fn>
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
})

beforeEach(() => {
  request = vi.fn()
  vi.stubGlobal('fetch', vi.fn((path: string, options: RequestInit) => (
    path === '/api/health' ? Promise.resolve(json(health)) : request(path, options)
  )))
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('translation studio', () => {
  it('disables empty submissions and shows loading, a validated result, and copy feedback', async () => {
    let complete!: (value: Response) => void
    request.mockReturnValue(new Promise<Response>(resolve => { complete = resolve }))
    const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) }
    vi.stubGlobal('navigator', { ...navigator, clipboard })
    render(<App />)
    expect(screen.getByRole('button', { name: 'AI Translate' })).toBeDisabled()
    fireEvent.change(screen.getByRole('textbox', { name: 'English' }), { target: { value: 'Can you open the window?' } })
    fireEvent.change(screen.getByLabelText('Formality'), { target: { value: 'formal' } })
    fireEvent.click(screen.getByRole('button', { name: 'AI Translate' }))
    expect(screen.getByText('Translating…')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy' })).toBeDisabled()
    expect(request.mock.calls[0][0]).toBe('/api/translate')
    expect(JSON.parse(request.mock.calls[0][1].body)).toEqual({
      text: 'Can you open the window?', mode: 'agent', context: { formality: 'formal' },
    })
    complete(json(resolved))
    expect(await screen.findByText('REFINE')).toBeInTheDocument()
    expect(screen.getByText(resolved.initial_translation)).toBeInTheDocument()
    expect(within(screen.getByRole('region', { name: 'French' })).getByText(resolved.final_translation!)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }))
    await screen.findByRole('button', { name: 'Copied' })
    expect(clipboard.writeText).toHaveBeenCalledWith(resolved.final_translation)
  })

  it('preserves submitted source and context when applying clarification', async () => {
    request.mockResolvedValueOnce(json(clarification)).mockResolvedValueOnce(json({
      ...resolved, final_translation: 'Je suis allé à la rive.',
    }))
    render(<App />)
    const source = screen.getByRole('textbox', { name: 'English' })
    fireEvent.change(source, { target: { value: 'I went to the bank.' } })
    fireEvent.change(screen.getByLabelText('Speaker gender'), { target: { value: 'masculine' } })
    fireEvent.change(screen.getByLabelText('Listener gender'), { target: { value: 'feminine' } })
    fireEvent.change(screen.getByLabelText('Intended meaning'), { target: { value: 'At dusk.' } })
    fireEvent.keyDown(source, { key: 'Enter', ctrlKey: true })
    expect(await screen.findByText(clarification.question!)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy' })).toBeDisabled()
    fireEvent.change(screen.getByRole('textbox', { name: 'Clarification answer' }), { target: { value: 'The bank of a river' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply clarification' }))
    await screen.findByText('REFINE')
    expect(JSON.parse(request.mock.calls[1][1].body)).toEqual({
      text: 'I went to the bank.', mode: 'agent', clarification: 'The bank of a river',
      context: { speaker_gender: 'masculine', listener_gender: 'feminine', meaning: 'At dusk.' },
    })
    expect(source).toHaveValue('I went to the bank.')
    expect(screen.queryByText('Clarification needed')).not.toBeInTheDocument()
  })

  it('withholds flagged content and removes stale insights when the source changes', async () => {
    request.mockResolvedValue(json(flagged))
    render(<App />)
    const source = screen.getByRole('textbox', { name: 'English' })
    fireEvent.change(source, { target: { value: 'Hello there.' } })
    fireEvent.click(screen.getByRole('button', { name: 'AI Translate' }))
    expect(await screen.findByText('Translation withheld')).toBeInTheDocument()
    expect(screen.getByText('Conflicting number.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Clear English input' }))
    expect(source).toHaveValue('')
    expect(screen.queryByText('Agent Insights')).not.toBeInTheDocument()
  })

  it('supports MT-only requests without agent context or insights', async () => {
    request.mockResolvedValue(json({
      ...resolved, mode: 'mt_only', decision: null, validation: 'not_run', final_translation: 'Bonjour.',
    }))
    render(<App />)
    fireEvent.change(screen.getByLabelText('Translation mode'), { target: { value: 'mt_only' } })
    fireEvent.change(screen.getByRole('textbox', { name: 'English' }), { target: { value: 'Hello.' } })
    fireEvent.click(screen.getByRole('button', { name: 'Translate' }))
    expect(await screen.findByText('Bonjour.')).toBeInTheDocument()
    expect(JSON.parse(request.mock.calls[0][1].body)).toEqual({ text: 'Hello.', mode: 'mt_only' })
    expect(screen.queryByLabelText('Formality')).not.toBeInTheDocument()
    expect(screen.queryByText('Agent Insights')).not.toBeInTheDocument()
  })

  it('shows a request failure and lets the user retry', async () => {
    request.mockResolvedValueOnce(json({ detail: 'Models could not load.' }, 503))
      .mockResolvedValueOnce(json(resolved))
    render(<App />)
    fireEvent.change(screen.getByRole('textbox', { name: 'English' }), { target: { value: 'Hello.' } })
    const button = screen.getByRole('button', { name: 'AI Translate' })
    fireEvent.click(button)
    expect(await screen.findByRole('alert')).toHaveTextContent('Models could not load.')
    expect(screen.getByText('Translation unavailable')).toBeInTheDocument()
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.click(button)
    expect(await screen.findByText('REFINE')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})
