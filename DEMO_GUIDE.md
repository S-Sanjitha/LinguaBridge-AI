# LinguaBridge AI demo guide

Allow 5–7 minutes. Install dependencies and cache both checkpoints before
presenting. Run one agent command in advance so the Python environment and
model downloads are ready. The model outputs below were observed on CPU
with the checkpoints recorded in the evaluation report. Loading and
translation times depend on the machine.

## 0:00–0:45 · Show the baseline

```bash
python -m linguabridge --mt-only "Hello, how are you?"
```

The observed output was “Bonjour, comment allez-vous ?”. Explain that
MarianMT alone translates the sentence without context or an agent decision.

## 0:45–2:15 · Show formality and agreement

```bash
python -m linguabridge -i --formality formal
python -m linguabridge -i --formality informal
python -m linguabridge "I am proud of my work." --speaker-gender feminine
python -m linguabridge "I am proud of my work." --speaker-gender masculine
```

In each interactive formality session, enter `Can you open the window?`,
press Enter at the context prompt, and enter `:quit` after the result before
starting the next command.

| Case | Observed decision | Final French |
|---|---|---|
| Formal | `REFINE` | Vous pouvez ouvrir la fenêtre ? |
| Informal | `ACCEPT_MT` | Tu peux ouvrir la fenêtre ? |
| Feminine speaker | `REFINE` | Je suis fière de mon travail. |
| Masculine speaker | `ACCEPT_MT` | Je suis fier de mon travail. |

The formal and feminine examples used validated contextual corrections
after invalid Qwen reviews. The CLI shows that detail.

## 2:15–4:30 · Show clarification, idioms, preservation, and review

Start one session and keep it open for all four inputs.

```bash
python -m linguabridge -i
```

| English input | What to enter at the prompt | Behavior to show |
|---|---|---|
| I went to the bank. | Leave context blank. At the clarification prompt, enter `The bank of a river`. | The first turn asks about the riverbank and financial senses. The follow-up returns “Je suis allé à la rive.” |
| It's raining cats and dogs. | Leave context blank. | `REFINE` replaces the literal draft with “Il pleut beaucoup.” |
| Marie did not buy 3 books. | Leave context blank. | `ACCEPT_MT` keeps “Marie n'a pas acheté 3 livres.” Point out the name, number, and negation. |
| The invoice lists 8 tickets. | Enter `The intended number of tickets is 80, but the English input must stay exactly as written.` | `FLAG` reports a conflicting number and leaves final French unset. |

Enter `:quit` to finish. To provide the sense up front, run
`python -m linguabridge "I went to the bank." --meaning "The bank of a river"`.

## 4:30–6:00 · Explain evidence and limits

Open the [architecture diagram](README.md#architecture) and the
[Phase 4.1 results](evaluation/PHASE_4_1.md). Explain when the agent keeps,
revises, asks, or withholds. Report the contextual holdout result of 17/20
against 12/20 for MarianMT and 4/20 for direct Qwen. Then show the FLORES-200
result of 51.05 BLEU for the agent versus 51.60 for MarianMT over 25 pairs.

If a run displays `Status: Unresolved`, keep that result visible. In the
tested CPU run, “I saw a bat.” raised a sensible clarification, but answering
“A baseball bat.” led to `FLAG` after an invalid review and failed repair.
A correct draft can be withheld. The holdout also revealed a missed feminine
agreement in a compound form and one literal idiom. If a lab machine cannot
download weights, point the CLI to a populated cache with `HF_HOME` and run
with `HF_HUB_OFFLINE=1`.
