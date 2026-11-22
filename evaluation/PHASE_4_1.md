# Phase 4.1 agent evaluation

[Phase 4](REPORT.md) and its original CSV and JSON exports remain the baseline.
Phase 4.1 reruns the same 25 aligned FLORES-200 `devtest` pairs and the same
20 constructed development cases. Its separate 20-case holdout was created
for evaluation only. An initial holdout process was stopped during a
development-driven revision before its predictions were inspected. The
committed holdout results come from the frozen final agent.
The models receive English, allowed context, and where applicable a French
draft. References and expected routes are used after inference.

## Root causes and changes

- The original few-shot prompt made retaining the draft an easy default.
  All 25 FLORES examples ended in `ACCEPT_MT`, including a literal
  mistranslation of “what makes people tick.” This establishes an observed
  acceptance bias. It does not establish the model's internal reasoning.
  The revised prompt distinguishes faithful drafts, fixable errors, unresolved
  material ambiguity, and conflicts. It provides a candidate when a narrow
  contextual rule yields one. Validated candidates from these narrow rules
  take precedence over free-form rewrites and can survive a malformed review.
- The prior French guard checked whether the candidate was empty or identical
  to the source. An English paraphrase passed. The agent now uses a statistical
  language detector and checks names, digit counts, common negation markers,
  question versus statement, and supported agreement and register constraints.
- The earlier checks caught some `tu`/`vous` mismatches but could not correct
  their verb forms. The new rules cover a limited list of finite verb pairs and
  adjective pairs when the relevant person and register are explicit.
- Qwen sometimes ignored supplied senses or accepted literal idioms. A small
  lexical inventory supplies sense-specific English paraphrases and concrete
  ambiguity questions. Recognized idioms can be paraphrased in English and
  translated by the same MarianMT instance. The code has no case-ID rules or
  stored French answers for the evaluation sentences.
- The parser accepts strict JSON or one complete JSON fence, while rejecting
  duplicate keys, extra fields, and inconsistent payloads. Qwen reviews
  free-form French revisions for semantic fidelity. An invalid review or
  candidate gets at most one correction attempt, using the same Qwen instance.
  Unverified corrections leave the draft visible and the final translation unset.

## Measurement

FLORES uses the same official archive, pair IDs, language codes and references
as Phase 4. The three methods are MarianMT only, direct Qwen and the
LinguaBridge agent. On contextual inputs, direct Qwen receives the available
context and MarianMT translates the sentence alone. These baselines produce
translations and have no routing labels.

Output coverage counts nonempty final answers. Valid-French coverage counts
the subset classified as French. `invalid` counts nonempty outputs failing
that classifier. Corpus BLEU and chrF use SacreBLEU 2.5.1. Resolved scores use
valid French outputs that have references. All-input scores use all rows
with references and replace abstentions or invalid-language outputs with
empty hypotheses. Challenge rows that expect only clarification or flagging
have no French reference and contribute to decision and context denominators.

Contextual correctness is a manual review of meaning, relevant context and
preservation. Either `ACCEPT_MT` or `REFINE` can be correct if the resulting
French is suitable. A case that asks for clarification requires a question
specific enough to distinguish the competing meanings. A case that asks
for review requires withholding the output for a relevant reason. Translation
baselines receive no credit when the input calls for clarification or a flag.
This is one review, with no independent annotator. Decision-label accuracy
uses the exact expected labels. Precision or recall is undefined if its
denominator is zero.

Average CPU latency includes generation, checks and any correction within
the agent. It excludes model loading and the evaluation-only language check.
Phase 4.1 used one PyTorch thread per run and overlapped runs on a two-CPU
workspace. The original Phase 4 benchmark used two threads under shared load.
Differences in latency across phases are descriptive. The final benchmark and
development tables reuse the real MarianMT and direct-Qwen predictions from
the first Phase 4.1 pass. Those two methods and their inputs were unchanged.
The final agent was rerun for every row, and all three holdout methods were
run with the final evaluator.

## Reproduce

Install dependencies using the [project README](../README.md). From the
repository root, run the commands below with the pinned FLORES archive.

```bash
curl -fL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz -o /tmp/flores200_dataset.tar.gz
python -m evaluation.run benchmark --archive /tmp/flores200_dataset.tar.gz --limit 25 --threads 1
python -m evaluation.run challenge --threads 1
python -m evaluation.run holdout --threads 1
python -m evaluation.compare
python -m pytest -q
```

## FLORES-200 before and after

| Run | Approach | Resolved BLEU / chrF | All-input BLEU / chrF | Output coverage | Valid French | Invalid French | Mean CPU |
|---|---|---:|---:|---:|---:|---:|---:|
| Phase 4 | MarianMT | 51.60 / 69.69 | 51.60 / 69.69 | 25/25 | 25/25 | 0 | 1.59 s |
| Phase 4 | Direct Qwen | 21.12 / 48.21 | 21.12 / 48.21 | 25/25 | 25/25 | 0 | 6.89 s |
| Phase 4 | LinguaBridge | 51.60 / 69.69 | 51.60 / 69.69 | 25/25 | 25/25 | 0 | 7.05 s |
| Phase 4.1 | MarianMT | 51.60 / 69.69 | 51.60 / 69.69 | 25/25 | 25/25 | 0 | 1.24 s |
| Phase 4.1 | Direct Qwen | 21.12 / 48.21 | 21.12 / 48.21 | 25/25 | 25/25 | 0 | 5.84 s |
| Phase 4.1 | LinguaBridge | 51.05 / 69.42 | 51.05 / 69.42 | 25/25 | 25/25 | 0 | 7.90 s |

## Contextual development and holdout

| Run | Approach | Resolved BLEU / chrF | All-input BLEU / chrF | Output coverage | Valid French | Invalid French | Mean CPU | Contextually correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Phase 4 development | LinguaBridge | 73.30 / 84.24 | 39.82 / 55.60 | 11/20 | 10/20 | 1 | 11.02 s | 6/20 |
| Phase 4.1 development | MarianMT | 55.85 / 70.06 | 55.85 / 70.06 | 20/20 | 20/20 | 0 | 0.34 s | 6/20 |
| Phase 4.1 development | Direct Qwen | 28.42 / 49.15 | 28.42 / 49.15 | 20/20 | 20/20 | 0 | 2.26 s | 3/20 |
| Phase 4.1 development | LinguaBridge | 68.03 / 76.76 | 65.85 / 72.57 | 15/20 | 15/20 | 0 | 7.05 s | 19/20 |
| Phase 4.1 holdout | MarianMT | 42.59 / 67.96 | 42.59 / 67.96 | 20/20 | 20/20 | 0 | 0.37 s | 12/20 |
| Phase 4.1 holdout | Direct Qwen | 26.33 / 46.29 | 23.19 / 40.94 | 20/20 | 18/20 | 2 | 2.10 s | 4/20 |
| Phase 4.1 holdout | LinguaBridge | 46.94 / 70.88 | 46.18 / 67.98 | 16/20 | 16/20 | 0 | 6.47 s | 17/20 |

### Agent decisions

| Run | ACCEPT_MT | REFINE | CLARIFY | FLAG | Exact label matches |
|---|---:|---:|---:|---:|---:|
| Phase 4 FLORES | 25 | 0 | 0 | 0 | — |
| Phase 4.1 FLORES | 24 | 1 | 0 | 0 | — |
| Phase 4 development | 8 | 3 | 5 | 4 | 8/20 |
| Phase 4.1 development | 4 | 11 | 2 | 3 | 18/20 |
| Phase 4.1 holdout | 11 | 5 | 2 | 2 | 15/20 |

Each cell reports precision / recall, with expected-label support in parentheses.

| Run | ACCEPT_MT | REFINE | CLARIFY | FLAG |
|---|---:|---:|---:|---:|
| Phase 4 development | 50% / 67% (6) | 67% / 20% (10) | 20% / 50% (2) | 25% / 50% (2) |
| Phase 4.1 development | 100% / 67% (6) | 91% / 100% (10) | 100% / 100% (2) | 67% / 100% (2) |
| Phase 4.1 holdout | 64% / 88% (8) | 100% / 56% (9) | 50% / 100% (1) | 100% / 100% (2) |

## Notable changes and remaining failures

- **FLORES row 655** · The agent changed “les gens cochent” to “ce qui
  motive les gens,” improving the idiom. “Les appelle” still mistranslates
  “appeals to them.” This was the only benchmark refinement. Overall BLEU
  and chrF were slightly lower than MarianMT's on the same 25 inputs.
- **Development C02, C04, C05, C07** · The agent replaced an English
  refinement and three incorrect or missing French outputs with “Pourrais-tu
  m'aider ?”, “Es-tu prête à partir ?”, “Je suis fière de mon travail.”,
  and “Vous êtes épuisée.”
- **Development C09, C10, C12** · The agent now distinguishes the two
  meanings of “bank” and “crane” when asking a question. With river context
  it returns “Je suis allé à la rive.” The sense is correct, though
  “au bord de la rivière” is more idiomatic.
- **Development C13–C15** · The revised outputs convey heavy rain,
  revealing a secret, and an easy exam. **C19 and C20** withheld translations
  for number and polarity conflicts.
- **Development regression C03** · A correct formal reservation question
  was withheld after an invalid structured review and failed repair.
  C11 changed to `REFINE` where `ACCEPT_MT` was expected. The revised
  French still preserves the specified security-seal sense.
- **Holdout successes** · H12 and H13 fixed literal idioms. H09 asked a
  specific question about the fish and musical meanings of “bass.” H19
  and H20 flagged contradictory time and polarity. H04 retained
  “Je m'inquiète du retard,” which is gender-neutral and consistent with a
  feminine speaker.
- **Holdout failures** · H06 corrected the address to `tu` but left
  “surpris” masculine after the compound copula “as été.” H08 asked about
  tree bark even though “heard” strongly implied a sound. H14 kept a literal
  translation of “costs an arm and a leg.” The two direct-Qwen English
  responses, H08 and H15, count as invalid outputs and contribute no score.

The reviewed development agent improved on 14 cases and worsened on one
(C03). On holdout it met the contextual rubric on 17 of 20, with exact
decision labels on 15 of 20. Some correct outputs differ from their single
reference, and precise decision labels can depend on whether the initial
MarianMT draft already fits. The small counts per route make the
precision and recall estimates unstable.

Language detection is statistical and can miss mixed-language text or
misclassify short phrases. Lexical rules, finite verb pairs and agreement
patterns cover only common forms. The Qwen verification step uses the same
0.5B checkpoint as the reviewer and can miss factual drift. The safeguards
do not establish semantic equivalence. These samples show better contextual
handling on the reviewed cases and no general benchmark-quality gain.

[Comparison JSON](results/phase4_1/comparison.json) ·
[Manual case judgments](contextual_review.json) ·
[FLORES outputs](results/phase4_1/benchmark.json) ·
[Development outputs](results/phase4_1/challenge.json) ·
[Holdout outputs](results/phase4_1/holdout.json)
