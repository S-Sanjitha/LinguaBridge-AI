# LinguaBridge AI evaluation

## FLORES-200 protocol

The benchmark uses the original Meta AI / No Language Left Behind FLORES-200
release, identified by the official archive's SHA-256
`b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`.
The public `devtest` split has 1,012 aligned pairs for English (`eng_Latn`)
and French (`fra_Latn`). The runner samples without replacement using Python
3.12 and seed 42. IDs are one-based line numbers in the original split. Source
sentences and French references are copied verbatim, excluding line
delimiters. Download time and model loading are excluded from timing.
[Archive](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz) ·
[Dataset documentation and citation](https://github.com/facebookresearch/flores/tree/main/flores200)

FLORES-200 excerpts in the result files are attributed to Meta AI and the NLLB
Team and distributed under CC BY-SA 4.0.
[Dataset card](https://huggingface.co/datasets/facebook/flores) ·
[License](https://creativecommons.org/licenses/by-sa/4.0/)

Each approach sees the same English inputs. MarianMT translates directly,
Qwen receives a translation-only prompt, and the agent runs MarianMT plus
Qwen review and validation. The French reference is used only by the scorer.
Inference is unbatched, each path is warmed up, and average latency is the
mean wall-clock time across all inputs, including unresolved agent responses.
Agent latency includes generating its own MarianMT draft. The benchmark used
two PyTorch CPU threads. The contextual run overlapped on the same 2-CPU
workspace, so latency is an observation under shared load. Run the two
commands consecutively on an idle machine for more controlled latency.

SacreBLEU 2.5.1 computes corpus BLEU with 13a tokenization and corpus chrF
using up to six character n-gram orders. The JSON includes metric signatures.
Coverage is the fraction of inputs with a nonempty final output. Resolved
quality scores only those rows. All-input quality keeps every original
reference and inserts an empty hypothesis for any missing output, so
unresolved `CLARIFY` and `FLAG` cases contribute no matching n-grams. The
all-input columns compare the same inputs across approaches. Resolved-only
columns can compare different subsets.

The script defaults to 100 pairs. The recorded run uses 25 due to CPU
generation time. The small subset and single reference per input limit
generalization. Resolved coverage measures whether a response was returned.
It does not judge whether that response is correct or entirely in French.

## FLORES-200 results

| Approach | Resolved BLEU | Resolved chrF | All-input BLEU | All-input chrF | Coverage | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| MarianMT only | 51.60 | 69.69 | 51.60 | 69.69 | 25/25 (100%) | 1.59 s |
| Qwen direct | 21.12 | 48.21 | 21.12 | 48.21 | 25/25 (100%) | 6.89 s |
| LinguaBridge Agent | 51.60 | 69.69 | 51.60 | 69.69 | 25/25 (100%) | 7.05 s |

The agent chose `ACCEPT_MT` on every benchmark input. Its outputs and scores
were identical to MarianMT, and it added latency in this run. All-input and
resolved scores coincide because every approach returned an output. The
constructed challenge cases below demonstrate unresolved outcomes.

- **Row 655** · The agent kept MarianMT's literal “les gens cochent” for
  “what makes people tick,” missing the sense of motivation.
- **Row 26** · Direct Qwen translated an announcement as “L'annulation”,
  changing the event to a cancellation.
- **Row 760** · Direct Qwen replaced Montevideo's location south of the
  Equator with “à l'ouest du cercle polar,” changing the geography.

The [benchmark JSON](results/benchmark.json) and [benchmark CSV](results/benchmark.csv)
record all 75 model outputs. The challenge results are available separately in
[JSON](results/challenge.json) and [CSV](results/challenge.csv).

## Reproduce

Install the dependencies as described in the [project README](../README.md).
The scripts use the pretrained MarianMT and Qwen checkpoints, downloading
them on the first run. Model revisions and package versions are in the
result JSON metadata.

```bash
curl -fL https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz -o /tmp/flores200_dataset.tar.gz
python -m evaluation.run benchmark --archive /tmp/flores200_dataset.tar.gz --limit 25 --output /tmp/linguabridge-results
python -m evaluation.run challenge --threads 1 --output /tmp/linguabridge-results
python -m pytest -q
```

Omit `--limit 25` to run 100 benchmark pairs.

## Constructed contextual challenge results

These 20 hand-authored cases probe formality, agreement, ambiguity, idioms,
preservation, and agent decisions. Their French references are examples of
acceptable translations. `ACCEPT_MT` versus `REFINE` expectations describe
the preferred route for the scenario and can differ even when both outputs
are acceptable. A matching route alone does not establish correctness.

The agent produced a final output for **11 of 20 cases (55%)** and matched
**8 of 20 expected decision labels**. It returned 8 `ACCEPT_MT`, 3
`REFINE`, 5 `CLARIFY`, and 4 `FLAG`. Cases intentionally designed for
clarification or review contribute to the denominator. Their expected lack
of a translation is not a translation-quality failure.

| Case | Expected behavior | Observed behavior |
|---|---|---|
| C01, C04, C08 | Use the requested form of address | Flagged conflicts instead of producing a correction. |
| C02 | Informal French request | Returned English, “Could you please help me?”, under `REFINE`. |
| C03 | Formal French reservation question | Returned acceptable “Avez-vous une réservation ?” even though the formal draft was also acceptable. |
| C05, C07 | Feminine agreement | Retained masculine “fier” and “épuisé”, respectively. C07 also changed a statement to a question. |
| C06 | Masculine agreement | Kept “Je suis fier de mon travail.” |
| C09, C10 | Clarify an ambiguous “bank” and resolve it as riverbank when given context | Accepted “à la banque” without context, then asked what “bank” meant after the riverbank context was supplied. |
| C12 | Ask whether “crane” means bird or machine | Asked what “crane” meant without naming the two alternatives. |
| C13–C15 | Translate idioms by meaning | Asked about “cats and dogs” and accepted literal translations of “spilled the beans” and “piece of cake”. |
| C16–C18 | Preserve names, number-to-item associations, and negation | Kept the three MarianMT drafts, which preserved these details. |
| C19, C20 | Flag conflicting context | Asked what “8” meant in C19. Flagged C20 because the output failed the negation check. |

The cases deliberately include things the agent can get wrong. They are too
few to estimate general contextual accuracy. No benchmark score combines
them with FLORES-200. See the [case definitions](challenge_cases.json) and
[per-case outputs](results/challenge.json).
