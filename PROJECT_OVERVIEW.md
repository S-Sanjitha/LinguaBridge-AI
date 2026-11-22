# LinguaBridge AI project overview

## Problem and objective

English can leave distinctions implicit that French must express. “You” can
require `tu` or `vous`, adjectives can mark gender, and words such as “bat” can
require different French nouns. The objective is to use available context,
surface unresolved choices, and show which changes improve translation.

## Proposed solution

MarianMT produces a French draft. A Python agent prepares supported contextual
corrections and asks Qwen2.5-0.5B-Instruct to review the English, context, and
French. Validation checks the language, structure, and common preservation
constraints. A failed review can receive one bounded repair. The CLI shows
`ACCEPT_MT`, `REFINE`, `CLARIFY`, or `FLAG`, and displays final French only when
the agent resolves the input. The [README diagram](README.md#architecture)
shows the workflow.

## Technology

Python 3.12, PyTorch CPU, Hugging Face Transformers, MarianMT,
Qwen2.5-0.5B-Instruct, Lingua language identification, pytest, and SacreBLEU.

## Results

| Evaluation | Result |
|---|---|
| Constructed development | Agent met the contextual rubric on 19/20, matched 18/20 decisions, and returned French on 15/20. |
| Separate constructed holdout | Agent met the rubric on 17/20, versus 12/20 for MarianMT and 4/20 for direct Qwen. Agent decision matches were 15/20 and valid-French coverage was 16/20. |
| FLORES-200, 25 aligned pairs | MarianMT scored 51.60 BLEU and 69.69 chrF. The agent scored 51.05 BLEU and 69.42 chrF. |

The contextual results are from a small manual review. The general benchmark
scores decreased slightly relative to MarianMT. [Evaluation details](evaluation/PHASE_4_1.md)

## Limitations and future scope

The agent can miss agreement in compound forms, request unnecessary
clarification, or retain literal idioms. Language identification and Qwen's
semantic checks cannot guarantee correctness. Future work could use more
independently annotated context cases, improve agreement and sense rules, and
compare latency under controlled CPU conditions.
