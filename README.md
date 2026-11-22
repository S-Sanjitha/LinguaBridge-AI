# LinguaBridge AI

Translate English text into French using the pretrained MarianMT model
[`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr).
MarianMT produces a draft. A single pretrained
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
agent reviews it against the English input and optional context. Both models run on CPU.

## CPU setup

From the repository root, use Python 3.12. On Linux, install CPU-only PyTorch
before installing the remaining dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6,<3'
python -m pip install -r requirements.txt
```

## Translate

```bash
python -m linguabridge "Hello, how are you?"
python -m linguabridge "Can you help me?" --formality formal
python -m linguabridge "Are you ready?" --formality informal --listener-gender feminine
python -m linguabridge "I saw a bat." --meaning "A baseball bat"
python -m linguabridge
python -m linguabridge --mt-only "Hello, how are you?"
```

Running without a sentence prompts for English text. The first run downloads the requested models
from Hugging Face into its local cache. Set `HF_HOME` to a writable directory if
the default cache location is unavailable. Later runs reuse the cached weights.
Inputs longer than the model's token limit are rejected.

## Agent decisions

- `ACCEPT_MT` returns the MarianMT draft.
- `REFINE` returns the French translation proposed by Qwen.
- `CLARIFY` displays a question. Rerun the CLI with the answer in the matching
  context option, such as `--meaning`.
- `FLAG` displays a review reason and leaves the final translation unset.

Use `--speaker-gender` or `--listener-gender` when French agreement depends on
that person. Both options accept `feminine` or `masculine`.

The agent validates the response structure and checks for changes to digit
sequences, common negation markers, and likely proper names shared by the
source and draft. These checks are heuristic. They can flag valid formatting
changes and miss semantic errors. Small-model decisions can also be wrong,
including unnecessary clarification and mistakes in gender agreement.

## Run the tests

```bash
python -m pytest -q
```

The MarianMT tests load real weights and require network access on the first
run unless the model is cached. Agent unit tests use mocked translation and
review responses. The CLI examples run real Qwen inference in agent mode.

## Evaluation

See [the evaluation report](evaluation/REPORT.md) for the FLORES-200 protocol,
CPU results, and a separate contextual challenge review.

[Phase 4.1](evaluation/PHASE_4_1.md) compares the revised agent with the
preserved Phase 4 results and reports a separate contextual holdout. The
agent checks language and basic factual and grammatical constraints,
asks Qwen to verify free-form revisions, and allows at most one repair.
