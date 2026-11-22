# LinguaBridge AI

Translate English text into French using the pretrained MarianMT model
[`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr).
This Phase 2 baseline runs on CPU.

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
python -m linguabridge
```

The second command prompts for English text. The first run downloads the model
from Hugging Face into its local cache. Set `HF_HOME` to a writable directory if
the default cache location is unavailable. Later runs reuse the cached model.
Inputs longer than the model's token limit are rejected.

## Run the tests

```bash
python -m pytest -q
```

The tests load the real pretrained weights and require network access on the
first run unless the model is already cached.
