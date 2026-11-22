"""Run real-model FLORES-200 or constructed challenge evaluation."""

import argparse
import csv
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
import transformers
from sacrebleu.metrics import BLEU, CHRF

from linguabridge.agent import Context, TranslationAgent
from linguabridge.qwen import QwenReviewer
from linguabridge.translator import MarianTranslator

SOURCE = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
SEED = 42
SPLIT = "devtest"
LANGUAGES = ("eng_Latn", "fra_Latn")
LICENSE = "CC BY-SA 4.0"
CHALLENGES = Path(__file__).with_name("challenge_cases.json")


def load_benchmark(archive: Path, limit: int) -> list[dict]:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("FLORES-200 archive SHA-256 does not match the pinned release.")
    with tarfile.open(archive) as dataset:
        columns = [
            dataset.extractfile(f"./flores200_dataset/{SPLIT}/{lang}.{SPLIT}")
            .read().decode("utf-8").splitlines()
            for lang in LANGUAGES
        ]
    if len(columns[0]) != len(columns[1]) or not 1 <= limit <= len(columns[0]):
        raise ValueError("Unexpected alignment or invalid sample size.")
    indices = random.Random(SEED).sample(range(len(columns[0])), limit)
    return [
        {"id": i + 1, "english": columns[0][i], "reference": columns[1][i]}
        for i in indices
    ]


def score(rows: list[dict]) -> dict:
    resolved = [row for row in rows if row["translation"]]
    eligible = [row for row in rows if row["reference"] is not None]

    def quality(subset):
        if not subset:
            return None
        hypotheses = [row["translation"] or "" for row in subset]
        references = [[row["reference"] for row in subset]]
        bleu, chrf = BLEU(), CHRF()
        return {
            "bleu": bleu.corpus_score(hypotheses, references).score,
            "chrf": chrf.corpus_score(hypotheses, references).score,
            "bleu_signature": str(bleu.get_signature()),
            "chrf_signature": str(chrf.get_signature()),
        }

    return {
        "inputs": len(rows),
        "resolved": len(resolved),
        "coverage": len(resolved) / len(rows),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "resolved_quality": quality([row for row in eligible if row["translation"]]),
        "all_input_quality": quality(eligible),
        "decisions": dict(Counter(row["decision"] for row in rows if row["decision"])),
    }


def direct_translate(qwen: QwenReviewer, english: str) -> str:
    messages = [
        {"role": "system", "content": "Translate English to French. Output only the French translation."},
        {"role": "user", "content": english},
    ]
    inputs = qwen.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    length = inputs.input_ids.shape[1]
    with torch.inference_mode():
        output = qwen.model.generate(
            **inputs, do_sample=False, temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=192
        )
    return qwen.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()


def evaluate(cases, translator, qwen, challenge):
    agent = TranslationAgent(translator, qwen)
    methods = ("agent",) if challenge else ("marianmt", "qwen_direct", "agent")
    for case in cases:
        for method in methods:
            row = dict(case, method=method, decision=None, draft=None, question=None, reason=None)
            start = perf_counter()
            if method == "marianmt":
                translation = translator.translate(case["english"])
            elif method == "qwen_direct":
                translation = direct_translate(qwen, case["english"])
            else:
                result = agent.translate(case["english"], Context(**case.get("context", {})))
                translation = result.final_translation
                row.update(decision=result.decision.value, draft=result.initial_translation,
                           question=result.question, reason=result.reason)
            elapsed = perf_counter() - start
            yield dict(row, translation=translation, latency_seconds=elapsed)
        print(f"Evaluated {case['id']}", flush=True)


def write_results(output, name, rows, metadata, summary):
    (output / f"{name}.json").write_text(
        json.dumps(dict(metadata=metadata, summary=summary, rows=rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("benchmark", "challenge"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    challenge = args.dataset == "challenge"
    if not challenge and args.archive is None:
        parser.error("--archive is required for the FLORES-200 benchmark.")

    cases = (json.loads(CHALLENGES.read_text(encoding="utf-8")) if challenge
             else load_benchmark(args.archive, args.limit))
    torch.set_num_threads(args.threads)
    translator, qwen = MarianTranslator(), QwenReviewer()
    # Warm up each actual inference path before measuring sentence latency.
    warmup = "The library opens at noon."
    translator.translate(warmup)
    if not challenge:
        direct_translate(qwen, warmup)
    TranslationAgent(translator, qwen).translate(warmup)
    rows = list(evaluate(cases, translator, qwen, challenge))
    metadata = {
        "dataset": "Constructed contextual challenge cases" if challenge else "FLORES-200",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": "cpu", "threads": args.threads,
        "models": {"marianmt": translator.model.config._commit_hash,
                   "qwen": qwen.model.config._commit_hash},
    }
    summary = {}
    if challenge:
        summary["decisions"] = dict(Counter(row["decision"] for row in rows))
        summary["expected_decision_matches"] = sum(
            row["decision"] == row["expected_decision"] for row in rows
        )
        summary["cases"] = len(rows)
        summary["coverage"] = sum(bool(row["translation"]) for row in rows) / len(rows)
    else:
        metadata.update(source=SOURCE, archive_sha256=SHA256, split=SPLIT,
                        languages=LANGUAGES, license=LICENSE, seed=SEED)
        for method in ("marianmt", "qwen_direct", "agent"):
            summary[method] = score([row for row in rows if row["method"] == method])
    args.output.mkdir(parents=True, exist_ok=True)
    write_results(args.output, args.dataset, rows, metadata, summary)


if __name__ == "__main__":
    main()
