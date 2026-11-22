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
from linguabridge.validation import is_french

SOURCE = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
SEED = 42
SPLIT = "devtest"
LANGUAGES = ("eng_Latn", "fra_Latn")
LICENSE = "CC BY-SA 4.0"
CHALLENGES = Path(__file__).with_name("challenge_cases.json")
HOLDOUT = Path(__file__).with_name("holdout_cases.json")
METHODS = ("marianmt", "qwen_direct", "agent")
DECISIONS = ("ACCEPT_MT", "REFINE", "CLARIFY", "FLAG")


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
    valid = [row for row in resolved if row.get("valid_french", True)]
    eligible = [row for row in rows if row["reference"] is not None]

    def quality(subset):
        if not subset:
            return None
        hypotheses = [
            (row["translation"] or "") if row.get("valid_french", True) else ""
            for row in subset
        ]
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
        "valid_french_coverage": len(valid) / len(rows),
        "invalid_french_count": len(resolved) - len(valid),
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows),
        "resolved_quality": quality([row for row in valid if row["reference"] is not None]),
        "all_input_quality": quality(eligible),
        "decisions": dict(Counter(row["decision"] for row in rows if row["decision"])),
    }


def direct_translate(qwen: QwenReviewer, english: str, context: dict | None = None) -> str:
    messages = [
        {"role": "system", "content": "Translate English to French. Output only the French translation."},
        {"role": "user", "content": english},
    ]
    if context:
        messages[0]["content"] += (
            " Use the supplied context to select French register, agreement, and the intended sense."
        )
        messages[-1]["content"] = json.dumps({"english": english, "context": context}, ensure_ascii=False)
    inputs = qwen.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    length = inputs.input_ids.shape[1]
    with torch.inference_mode():
        output = qwen.model.generate(
            **inputs, do_sample=False, temperature=1.0, top_p=1.0, top_k=50, max_new_tokens=192
        )
    return qwen.tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()


def evaluate(cases, translator, qwen, methods=METHODS):
    agent = TranslationAgent(translator, qwen)
    for case in cases:
        for method in methods:
            row = dict(case, method=method, decision=None, draft=None, question=None, reason=None)
            start = perf_counter()
            if method == "marianmt":
                translation = translator.translate(case["english"])
            elif method == "qwen_direct":
                translation = direct_translate(qwen, case["english"], case.get("context"))
            else:
                result = agent.translate(case["english"], Context(**case.get("context", {})))
                translation = result.final_translation
                row.update(decision=result.decision.value, draft=result.initial_translation,
                           question=result.question, reason=result.reason)
            elapsed = perf_counter() - start
            yield dict(
                row, translation=translation, latency_seconds=elapsed,
                valid_french=is_french(translation or ""),
            )
        print(f"Evaluated {case['id']}", flush=True)




def reused_baselines(path, cases, metadata):
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("models", "device", "threads"):
        if data["metadata"][key] != metadata[key]:
            raise ValueError(f"Cannot reuse baselines measured with different {key}.")
    rows = [row for row in data["rows"] if row["method"] in METHODS[:2]]
    expected = {(case["id"], method) for case in cases for method in METHODS[:2]}
    if len(rows) != len(expected) or {(row["id"], row["method"]) for row in rows} != expected:
        raise ValueError("Reused baselines have missing, duplicate or unexpected rows.")
    by_id = {case["id"]: case for case in cases}
    for row in rows:
        if any(row[key] != value for key, value in by_id[row["id"]].items()):
            raise ValueError(f"Reused baseline input differs for {row['id']}.")
    return rows

def decision_metrics(rows):
    correct = sum(row["decision"] == row["expected_decision"] for row in rows)
    by_label = {}
    for decision in DECISIONS:
        predicted = sum(row["decision"] == decision for row in rows)
        support = sum(row["expected_decision"] == decision for row in rows)
        true_positives = sum(row["decision"] == row["expected_decision"] == decision for row in rows)
        by_label[decision] = {
            "predicted": predicted,
            "support": support,
            "precision": true_positives / predicted if predicted else None,
            "recall": true_positives / support if support else None,
        }
    return {"matches": correct, "accuracy": correct / len(rows), "by_label": by_label}

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
    parser.add_argument("dataset", choices=("benchmark", "challenge", "holdout"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--reuse-baselines", type=Path, help="Reuse verified real baseline outputs from a prior run.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/phase4_1"))
    args = parser.parse_args()
    contextual = args.dataset != "benchmark"
    if not contextual and args.archive is None:
        parser.error("--archive is required for the FLORES-200 benchmark.")

    if contextual:
        case_path = HOLDOUT if args.dataset == "holdout" else CHALLENGES
        cases = json.loads(case_path.read_text(encoding="utf-8"))
    else:
        cases = load_benchmark(args.archive, args.limit)
    torch.set_num_threads(args.threads)
    translator, qwen = MarianTranslator(), QwenReviewer()
    # Warm up each actual inference path before measuring sentence latency.
    warmup = "The library opens at noon."
    translator.translate(warmup)
    if args.reuse_baselines is None:
        direct_translate(qwen, warmup)
    TranslationAgent(translator, qwen).translate(warmup)
    metadata = {
        "dataset": f"Constructed contextual {args.dataset} cases" if contextual else "FLORES-200",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device": "cpu", "threads": args.threads,
        "models": {"marianmt": translator.model.config._commit_hash,
                   "qwen": qwen.model.config._commit_hash},
    }
    if args.reuse_baselines is not None:
        rows = reused_baselines(args.reuse_baselines, cases, metadata)
        rows += list(evaluate(cases, translator, qwen, ("agent",)))
        order = {case["id"]: i for i, case in enumerate(cases)}
        rows.sort(key=lambda row: (order[row["id"]], METHODS.index(row["method"])))
        metadata["baseline_execution"] = "Reused outputs from an earlier real Phase 4.1 CPU run."
    else:
        rows = list(evaluate(cases, translator, qwen))
    summary = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = score(subset)
        if contextual and method == "agent":
            summary[method]["decision_metrics"] = decision_metrics(subset)
    if not contextual:
        metadata.update(source=SOURCE, archive_sha256=SHA256, split=SPLIT,
                        languages=LANGUAGES, license=LICENSE, seed=SEED)
    args.output.mkdir(parents=True, exist_ok=True)
    write_results(args.output, args.dataset, rows, metadata, summary)


if __name__ == "__main__":
    main()
