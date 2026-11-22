"""Compare preserved Phase 4 outputs with the Phase 4.1 CPU runs."""

import argparse
import json
from pathlib import Path

from linguabridge.validation import is_french

from .run import METHODS, decision_metrics, score

ROOT = Path(__file__).parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path, *, contextual=False, audit=None):
    data = read_json(path)
    rows = [
        dict(row, valid_french=is_french(row["translation"] or ""))
        for row in data["rows"]
    ]
    result = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        if not subset:
            continue
        summary = score(subset)
        if contextual and method == "agent":
            summary["decision_metrics"] = decision_metrics(subset)
        if audit is not None:
            verdicts = audit[method]
            if set(verdicts) != {row["id"] for row in subset}:
                raise ValueError(f"Incomplete contextual review for {path.name} / {method}")
            correct = sum(verdicts.values())
            summary["contextual_correctness"] = {
                "correct": correct, "total": len(subset), "rate": correct / len(subset)
            }
        result[method] = summary
    return result


def verify_same_inputs(baseline, current, *, contextual=False):
    fields = ["english", "reference"]
    if contextual:
        fields += ["context", "expected_decision", "expected_behavior"]

    def cases(path):
        return {
            row["id"]: tuple(row[key] for key in fields)
            for row in read_json(path)["rows"] if row["method"] == "agent"
        }

    if cases(baseline) != cases(current):
        raise ValueError(f"Phase 4 and 4.1 input sets differ for {current.name}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "results")
    parser.add_argument("--current", type=Path, default=ROOT / "results" / "phase4_1")
    parser.add_argument("--audit", type=Path, default=ROOT / "contextual_review.json")
    args = parser.parse_args()
    audit = read_json(args.audit)
    verify_same_inputs(args.baseline / "benchmark.json", args.current / "benchmark.json")
    verify_same_inputs(
        args.baseline / "challenge.json", args.current / "challenge.json", contextual=True,
    )
    comparison = {
        "phase4": {
            "benchmark": summarize(args.baseline / "benchmark.json"),
            "challenge": summarize(
                args.baseline / "challenge.json", contextual=True, audit=audit["phase4_challenge"],
            ),
        },
        "phase4_1": {
            "benchmark": summarize(args.current / "benchmark.json"),
            "challenge": summarize(
                args.current / "challenge.json", contextual=True, audit=audit["phase4_1_challenge"],
            ),
            "holdout": summarize(
                args.current / "holdout.json", contextual=True, audit=audit["phase4_1_holdout"],
            ),
        },
    }
    before = audit["phase4_challenge"]["agent"]
    after = audit["phase4_1_challenge"]["agent"]
    comparison["development_agent_changes"] = {
        "improved": [case for case in before if not before[case] and after[case]],
        "worsened": [case for case in before if before[case] and not after[case]],
    }
    (args.current / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
