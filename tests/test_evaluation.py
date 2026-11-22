import hashlib
import io
import json
import tarfile

import pytest

from evaluation import run


def test_sample_keeps_aligned_references_and_stable_prefix(tmp_path, monkeypatch):
    archive = tmp_path / "flores.tar"
    with tarfile.open(archive, "w") as bundle:
        for language, payload in zip(run.LANGUAGES, (b"English 1\nEnglish 2\n", b"  FR 1  \nFR 2\n")):
            entry = tarfile.TarInfo(f"./flores200_dataset/devtest/{language}.devtest")
            entry.size = len(payload)
            bundle.addfile(entry, io.BytesIO(payload))
    monkeypatch.setattr(run, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    rows = run.load_benchmark(archive, 2)
    assert rows[0] == run.load_benchmark(archive, 1)[0]
    assert {row["english"]: row["reference"] for row in rows} == {
        "English 1": "  FR 1  ", "English 2": "FR 2",
    }
    with pytest.raises(ValueError, match="invalid sample size"):
        run.load_benchmark(archive, 3)


def test_archive_digest_mismatch(tmp_path):
    archive = tmp_path / "wrong.tar"
    archive.write_bytes(b"not the official archive")
    with pytest.raises(ValueError, match="SHA-256"):
        run.load_benchmark(archive, 1)


def test_unresolved_outputs_reduce_both_corpus_scores():
    reference = "Un chat dort sur le tapis."
    rows = [
        dict(reference=reference, translation=reference, latency_seconds=1, decision="ACCEPT_MT"),
        dict(reference=reference, translation=None, latency_seconds=3, decision="CLARIFY"),
    ]
    result = run.score(rows)
    assert result["coverage"] == 0.5
    assert result["average_latency_seconds"] == 2
    for metric in ("bleu", "chrf"):
        assert result["resolved_quality"][metric] == pytest.approx(100)
        assert result["all_input_quality"][metric] < 100


def test_zero_coverage_reports_no_resolved_quality():
    result = run.score([
        dict(reference="Une réponse est nécessaire.", translation=None, latency_seconds=1, decision="FLAG")
    ])
    assert result["resolved_quality"] is None
    assert result["all_input_quality"]["bleu"] == result["all_input_quality"]["chrf"] == 0


def test_constructed_set_has_all_decisions_and_context():
    cases = json.loads(run.CHALLENGES.read_text())
    assert 20 <= len(cases) <= 30
    assert len({case["id"] for case in cases}) == len(cases)
    assert {case["expected_decision"] for case in cases} == {"ACCEPT_MT", "REFINE", "CLARIFY", "FLAG"}
    for case in cases:
        assert case["english"] and case["expected_behavior"]
        assert isinstance(case["context"], dict)
        assert bool(case["reference"]) == (case["expected_decision"] in {"ACCEPT_MT", "REFINE"})


def test_english_output_counts_as_unresolved_for_quality():
    reference = "Il a écrit trois livres."
    result = run.score([
        dict(reference=reference, translation="He wrote three books.", valid_french=False,
             latency_seconds=1, decision="REFINE")
    ])
    assert result["coverage"] == 1
    assert result["valid_french_coverage"] == 0
    assert result["invalid_french_count"] == 1
    assert result["resolved_quality"] is None
    assert result["all_input_quality"]["bleu"] == 0


def test_unsupported_precision_or_recall_are_null():
    result = run.decision_metrics([{"decision": "FLAG", "expected_decision": "CLARIFY"}])
    assert result["accuracy"] == 0
    assert result["by_label"]["FLAG"]["precision"] == 0
    assert result["by_label"]["FLAG"]["recall"] is None
    assert result["by_label"]["CLARIFY"]["precision"] is None
    assert result["by_label"]["CLARIFY"]["recall"] == 0


def test_holdout_is_separate_from_development_cases():
    holdout = json.loads(run.HOLDOUT.read_text())
    development = json.loads(run.CHALLENGES.read_text())
    assert len(holdout) == 20
    assert {row["english"] for row in holdout}.isdisjoint(row["english"] for row in development)
