import json
from unittest.mock import Mock

import pytest

from linguabridge.agent import Context, Decision, TranslationAgent, parse_review
from linguabridge.qwen import review_messages


@pytest.fixture
def components():
    draft = "Marie n'a pas acheté 3 livres."
    translator = Mock()
    translator.translate.return_value = draft
    reviewer = Mock()
    return TranslationAgent(translator, reviewer), reviewer, draft


@pytest.mark.parametrize(("decision", "text", "field"), [
    ("ACCEPT_MT", "", "final_translation"),
    ("REFINE", "Marie n’a pas acquis 3 livres.", "final_translation"),
    ("CLARIFY", "Do you mean novels or textbooks?", "question"),
    ("FLAG", "Meaning uncertain.", "reason"),
])
def test_routing(components, decision, text, field):
    agent, reviewer, draft = components
    reviewer.review.return_value = json.dumps({"decision": decision, "text": text})
    context = Context(formality="formal", speaker_gender="feminine",
                      listener_gender="masculine", meaning="At a bookstore")
    source = "Marie did not buy 3 books."
    result = agent.translate(source, context)
    reviewer.review.assert_called_once_with(source, draft, context)
    assert result.initial_translation == draft
    assert result.decision == Decision(decision)
    assert getattr(result, field) == (text or draft)
    if decision in ("CLARIFY", "FLAG"):
        assert result.final_translation is None


@pytest.mark.parametrize("response", [
    "not JSON",
    '{"decision":"OTHER","text":""}',
    '{"decision":"ACCEPT_MT","decision":"REFINE","text":"Bonjour."}',
    '{"decision":"ACCEPT_MT","text":"Bonjour."}',
    '{"decision":"REFINE","text":" "}',
    '{"decision":"REFINE","text":5}',
    '{"decision":"CLARIFY","text":"Ambiguous."}',
    '{"decision":"FLAG","text":""}',
    '{"decision":"FLAG","text":"Uncertain","extra":true}',
])
def test_invalid_reviews_are_flagged(components, response):
    agent, reviewer, _ = components
    reviewer.review.return_value = response
    with pytest.raises(ValueError):
        parse_review(response)
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.FLAG
    assert result.final_translation is None


@pytest.mark.parametrize("candidate", [
    "Marie n'a pas acheté 4 livres.",
    "Marie a acheté 3 livres.",
    "Alice n'a pas acheté 3 livres.",
])
def test_refinement_preservation_guards(components, candidate):
    agent, reviewer, _ = components
    reviewer.review.return_value = json.dumps({"decision": "REFINE", "text": candidate})
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.FLAG
    assert result.final_translation is None


def test_accept_checks_draft(components):
    agent, reviewer, _ = components
    reviewer.review.return_value = '{"decision":"ACCEPT_MT","text":""}'
    result = agent.translate("Marie did not buy 4 books.")
    assert result.decision == Decision.FLAG


def test_review_error_is_visible(components):
    agent, reviewer, draft = components
    reviewer.review.side_effect = RuntimeError("generation failed")
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.FLAG
    assert result.initial_translation == draft
    assert result.final_translation is None
    assert "generation failed" in result.reason


def test_prompt_includes_only_supplied_context():
    payload = json.loads(review_messages(
        'Are you "ready"?', "Es-tu prêt ?", Context(listener_gender="feminine")
    )[-1]["content"])
    assert payload == {
        "english": 'Are you "ready"?',
        "marian": "Es-tu prêt ?",
        "context": {"listener_gender": "feminine"},
    }


def test_conflicting_formality_is_flagged(components):
    agent, reviewer, _ = components
    reviewer.review.return_value = json.dumps({
        "decision": "REFINE", "text": "Tu peux m’aider ?",
    })
    result = agent.translate("Can you help me?", Context(formality="formal"))
    assert result.decision == Decision.FLAG
    assert "formality" in result.reason


def test_identical_refinement_keeps_draft(components):
    agent, reviewer, draft = components
    reviewer.review.return_value = json.dumps({"decision": "REFINE", "text": draft})
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.ACCEPT_MT
    assert result.final_translation == draft


@pytest.mark.parametrize(("source", "draft", "context", "expected"), [
    ("Can you wait?", "Tu peux attendre ?", Context(formality="formal"), "Vous pouvez attendre ?"),
    ("Do you want to leave?", "Voulez-vous partir ?", Context(formality="informal"), "Veux-tu partir ?"),
    ("I am satisfied.", "Je suis satisfait.", Context(speaker_gender="feminine"), "Je suis satisfaite."),
    ("I am worried.", "Je suis inquiète.", Context(speaker_gender="masculine"), "Je suis inquiet."),
    ("Are you relieved?", "Es-tu soulagé ?", Context(listener_gender="feminine"), "Es-tu soulagée ?"),
])
def test_context_adjusts_french_agreement_and_finite_verbs(source, draft, context, expected):
    translator, reviewer = Mock(), Mock()
    translator.translate.return_value = draft
    reviewer.review.return_value = '{"decision":"ACCEPT_MT","text":""}'
    result = TranslationAgent(translator, reviewer).translate(source, context)
    assert result.initial_translation == draft
    assert result.final_translation == expected
    assert result.decision == Decision.REFINE


@pytest.mark.parametrize(("source", "context", "paraphrase", "candidate"), [
    ("We inspected the seal.", Context(meaning="The package security seal."),
     "We inspected the security seal.", "Nous avons inspecté le sceau de sécurité."),
    ("They need to break the ice.", Context(),
     "They need to ease the initial tension.", "Ils doivent détendre l’atmosphère."),
])
def test_lexical_context_uses_marian_on_paraphrases(source, context, paraphrase, candidate):
    translator, reviewer = Mock(), Mock()
    translator.translate.side_effect = ["La traduction originale.", candidate]
    reviewer.review.return_value = '{"decision":"ACCEPT_MT","text":""}'
    assert TranslationAgent(translator, reviewer).translate(source, context).final_translation == candidate
    assert translator.translate.call_args_list[-1].args == (paraphrase,)


def test_ambiguity_routes_to_specific_question(components):
    agent, reviewer, _ = components
    reviewer.review.return_value = '{"decision":"ACCEPT_MT","text":""}'
    result = agent.translate("They noticed a crane.")
    assert result.decision == Decision.CLARIFY
    assert "lifting machine" in result.question and "bird" in result.question


def test_context_conflict_with_numbers_is_flagged(components):
    agent, _, _ = components
    result = agent.translate("There are 5 seats.", Context(meaning="The actual number is 8."))
    assert result.decision == Decision.FLAG
    assert "number" in result.reason


def test_fenced_json_is_accepted_but_extra_prose_is_rejected():
    fenced = '```json\n{"decision":"ACCEPT_MT","text":""}\n```'
    assert parse_review(fenced) == (Decision.ACCEPT_MT, "")
    with pytest.raises(ValueError):
        parse_review("Here is the result: " + fenced)


class ReviewerWithCorrection:
    def __init__(self, first, second, verified=True):
        self.review = Mock(return_value=first)
        self.repair = Mock(return_value=second)
        self.verify = Mock(return_value=verified)

    def repair(self, source, draft, context, issue):
        raise AssertionError("The test must inject a response")

    def verify(self, source, candidate, context):
        raise AssertionError("The test must inject a verdict")


@pytest.mark.parametrize("first", [
    "invalid JSON",
    '{"decision":"REFINE","text":"Marie did not purchase 3 books yesterday."}',
    '{"decision":"REFINE","text":"Fecha de entrega."}',
])
def test_invalid_output_gets_exactly_one_correction(components, first):
    agent, _, _ = components
    repaired = "Marie n’a pas acquis 3 livres."
    reviewer = ReviewerWithCorrection(first, json.dumps({"decision": "REFINE", "text": repaired}))
    agent.reviewer = reviewer
    result = agent.translate("Marie did not buy 3 books.")
    assert result.final_translation == repaired
    assert reviewer.review.call_count == reviewer.repair.call_count == reviewer.verify.call_count == 1


def test_failed_correction_cannot_loop_or_leak_english(components):
    agent, _, _ = components
    reviewer = ReviewerWithCorrection("bad", '{"decision":"REFINE","text":"Marie did not buy 3 novels."}')
    agent.reviewer = reviewer
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.FLAG and result.final_translation is None
    reviewer.repair.assert_called_once()
    reviewer.verify.assert_not_called()


def test_semantic_rejection_cannot_release_candidate(components):
    agent, _, _ = components
    reviewer = ReviewerWithCorrection(
        '{"decision":"REFINE","text":"Marie n’a pas vendu 3 livres."}',
        '{"decision":"FLAG","text":"The proposed correction changed the action."}',
        verified=False,
    )
    agent.reviewer = reviewer
    result = agent.translate("Marie did not buy 3 books.")
    assert result.decision == Decision.FLAG and result.final_translation is None
    reviewer.verify.assert_called_once()
    reviewer.repair.assert_called_once()
