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
