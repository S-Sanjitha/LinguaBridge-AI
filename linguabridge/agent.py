"""Review a MarianMT draft and validate the agent's decision."""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol


from .contextual import agreement, context_conflict, lexical_context, register
from .validation import preservation_issue

class Decision(str, Enum):
    ACCEPT_MT = "ACCEPT_MT"
    REFINE = "REFINE"
    CLARIFY = "CLARIFY"
    FLAG = "FLAG"


Gender = Literal["feminine", "masculine"]
Formality = Literal["formal", "informal"]


@dataclass(frozen=True)
class Context:
    formality: Formality | None = None
    speaker_gender: Gender | None = None
    listener_gender: Gender | None = None
    meaning: str | None = None


@dataclass(frozen=True)
class Result:
    initial_translation: str
    decision: Decision
    final_translation: str | None = None
    question: str | None = None
    reason: str | None = None


class Translator(Protocol):
    def translate(self, text: str) -> str: ...


class Reviewer(Protocol):
    def review(self, source: str, draft: str, context: Context) -> str: ...


def parse_review(response: str) -> tuple[Decision, str]:
    def unique_fields(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("Duplicate fields")
        return fields

    if not isinstance(response, str):
        raise ValueError("Expected a JSON string")
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\n(.*?)\n```", response.strip(), re.S | re.I)
    review = json.loads(fenced[1] if fenced else response, object_pairs_hook=unique_fields)
    if not isinstance(review, dict) or review.keys() != {"decision", "text"}:
        raise ValueError("Expected a decision and text")
    decision = Decision(review["decision"])
    if not isinstance(review["text"], str):
        raise ValueError("Expected string text")
    text = review["text"].strip()
    if (decision == Decision.ACCEPT_MT) != (not text):
        raise ValueError("Expected empty text only for ACCEPT_MT")
    if decision == Decision.CLARIFY and not text.endswith("?"):
        raise ValueError("Expected a clarification question")
    return decision, text


def clarification_issue(question, resolved_meaning):
    if resolved_meaning:
        return "The supplied meaning already needs to be used before requesting clarification."
    if not re.search(r"\b(?:or|whether)\b", question, re.I):
        return "The clarification must specify the alternatives that affect translation."
    return None


class TranslationAgent:
    def __init__(self, translator: Translator, reviewer: Reviewer) -> None:
        self.translator = translator
        self.reviewer = reviewer

    def translate(self, source: str, context: Context | None = None) -> Result:
        source = source.strip()
        context = context or Context()
        draft = self.translator.translate(source)
        paraphrase, question = lexical_context(source, context.meaning)
        prepared = self.translator.translate(paraphrase) if paraphrase else draft
        prepared = register(prepared, source, context.formality)
        prepared = agreement(prepared, source, context)
        conflict = context_conflict(source, context.meaning)
        if conflict:
            return Result(draft, Decision.FLAG, reason=conflict)
        supported = prepared != draft and not preservation_issue(source, draft, prepared, context)

        try:
            if prepared != draft and getattr(type(self.reviewer), "review_prepared", None):
                response = self.reviewer.review_prepared(source, draft, prepared, context)
            else:
                response = self.reviewer.review(source, prepared, context)
            decision, text = parse_review(response)
            if supported:
                return Result(draft, Decision.REFINE, final_translation=prepared)
            result = self._result(source, draft, prepared, context, decision, text, question, paraphrase)
            if result.decision == Decision.FLAG and decision != Decision.FLAG:
                return self._retry(source, draft, prepared, context, result.reason, question, paraphrase)
            return result
        except (ValueError, TypeError, RecursionError):
            if supported:
                return Result(draft, Decision.REFINE, final_translation=prepared,
                              reason="Used a validated contextual correction after an invalid Qwen review.")
            return self._retry(source, draft, prepared, context, "Qwen returned an invalid structured review.", question, paraphrase)
        except (RuntimeError, OSError) as error:
            return Result(draft, Decision.FLAG, reason=f"Qwen review failed: {error}")

    def _result(self, source, draft, prepared, context, decision, text, lexical_question, paraphrase):
        if decision == Decision.FLAG:
            return Result(draft, decision, reason=text)
        if lexical_question:
            return Result(draft, Decision.CLARIFY, question=lexical_question)
        if decision == Decision.CLARIFY:
            issue = clarification_issue(text, paraphrase and context.meaning)
            if issue:
                return Result(draft, Decision.FLAG, reason=issue)
            return Result(draft, decision, question=text)
        candidate = prepared if decision == Decision.ACCEPT_MT else text
        issue = preservation_issue(source, draft, candidate, context)
        if issue:
            return Result(draft, Decision.FLAG, reason=issue)
        verify = getattr(type(self.reviewer), "verify", None)
        if candidate not in (draft, prepared) and verify and not self.reviewer.verify(
            source, candidate, context
        ):
            return Result(draft, Decision.FLAG, reason="Qwen could not confirm the refinement preserves meaning and context.")
        route = Decision.ACCEPT_MT if candidate == draft else Decision.REFINE
        return Result(draft, route, final_translation=candidate)

    def _retry(self, source, draft, prepared, context, issue, lexical_question, paraphrase):
        if lexical_question:
            return Result(draft, Decision.CLARIFY, question=lexical_question)
        if getattr(type(self.reviewer), "repair", None) is None:
            return Result(draft, Decision.FLAG, reason=issue)
        try:
            decision, text = parse_review(self.reviewer.repair(source, prepared, context, issue))
            return self._result(source, draft, prepared, context, decision, text, None, paraphrase)
        except (ValueError, TypeError, RecursionError, RuntimeError, OSError):
            return Result(draft, Decision.FLAG, reason=f"{issue} One correction attempt failed.")
