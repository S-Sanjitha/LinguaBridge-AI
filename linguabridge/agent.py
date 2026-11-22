"""Review a MarianMT draft and validate the agent's decision."""

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol


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

    review = json.loads(response, object_pairs_hook=unique_fields)
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


DIGITS = re.compile(r"\d+")
CAPITALIZED = re.compile(r"\b[A-Z][a-z]+\b")
EN_NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|cannot|nothing|nobody|nowhere)\b|n['’]t\b", re.I)
FR_NEGATION = re.compile(
    r"\b(?:pas|jamais|aucun(?:e|es|s)?|ni|sans|non)\b"
    r"|\b(?:ne\b|n['’])[^.!?]*\b(?:plus|rien|personne|guère)\b",
    re.I,
)


def preservation_issue(
    source: str, draft: str, candidate: str, context: Context,
) -> str | None:
    if not candidate.strip() or candidate.casefold() == source.casefold():
        return "The translation is empty or repeats the English source."
    address = {
        "formal": r"\b(?:tu|toi|te|tes|ton|ta)\b",
        "informal": r"\b(?:vous|votre|vos)\b",
    }
    if context.formality and re.search(address[context.formality], candidate, re.I):
        return "The French form of address conflicts with the requested formality."
    if Counter(DIGITS.findall(source)) != Counter(DIGITS.findall(candidate)):
        return "The translation changed or reformatted a digit sequence."
    if EN_NEGATION.search(source) and not FR_NEGATION.search(candidate):
        return "The English negation could not be verified in French."
    # A capitalized word shared by source and draft is a likely name. Checking
    # the intersection avoids treating every English sentence opener as a name.
    likely_names = set(CAPITALIZED.findall(source)) & set(CAPITALIZED.findall(draft))
    if likely_names - set(CAPITALIZED.findall(candidate)):
        return "The refinement dropped or changed a likely proper name."
    return None


class TranslationAgent:
    def __init__(self, translator: Translator, reviewer: Reviewer) -> None:
        self.translator = translator
        self.reviewer = reviewer

    def translate(self, source: str, context: Context | None = None) -> Result:
        source = source.strip()
        draft = self.translator.translate(source)
        context = context or Context()
        try:
            response = self.reviewer.review(source, draft, context)
        except (RuntimeError, OSError, ValueError) as error:
            return Result(draft, Decision.FLAG, reason=f"Qwen review failed: {error}")
        try:
            decision, text = parse_review(response)
        except (ValueError, TypeError, RecursionError):
            return Result(draft, Decision.FLAG, reason="Qwen returned an invalid structured review.")
        if decision == Decision.CLARIFY:
            return Result(draft, decision, question=text)
        if decision == Decision.FLAG:
            return Result(draft, decision, reason=text)
        candidate = draft if decision == Decision.ACCEPT_MT else text
        issue = preservation_issue(source, draft, candidate, context)
        if issue:
            return Result(draft, Decision.FLAG, reason=issue)
        if candidate == draft:
            decision = Decision.ACCEPT_MT
        return Result(draft, decision, final_translation=candidate)
