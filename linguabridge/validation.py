"""Checks that catch common French output and source preservation failures."""

import re
from collections import Counter
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

from .contextual import agreement

DIGITS = re.compile(r"\d+")
CAPITALIZED = re.compile(r"\b[^\W\d_][\w’'-]*\b")
EN_NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|cannot|nothing|nobody|nowhere)\b|n['’]t\b", re.I)
FR_NEGATION = re.compile(
    r"\b(?:pas|jamais|aucun(?:e|es|s)?|ni|sans|non)\b"
    r"|\b(?:ne\b|n['’])[^.!?]*\b(?:plus|rien|personne|guère)\b",
    re.I,
)
ADDRESS = {
    "formal": r"\b(?:tu|toi|te|tes|ton|ta)\b|\bt['’]",
    "informal": r"\b(?:vous|votre|vos)\b",
}


@lru_cache(maxsize=1)
def _identifier():
    return LanguageDetectorBuilder.from_languages(
        Language.FRENCH, Language.ENGLISH, Language.SPANISH, Language.ITALIAN,
        Language.PORTUGUESE, Language.GERMAN, Language.DUTCH, Language.ROMANIAN,
    ).build()


def is_french(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if text.casefold().strip(" .!?") in {
        "bonjour", "bonsoir", "merci", "oui", "non", "salut", "bien sûr", "au revoir",
    }:
        return True
    return _identifier().detect_language_of(text) == Language.FRENCH


def _likely_names(source: str, draft: str) -> set[str]:
    draft_words = {word.casefold() for word in CAPITALIZED.findall(draft)}
    return {
        word.casefold() for match in CAPITALIZED.finditer(source)
        if (word := match.group()) not in {"I"}
        and len(word) > 1 and word[0].isupper()
        and (word.casefold() in draft_words or word.isupper())
    }


def preservation_issue(source, draft, candidate, context):
    if not candidate.strip() or candidate.casefold() == source.casefold():
        return "The translation is empty or repeats the English source."
    if not is_french(candidate):
        return "The output was not identified as French."
    if re.search(r"\b(?:ne|je)\s+(?:ai|as|a|avons|avez|ont|est|étais|était)\b", candidate, re.I):
        return "The translation has an invalid elision before a French auxiliary."
    if context.formality and re.search(ADDRESS[context.formality], candidate, re.I):
        return "The French form of address conflicts with the requested formality."
    if agreement(candidate, source, context) != candidate:
        return "French agreement conflicts with the specified gender."
    if Counter(DIGITS.findall(source)) != Counter(DIGITS.findall(candidate)):
        return "The translation changed or reformatted a digit sequence."
    if EN_NEGATION.search(source) and not FR_NEGATION.search(candidate):
        return "The English negation could not be verified in French."
    if not EN_NEGATION.search(source) and not FR_NEGATION.search(draft) and FR_NEGATION.search(candidate):
        return "The proposed refinement introduced an unexpected negation marker."
    if candidate.rstrip().endswith("?") != source.rstrip().endswith("?"):
        return "The translation changed between a question and a statement."
    candidate_words = {word.casefold() for word in CAPITALIZED.findall(candidate)}
    if _likely_names(source, draft) - candidate_words:
        return "The translation dropped or changed a likely proper name."
    return None
