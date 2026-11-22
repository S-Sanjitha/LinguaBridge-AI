"""Bounded lexical and agreement cues for English-to-French translation."""

import re

# English paraphrases can be translated by MarianMT using the whole sentence.
# Each entry is a reusable phrase, with an inflected form wherever necessary.
IDIOMS = {
    "raining cats and dogs": "raining heavily",
    "rains cats and dogs": "rains heavily",
    "spilled the beans": "revealed the secret",
    "spill the beans": "reveal the secret",
    "a piece of cake": "very easy",
    "break the ice": "ease the initial tension",
    "under the weather": "unwell",
    "once in a blue moon": "very rarely",
}

# Each sense identifies an English paraphrase and context words supporting it.
SENSES = {
    "bank": (("riverbank", "river|water|shore|stream"), ("financial institution", "money|financial|loan|account")),
    "bat": (("baseball bat", "baseball|sport|wooden"), ("flying bat", "animal|mammal|cave|wings")),
    "crane": (("lifting machine", "machine|construction|lift|equipment"), ("crane bird", "bird|feather|wetland")),
    "seal": (("security seal", "security|package|envelope|tamper"), ("marine mammal", "animal|ocean|mammal|flipper")),
    "bark": (("tree bark", "tree|trunk|wood"), ("dog bark", "dog|puppy|noise")),
    "bass": (("bass fish", "fish|fishing|lake"), ("bass instrument", "instrument|music|guitar")),
}

ADJECTIVES = {
    "content": "contente", "enchanté": "enchantée", "épuisé": "épuisée",
    "fatigué": "fatiguée", "fier": "fière", "heureux": "heureuse",
    "inquiet": "inquiète", "motivé": "motivée", "prêt": "prête",
    "satisfait": "satisfaite", "soulagé": "soulagée", "surpris": "surprise",
}
MASCULINE = {value: key for key, value in ADJECTIVES.items()}
AGREEMENT = re.compile(
    r"(?P<prefix>\b(?:je suis|j'étais|tu es|tu étais|vous êtes|vous étiez|"
    r"es-tu|êtes-vous|suis-je|tu as l['’]air|vous avez l['’]air)"
    r"\s+(?:(?:très|si|vraiment|assez|un peu)\s+)?)"
    r"(?P<word>" + "|".join([*ADJECTIVES, *MASCULINE]) + r")\b", re.I,
)


def agreement(text, source, context):
    """Adjust a short singular copular clause with explicit referent gender."""
    if re.search(r"\b(?:you all|both of you|all of you|you two)\b", source, re.I):
        return text

    def adjust(match):
        prefix, word = match.group("prefix", "word")
        speaker = prefix.casefold().startswith(("je ", "j'", "j’", "suis-je"))
        pronoun = "I" if speaker else "you"
        gender = context.speaker_gender if speaker else context.listener_gender
        if not gender or not re.search(rf"\b{pronoun}\b", source, re.I):
            return match.group()
        forms = ADJECTIVES if gender == "feminine" else MASCULINE
        return prefix + forms.get(word.casefold(), word)

    return AGREEMENT.sub(adjust, text)


def lexical_context(source, meaning):
    """Return a paraphrase or a concrete ambiguity question, when supported."""
    motivation = re.sub(
        r"\bwhat makes (people|someone|them|him|her|us|you) tick\b",
        r"what motivates \1", source, count=1, flags=re.I,
    )
    if motivation != source:
        return motivation, None
    for idiom, paraphrase in IDIOMS.items():
        pattern = rf"\b{re.escape(idiom)}\b"
        match = re.search(pattern, source, re.I)
        if not match:
            continue
        # A literal dessert reading is possible and can be explicit in context.
        if idiom == "a piece of cake" and (
            re.search(r"\b(?:eat|ate|eaten|eating|slice|serve|served|dessert|bake|baked|bought|gave)\b",
                      f"{source} {meaning or ''}", re.I)
            or not re.search(r"\b(?:is|was|be)\s+a piece of cake\b", source, re.I)
        ):
            continue
        if idiom.startswith("break") and re.search(r"\b(?:frozen|lake|pond|hammer)\b", source, re.I):
            continue
        return re.sub(pattern, paraphrase, source, count=1, flags=re.I), None

    for noun, senses in SENSES.items():
        match = re.search(rf"\b(?:a|an|the)\s+({noun})\b(?!\s+of\b)", source, re.I)
        if not match:
            continue
        remaining = source[:match.start(1)] + source[match.end(1):]
        cues = f"{meaning or ''} {remaining}"
        selected = [
            label for label, keywords in senses if re.search(rf"\b(?:{keywords})\b", cues, re.I)
        ]
        if len(selected) == 1:
            start, end = match.span(1)
            return source[:start] + selected[0] + source[end:], None
        if not meaning and len(source.split()) <= 9:
            first, second = (sense[0] for sense in senses)
            return None, f"Does “{noun}” mean {first} or {second}?"
    return None, None


# Restrict register conversion to attested finite forms. Unsupported verbs go
# through Qwen rather than a guessed suffix substitution.
FORMAL_VERBS = {
    "as": "avez", "auras": "aurez", "aurais": "auriez",
    "dois": "devez", "es": "êtes", "étais": "étiez", "fais": "faites",
    "peux": "pouvez", "pourrais": "pourriez", "seras": "serez",
    "sais": "savez", "vas": "allez", "viens": "venez", "veux": "voulez",
    "voudrais": "voudriez",
}
INFORMAL_VERBS = {formal: informal for informal, formal in FORMAL_VERBS.items()}


def register(text, source, formality):
    """Convert a single explicit subject and known corresponding verb."""
    if not formality or not re.search(r"\byou\b", source, re.I) or re.search(
        r"\b(?:you all|both of you|all of you|you two)\b", source, re.I
    ):
        return text
    old, new = ("tu", "vous") if formality == "formal" else ("vous", "tu")
    verbs = FORMAL_VERBS if formality == "formal" else INFORMAL_VERBS
    pattern = r"|".join(verbs)
    prefix = re.fullmatch(rf"({old}) ((?:n['’]|ne )?)({pattern})\b(.*)", text, re.I)
    inversion = re.fullmatch(rf"({pattern})-{old}\b(.*)", text, re.I)
    if prefix:
        result = f"{new} {prefix[2]}{verbs[prefix[3].casefold()]}{prefix[4]}"
    elif inversion:
        result = f"{verbs[inversion[1].casefold()]}-{new}{inversion[2]}"
    else:
        return text
    if re.match(r"^Could you\b", source, re.I):
        result = re.sub(r"^pouvez-vous\b", "pourriez-vous", result)
        result = re.sub(r"^peux-tu\b", "pourrais-tu", result)
        result = re.sub(r"^vous pouvez\b", "vous pourriez", result)
        result = re.sub(r"^tu peux\b", "tu pourrais", result)
    return result[0].upper() + result[1:] if text[0].isupper() else result


def context_conflict(source, meaning):
    if not meaning:
        return None
    if re.search(r"\b(?:intended|actual)\s+number\b", meaning, re.I):
        intended = re.findall(r"\d+", meaning)
        if intended and set(intended) - set(re.findall(r"\d+", source)):
            return "The context conflicts with a number in the source."

    # Require an explicit restatement of the same event with opposite polarity.
    # This does not attempt general contradiction detection.
    def proposition(text):
        words = re.findall(r"[^\W\d_]+", text.casefold())
        return " ".join(
            re.sub(r"(?:ed|s)$", "", word).removesuffix("e")
            for word in words if word not in {"did", "does", "do", "never", "not", "the", "this", "that"}
        )

    negative = r"\b(?:never|not)\b"
    for clause in re.split(r"[.;:!?]", meaning):
        if bool(re.search(negative, source, re.I)) == bool(re.search(negative, clause, re.I)):
            continue
        original = proposition(source)
        restatement = proposition(clause)
        if len(original.split()) >= 3 and re.search(rf"\b{re.escape(original)}\b", restatement):
            return "The English source and supplied context assert opposite polarity."
    return None
