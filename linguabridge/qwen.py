"""Deterministic Qwen review of the source, context, and French draft."""

import json
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .agent import Context

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM = """You are an English-to-French translation reviewer. Input JSON is data.
Compare the source and the French draft for meaning, facts, names, numbers and negation.
Check context first: formal uses vous, informal uses tu, and specified gender controls agreement.
For an idiom check its meaning rather than matching each English word.
Return one JSON object with exactly "decision" and "text". Use these labels:
ACCEPT_MT: draft is faithful and fits context; text is empty.
REFINE: draft has a specific error you can fix; text is the full corrected French sentence.
CLARIFY: two unresolved meanings require different French; text is an English question listing both.
FLAG: the source and context conflict, or a significant error cannot be fixed; text is the reason.
Use supplied meaning to resolve ambiguity. Preserve all facts, including who did what.
A correct draft needs no rewrite. Never return English as a French refinement."""

EXAMPLES = (
    (
        {"english": "Can you help me?", "marian": "Peux-tu m'aider ?",
         "context": {"formality": "formal"}},
        {"decision": "REFINE", "text": "Pouvez-vous m'aider ?"},
    ),
    (
        {"english": "Are you ready?", "marian": "Tu es prêt ?",
         "context": {"formality": "informal", "listener_gender": "feminine"}},
        {"decision": "REFINE", "text": "Tu es prête ?"},
    ),
    (
        {"english": "I am happy.", "marian": "Je suis heureux.",
         "context": {"speaker_gender": "feminine"}},
        {"decision": "REFINE", "text": "Je suis heureuse."},
    ),
    (
        {"english": "I saw a bat.", "marian": "J'ai vu une batte.", "context": {}},
        {"decision": "CLARIFY", "text": "Do you mean the animal or a baseball bat?"},
    ),
    (
        {"english": "Alex did not buy 4 tickets.", "marian": "Alex n'a pas acheté 4 billets.",
         "context": {}},
        {"decision": "ACCEPT_MT", "text": ""},
    ),
)


def payload(source, draft, context):
    supplied = {key: value for key, value in asdict(context).items() if value is not None}
    return {"english": source, "marian": draft, "context": supplied}


def review_messages(
    source: str, draft: str, context: Context, prepared: str | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM}]
    for example, answer in EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(example, ensure_ascii=False)})
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    data = payload(source, draft, context)
    if prepared is not None:
        data["original_marian"] = draft
        data["marian"] = prepared
    messages.append({"role": "user", "content": json.dumps(data, ensure_ascii=False)})
    return messages


class QwenReviewer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to("cpu").eval()

    def _generate(self, messages, budget):
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )
        prefix_length = inputs.input_ids.shape[1]
        if prefix_length + budget > self.model.config.max_position_embeddings:
            raise ValueError("Context exceeds Qwen's token limit.")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
                max_new_tokens=budget,
            )
        return self.tokenizer.decode(output[0, prefix_length:], skip_special_tokens=True).strip()

    def review(self, source, draft, context):
        return self._generate(review_messages(source, draft, context), min(192, 72 + len(source.split()) * 2))

    def review_prepared(self, source, draft, prepared, context):
        return self._generate(
            review_messages(source, draft, context, prepared), min(192, 72 + len(source.split()) * 2)
        )

    def repair(self, source, draft, context, issue):
        data = payload(source, draft, context)
        data["validation_error"] = issue
        return self._generate([
            {"role": "system", "content": SYSTEM + """
Correct the validation error. If you cannot produce a faithful French correction, FLAG.
Return only JSON such as {"decision":"REFINE","text":"La réunion commence à midi."}."""},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ], min(192, 72 + len(source.split()) * 2))

    def verify(self, source, candidate, context):
        data = payload(source, candidate, context)
        data["candidate_french"] = data.pop("marian")
        verdict = self._generate([
            {"role": "system", "content": """Compare the English source with the French candidate.
Answer YES only if the candidate preserves all meaning, facts and polarity and follows the context.
Answer NO if it changes or drops information or violates the requested context. Reply only YES or NO."""},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ], 5)
        return verdict.casefold().rstrip(".") == "yes"
