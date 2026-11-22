"""English-to-French translation with pretrained MarianMT weights."""

import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_ID = "Helsinki-NLP/opus-mt-en-fr"


class MarianTranslator:
    """Load the tokenizer and model once for repeated CPU inference."""

    def __init__(self) -> None:
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_ID)
        self.model = MarianMTModel.from_pretrained(MODEL_ID).to("cpu").eval()

    def translate(self, text: str) -> str:
        text = text.strip()
        if not text:
            raise ValueError("Enter non-empty English text.")

        inputs = self.tokenizer(text, return_tensors="pt")
        limit = self.model.config.max_position_embeddings
        if inputs.input_ids.shape[1] > limit:
            raise ValueError(f"Input exceeds the model's {limit}-token limit.")

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=limit - 1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
