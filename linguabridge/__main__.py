"""Run the English-to-French translator from the command line."""

import argparse

from .agent import Context, TranslationAgent
from .translator import MarianTranslator


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English to French.")
    parser.add_argument("text", nargs="?", help="English text. Omit to enter text at a prompt.")
    parser.add_argument("--formality", choices=("formal", "informal"))
    parser.add_argument("--speaker-gender", choices=("feminine", "masculine"))
    parser.add_argument("--listener-gender", choices=("feminine", "masculine"))
    parser.add_argument("--meaning", help="Intended meaning or situational context")
    parser.add_argument("--mt-only", action="store_true", help="Run the original MarianMT baseline")
    args = parser.parse_args()
    context = Context(args.formality, args.speaker_gender, args.listener_gender, args.meaning)
    if args.mt_only and context != Context():
        parser.error("Context options require agent mode.")

    try:
        text = args.text if args.text is not None else input("English: ")
    except EOFError:
        parser.error("Enter English text as an argument or at the prompt.")
    if not text.strip():
        parser.error("Enter non-empty English text.")

    translator = MarianTranslator()
    try:
        if args.mt_only:
            print(translator.translate(text))
            return

        from .qwen import QwenReviewer

        result = TranslationAgent(translator, QwenReviewer()).translate(text, context)
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))
    print(f"MarianMT: {result.initial_translation}")
    print(f"Decision: {result.decision.value}")
    if result.final_translation is not None:
        print(f"Final French: {result.final_translation}")
    if result.question is not None:
        print(f"Clarification: {result.question}")
    if result.reason is not None:
        print(f"Review needed: {result.reason}")


if __name__ == "__main__":
    main()
