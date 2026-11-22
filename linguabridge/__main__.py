"""Run LinguaBridge from the command line."""

import argparse
import sys
from dataclasses import replace

from .agent import Context, Decision, Result, TranslationAgent
from .translator import MarianTranslator

EXIT_COMMANDS = {":q", ":quit", ":exit"}
EXPLANATIONS = {
    Decision.ACCEPT_MT: "The agent kept the MarianMT draft.",
    Decision.REFINE: "The agent selected a revised French translation.",
    Decision.CLARIFY: "More context is needed to choose a translation.",
    Decision.FLAG: "The agent withheld the translation for review.",
}
TRANSLATION_ERRORS = (ValueError, OSError, RuntimeError)


def create_agent() -> TranslationAgent:
    from .qwen import QwenReviewer

    return TranslationAgent(MarianTranslator(), QwenReviewer())


def show_result(result: Result) -> None:
    print(f"MarianMT: {result.initial_translation}")
    print(f"Decision: {result.decision.value}")
    print(f"Explanation: {EXPLANATIONS[result.decision]}")
    if result.decision in (Decision.ACCEPT_MT, Decision.REFINE) and result.final_translation:
        print(f"Final French: {result.final_translation}")
    else:
        print("Status: Unresolved")
    if result.question:
        print(f"Clarification: {result.question}")
    if result.reason:
        print(f"Details: {result.reason}")


def interactive(context: Context, mt_only: bool) -> None:
    model = None
    mode = "MarianMT only" if mt_only else "LinguaBridge agent"
    print(f"{mode}. Enter :quit to exit.")
    try:
        while True:
            text = input("English: ").strip()
            if text.casefold() in EXIT_COMMANDS:
                break
            if not text:
                print("Enter non-empty English text.")
                continue
            turn = context
            if not mt_only:
                hint = "Context (optional, Enter to keep the default): "
                meaning = input(hint).strip()
                if meaning.casefold() in EXIT_COMMANDS:
                    break
                if meaning:
                    turn = replace(turn, meaning=meaning)
            try:
                if mt_only:
                    if model is None:
                        model = MarianTranslator()
                    print(f"Final French: {model.translate(text)}")
                    continue
                if model is None:
                    model = create_agent()
                result = model.translate(text, turn)
                show_result(result)
                if result.decision == Decision.CLARIFY:
                    answer = input("Clarification answer (Enter to skip): ").strip()
                    if answer.casefold() in EXIT_COMMANDS:
                        break
                    if answer:
                        meaning = f"{turn.meaning}\n{answer}" if turn.meaning else answer
                        show_result(model.translate(text, replace(turn, meaning=meaning)))
            except TRANSLATION_ERRORS as error:
                print(f"Translation error: {error}", file=sys.stderr)
    except (EOFError, KeyboardInterrupt):
        print()
    print("Goodbye.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English to French.")
    parser.add_argument("text", nargs="?", help="English text. Omit to enter text at a prompt.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Translate multiple sentences in one session")
    parser.add_argument("--formality", choices=("formal", "informal"))
    parser.add_argument("--speaker-gender", choices=("feminine", "masculine"))
    parser.add_argument("--listener-gender", choices=("feminine", "masculine"))
    parser.add_argument("--meaning", help="Intended meaning or situational context")
    parser.add_argument("--mt-only", action="store_true", help="Run the original MarianMT baseline")
    args = parser.parse_args()
    context = Context(args.formality, args.speaker_gender, args.listener_gender, (args.meaning or "").strip() or None)
    if args.mt_only and context != Context():
        parser.error("Context options require agent mode.")
    if args.interactive:
        if args.text is not None:
            parser.error("Enter sentences at the prompt in interactive mode.")
        interactive(context, args.mt_only)
        return

    try:
        text = args.text if args.text is not None else input("English: ")
    except EOFError:
        parser.error("Enter English text as an argument or at the prompt.")
    if not text.strip():
        parser.error("Enter non-empty English text.")

    try:
        if args.mt_only:
            print(MarianTranslator().translate(text))
        else:
            show_result(create_agent().translate(text, context))
    except TRANSLATION_ERRORS as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
