"""Run the MarianMT translator from the command line."""

import argparse

from .translator import MarianTranslator


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English to French with MarianMT.")
    parser.add_argument("text", nargs="?", help="English text. Omit to enter text at a prompt.")
    args = parser.parse_args()

    try:
        text = args.text if args.text is not None else input("English: ")
    except EOFError:
        parser.error("Enter English text as an argument or at the prompt.")
    if not text.strip():
        parser.error("Enter non-empty English text.")

    translator = MarianTranslator()
    try:
        print(translator.translate(text))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
