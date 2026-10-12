"""Extraction of the chosen option letter from a model response."""

import re

THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"<think>.*", flags=re.DOTALL | re.IGNORECASE)


def strip_reasoning(text):
    """Remove reasoning blocks so that only the final response is parsed."""
    text = THINK_BLOCK.sub("", text)
    return UNCLOSED_THINK.sub("", text).strip()


def extract_answer(text, valid_letters):
    """Return the chosen letter, or None if no unambiguous answer is found.

    Rules, applied in order:
      1. the last explicit statement such as "Answer: B" or "the answer is (B)"
      2. a letter at the very start of the response, e.g. "B", "B.", "(B)"
      3. a single distinct standalone letter anywhere in the response
    """
    letters = "".join(sorted(valid_letters))
    response = strip_reasoning(text)
    if not response:
        return None

    explicit = re.findall(
        rf"answer\s*(?:is|:)?\s*(?:option\s*)?\(?\**([{letters}])\b", response, flags=re.IGNORECASE
    )
    if explicit:
        return explicit[-1].upper()

    leading = re.match(rf"^\s*[\(\*]*([{letters}])(?:[\)\.\:\*,]|\s|$)", response)
    # A leading "A" followed by a lowercase word is the English article, not an option.
    if leading and not re.match(r"^\s*A\s+[a-z]", response):
        return leading.group(1)

    without_articles = re.sub(r"(?<![A-Za-z])A(?=\s+[a-z])", "", response)
    standalone = set(re.findall(rf"(?<![A-Za-z])([{letters}])(?![A-Za-z])", without_articles))
    if len(standalone) == 1:
        return standalone.pop()
    return None
