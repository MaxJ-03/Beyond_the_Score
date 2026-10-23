"""Question text, scores and seen/unseen labels of the contamination tests (no model needed)."""

import math
import re
import zlib

import numpy as np

try:  # fast edit distances; the pure Python versions below give the same results
    from rapidfuzz.distance import LCSseq, Levenshtein
except ImportError:
    LCSseq = Levenshtein = None

OPTION_LABELS = ("numbers", "letters", "none")
SEEN_LABELS = ("seen", "unseen")
STRICT_LABELS = ("seen", "unseen", "ambiguous")


# ---------------------------------------------------------------- 1. question text

def question_number(item_id):
    """Booklet question number from an item id such as 'mir2022-026', or None."""
    match = re.search(r"(\d+)$", item_id)
    return int(match.group(1)) if match else None


def question_text(item, text_cfg, with_options=None):
    """The question as raw text: the stem, then the options numbered as in the MIR booklet."""
    stem = item.question
    if text_cfg.get("include_question_number"):
        number = question_number(item.item_id)
        if number is not None:
            stem = f"{number}. {stem}"
    if with_options is None:
        with_options = text_cfg.get("include_options", True)
    if not with_options:
        return stem
    style = text_cfg.get("option_labels", "numbers")
    if style not in OPTION_LABELS:
        raise ValueError(f"option_labels must be one of {OPTION_LABELS}, got '{style}'.")
    lines = [stem]
    for position, (letter, option) in enumerate(item.options.items(), start=1):
        label = {"numbers": f"{position}. ", "letters": f"{letter}. ", "none": ""}[style]
        lines.append(label + option)
    return "\n".join(lines)


def split_at_words(text, ratio):
    """Split a text after int(ratio * n) of its n words; returns (prefix, reference, n_words, n_prefix_words).
    prefix + reference == text; the reference starts with the whitespace after the prefix."""
    starts = [match.start() for match in re.finditer(r"\S+", text)]
    if len(starts) < 2:
        raise ValueError(f"Cannot split a text of fewer than two words: {text!r}")
    cut = min(max(int(len(starts) * ratio + 1e-9), 1), len(starts) - 1)
    prefix = text[:starts[cut]].rstrip()
    return prefix, text[len(prefix):], len(starts), cut


# ---------------------------------------------------------------- 2. Min-K% Prob

def k_label(k, prefix=""):
    """Column name of a Min-K% score, e.g. k=0.2 -> 'mink_20'."""
    return f"{prefix}mink_{k * 100:g}"


def text_logprobs(token_ids, logprobs, special_ids=()):
    """Log-probabilities of the text tokens, without the first position (BOS) and special tokens."""
    if len(token_ids) != len(logprobs):
        raise ValueError(f"{len(token_ids)} tokens but {len(logprobs)} log-probabilities.")
    return [float(value) for token, value in zip(token_ids[1:], logprobs[1:])
            if token not in special_ids and value is not None]


def min_k(logprobs, k):
    """Min-K% Prob (Shi et al., 2023): mean log-probability of the k% least likely tokens."""
    values = np.sort(np.asarray(logprobs, dtype=float))
    if values.size == 0:
        return float("nan")
    count = max(1, int(values.size * k + 1e-9))
    return float(values[:count].mean())


def likelihood_scores(logprobs, text, k_values, prefix=""):
    """Min-K% for every k, the mean log-probability and perplexity of all tokens, and the zlib
    ratio of LLMSanitize (log-perplexity divided by the zlib-compressed length in bytes)."""
    values = np.asarray(logprobs, dtype=float)
    mean = float(values.mean()) if values.size else float("nan")
    scores = {k_label(k, prefix): min_k(values, k) for k in k_values}
    scores[f"{prefix}mean_logprob"] = mean
    scores[f"{prefix}perplexity"] = math.exp(-mean) if values.size else float("nan")
    scores[f"{prefix}zlib_ratio"] = -mean / len(zlib.compress(text.encode("utf-8")))
    return scores


# ---------------------------------------------------------------- 3. completion

def words(text):
    """Lowercased Unicode words (letters with accents and ñ stay intact)."""
    return re.findall(r"\w+", text.lower())


def _lcs_length(a, b):
    if LCSseq is not None:
        return LCSseq.similarity(a, b)
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def _levenshtein(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def rouge_l_f1(reference, candidate):
    """ROUGE-L F1 on words: F1 of the longest common subsequence of words."""
    ref, cand = words(reference), words(candidate)
    if not ref or not cand:
        return 0.0
    lcs = _lcs_length(ref, cand)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def levenshtein_ratio(reference, candidate):
    """Normalized Levenshtein similarity of two strings: 1 - distance / length of the longer."""
    if not reference and not candidate:
        return 1.0
    if Levenshtein is not None:
        return float(Levenshtein.normalized_similarity(reference, candidate))
    return 1.0 - _levenshtein(reference, candidate) / max(len(reference), len(candidate))


def common_prefix_length(a, b):
    """Number of leading elements two sequences share."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def completion_scores(reference, generation, reference_ids, generated_ids, threshold):
    """Similarity of a continuation to the real second half of the question."""
    lcp = common_prefix_length(reference_ids, generated_ids)
    lev = levenshtein_ratio(reference.strip(), generation.strip())
    return {
        "rouge_l_f1": rouge_l_f1(reference, generation),
        "levenshtein_ratio": lev,
        "lcp_tokens": lcp,
        "lcp_fraction": lcp / len(reference_ids) if reference_ids else float("nan"),
        "near_verbatim": bool(lev >= threshold),
    }


# ---------------------------------------------------------------- 4. seen or unseen

def _end_of_month(date):
    """'2024-08' -> '2024-08-31' (the end of the month), full dates unchanged."""
    date = str(date)
    return f"{date}-31" if len(date) == 7 else date


def seen_labels(model_cfg, year, exam_date=None):
    """Label of an edition for a model: (main from unseen_from, strict from release date;
    'ambiguous' between training cutoff and release)."""
    unseen_from = model_cfg.get("unseen_from")
    main = None if unseen_from is None else ("unseen" if int(year) >= int(unseen_from) else "seen")
    release, cutoff = model_cfg.get("release"), model_cfg.get("training_cutoff")
    if not exam_date or not release:
        return main, None
    exam_date = str(exam_date)
    if exam_date > _end_of_month(release):
        strict = "unseen"
    elif cutoff and exam_date <= _end_of_month(cutoff):
        strict = "seen"
    else:
        strict = "ambiguous"
    return main, strict
