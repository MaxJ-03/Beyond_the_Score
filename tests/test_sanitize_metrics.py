"""Unit tests of the contamination-test scores and vLLM log-prob parsing (no GPU needed)."""

import math
from types import SimpleNamespace

import pytest

import sanitize_metrics as sm
from items import Item
from vllm_scoring import prompt_token_logprobs

# Ten tokens with known log-probabilities. Sorted: -4.0, -3.0, -2.0, -1.0, -0.5, -0.4, -0.3,
# -0.2, -0.1, -0.05 (mean -1.155).
TOY = [-0.1, -0.2, -0.5, -1.0, -2.0, -3.0, -0.3, -0.4, -4.0, -0.05]


# ---------------------------------------------------------------- Min-K%

def test_min_k_on_known_log_probabilities():
    assert sm.min_k(TOY, 0.2) == pytest.approx(-3.5)  # the 2 least likely tokens: -4.0, -3.0
    assert sm.min_k(TOY, 0.3) == pytest.approx(-3.0)  # -4.0, -3.0, -2.0
    assert sm.min_k(TOY, 0.6) == pytest.approx((-4.0 - 3.0 - 2.0 - 1.0 - 0.5 - 0.4) / 6)
    assert sm.min_k(TOY, 1.0) == pytest.approx(sum(TOY) / 10)


def test_min_k_uses_at_least_one_token_and_handles_empty_input():
    assert sm.min_k(TOY, 0.05) == pytest.approx(-4.0)  # int(10 * 0.05) = 0 -> one token
    assert math.isnan(sm.min_k([], 0.2))


def test_min_k_is_at_most_the_mean():
    for k in (0.05, 0.1, 0.2, 0.5, 1.0):
        assert sm.min_k(TOY, k) <= sum(TOY) / len(TOY) + 1e-12


def test_likelihood_scores():
    scores = sm.likelihood_scores(TOY, "texto", [0.2, 0.05], prefix="stem_")
    assert scores["stem_mink_20"] == pytest.approx(-3.5)
    assert scores["stem_mink_5"] == pytest.approx(-4.0)
    assert scores["stem_mean_logprob"] == pytest.approx(-1.155)
    assert scores["stem_perplexity"] == pytest.approx(math.exp(1.155))
    assert scores["stem_zlib_ratio"] > 0


def test_text_logprobs_excludes_bos_and_special_tokens():
    # With a BOS token (id 2) at position 0, every text token is scored.
    assert sm.text_logprobs([2, 10, 11, 12], [None, -1.0, -2.0, -3.0], {2}) == [-1.0, -2.0, -3.0]
    # A special token inside the text (id 1) is left out.
    assert sm.text_logprobs([2, 10, 1, 12], [None, -1.0, -9.0, -3.0], {1, 2}) == [-1.0, -3.0]
    # Without a BOS token, the first text token has no context and is not scored.
    assert sm.text_logprobs([10, 11, 12], [None, -2.0, -3.0]) == [-2.0, -3.0]
    with pytest.raises(ValueError):
        sm.text_logprobs([10, 11], [None])


def test_prompt_token_logprobs_reads_the_vllm_output_format():
    # vLLM 0.31: None for the first prompt token, then {token id: Logprob} with the prompt
    # token and the most likely token.
    lp = lambda value: SimpleNamespace(logprob=value, rank=1, decoded_token=None)  # noqa: E731
    output = [None, {11: lp(-0.5), 99: lp(-0.1)}, {12: lp(-2.0)}]
    assert prompt_token_logprobs(output, [10, 11, 12]) == [None, -0.5, -2.0]
    with pytest.raises(RuntimeError):
        prompt_token_logprobs(output[:2], [10, 11, 12])
    with pytest.raises(RuntimeError):
        prompt_token_logprobs([{10: lp(-1.0)}, *output[1:]], [10, 11, 12])


# ---------------------------------------------------------------- question text and split

def item(**kwargs):
    defaults = dict(item_id="mir2022-026", dataset="mir_2022_es", question="¿Cuál es la causa?:",
                    options={"A": "Uno.", "B": "Dos.", "C": "Tres.", "D": "Cuatro."}, answer="C", year="2022")
    return Item(**{**defaults, **kwargs})


def test_question_text_follows_the_booklet():
    text_cfg = {"option_labels": "numbers", "include_options": True}
    assert sm.question_text(item(), text_cfg) == "¿Cuál es la causa?:\n1. Uno.\n2. Dos.\n3. Tres.\n4. Cuatro."
    assert sm.question_text(item(), {**text_cfg, "option_labels": "letters"}).endswith("\nD. Cuatro.")
    assert sm.question_text(item(), {**text_cfg, "include_options": False}) == "¿Cuál es la causa?:"
    numbered = sm.question_text(item(), {**text_cfg, "include_question_number": True})
    assert numbered.startswith("26. ¿Cuál")
    with pytest.raises(ValueError):
        sm.question_text(item(), {**text_cfg, "option_labels": "roman"})


def test_split_at_words_keeps_the_layout():
    text = "¿Cuál es la causa?:\n1. Uno.\n2. Dos."
    prefix, reference, n_words, n_prefix = sm.split_at_words(text, 0.5)
    assert (prefix, reference, n_words, n_prefix) == ("¿Cuál es la causa?:", "\n1. Uno.\n2. Dos.", 8, 4)
    assert prefix + reference == text
    # At least one word on each side.
    assert sm.split_at_words("una dos", 0.0)[0] == "una"
    assert sm.split_at_words("una dos", 1.0)[1] == " dos"
    with pytest.raises(ValueError):
        sm.split_at_words("una", 0.5)


# ---------------------------------------------------------------- completion scores

def test_rouge_l_keeps_spanish_words_intact():
    assert sm.words("Señale la afirmación FALSA") == ["señale", "la", "afirmación", "falsa"]
    assert sm.rouge_l_f1("señale la afirmación falsa", "señale la afirmación falsa") == pytest.approx(1.0)
    # LCS of 3 words out of 4 on both sides: precision = recall = 0.75.
    assert sm.rouge_l_f1("señale la afirmación falsa", "señale la afirmación verdadera") == pytest.approx(0.75)
    assert sm.rouge_l_f1("uno dos", "tres cuatro") == 0.0
    assert sm.rouge_l_f1("uno dos", "") == 0.0


def test_levenshtein_ratio():
    assert sm.levenshtein_ratio("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert sm.levenshtein_ratio("abc", "abc") == 1.0
    assert sm.levenshtein_ratio("", "") == 1.0
    assert sm.levenshtein_ratio("abc", "") == 0.0


def test_pure_python_distances_match_the_definitions():
    assert sm._levenshtein("kitten", "sitting") == 3
    assert sm._lcs_length(["a", "b", "c", "d"], ["a", "c", "d", "e"]) == 3


def test_completion_scores():
    scores = sm.completion_scores(" 1. Uno.", " 1. Uno.", [5, 6, 7], [5, 6, 7], 0.9)
    assert scores == {"rouge_l_f1": 1.0, "levenshtein_ratio": 1.0, "lcp_tokens": 3, "lcp_fraction": 1.0,
                      "near_verbatim": True}
    scores = sm.completion_scores(" 1. Uno.", " 2. Dos.", [5, 6, 7], [5, 8, 7], 0.9)
    assert scores["lcp_tokens"] == 1 and not scores["near_verbatim"]


# ---------------------------------------------------------------- seen or unseen

GEMMA_LIKE = {"unseen_from": 2025, "training_cutoff": "2024-08", "release": "2025-03-12"}
MONTH_RELEASE = {"unseen_from": 2024, "training_cutoff": "2023-03", "release": "2024-08"}


def test_seen_labels():
    assert sm.seen_labels(GEMMA_LIKE, 2024, "2024-01-20") == ("seen", "seen")
    # After the training cutoff but before the release: unseen by cutoff, ambiguous by release.
    assert sm.seen_labels(GEMMA_LIKE, 2025, "2025-01-25") == ("unseen", "ambiguous")
    assert sm.seen_labels(GEMMA_LIKE, 2026, "2026-01-24") == ("unseen", "unseen")
    # A cutoff month counts until its end; a release given as a month counts as its last day.
    assert sm.seen_labels(MONTH_RELEASE, 2023, "2023-03-30") == ("seen", "seen")
    assert sm.seen_labels(MONTH_RELEASE, 2024, "2024-08-30") == ("unseen", "ambiguous")
    assert sm.seen_labels(MONTH_RELEASE, 2025, "2025-01-25") == ("unseen", "unseen")
    # Without an exam date only the main label is known.
    assert sm.seen_labels(GEMMA_LIKE, 2025, None) == ("unseen", None)


def test_every_model_and_edition_gets_a_label():
    from config import load_models, load_yaml
    from sanitize import evaluated_models

    models, settings = load_models(), load_yaml("sanitize.yaml")
    for key in evaluated_models(models):
        for year, date in settings["exam_dates"].items():
            main, strict = sm.seen_labels(models[key], year, date)
            assert main in sm.SEEN_LABELS and strict in sm.STRICT_LABELS
