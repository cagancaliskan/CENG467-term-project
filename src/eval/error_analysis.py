"""Heuristic flags for qualitative error analysis.

These flags are not ground truth — they surface candidates for hand annotation.
Categories follow the project plan: hallucination, omission, repetition,
morphological errors. We pair the flags with token statistics so the report can
claim concrete differences between systems.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass
class ErrorFlags:
    has_repetition: bool
    repetition_ratio: float
    has_hallucinated_numbers: bool
    extractive_overlap: float       # unigram overlap with article (low = abstractive but riskier)
    length_ratio: float             # prediction tokens / reference tokens
    morpho_red_flag: bool           # very long agglutinated tokens vs. article distribution


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def detect(article: str, prediction: str, reference: str | None = None) -> ErrorFlags:
    pred_tokens = _tokens(prediction)
    art_tokens = _tokens(article)
    ref_tokens = _tokens(reference or "")

    # Repetition: any 4-gram appears more than once.
    has_rep = False
    rep_ratio = 0.0
    if len(pred_tokens) >= 4:
        ngrams = [tuple(pred_tokens[i : i + 4]) for i in range(len(pred_tokens) - 3)]
        ctr = Counter(ngrams)
        repeated = sum(c for c in ctr.values() if c > 1)
        has_rep = any(c > 1 for c in ctr.values())
        rep_ratio = repeated / max(1, len(ngrams))

    # Hallucinated numbers: numeric tokens in prediction not present in article.
    pred_numbers = set(_NUM_RE.findall(prediction))
    art_numbers = set(_NUM_RE.findall(article))
    halluc_nums = bool(pred_numbers - art_numbers)

    # Extractive overlap: how many prediction unigrams come from the article.
    art_set = set(art_tokens)
    overlap = sum(1 for t in pred_tokens if t in art_set)
    overlap_ratio = overlap / max(1, len(pred_tokens))

    # Length ratio vs. reference (or article when no reference present).
    target_len = len(ref_tokens) if ref_tokens else max(1, len(art_tokens) // 10)
    length_ratio = len(pred_tokens) / max(1, target_len)

    # Morphological red flag: any prediction token is much longer than the 95th
    # percentile of article tokens. Catches the model dropping concatenated affixes.
    morph_flag = False
    if art_tokens:
        sorted_lens = sorted(len(t) for t in art_tokens)
        cutoff = sorted_lens[int(0.95 * (len(sorted_lens) - 1))]
        morph_flag = any(len(t) > cutoff + 4 for t in pred_tokens)

    return ErrorFlags(
        has_repetition=has_rep,
        repetition_ratio=rep_ratio,
        has_hallucinated_numbers=halluc_nums,
        extractive_overlap=overlap_ratio,
        length_ratio=length_ratio,
        morpho_red_flag=morph_flag,
    )


def aggregate(flags: list[ErrorFlags]) -> dict:
    if not flags:
        return {}
    n = len(flags)
    return {
        "n": n,
        "frac_repetition": sum(1 for f in flags if f.has_repetition) / n,
        "mean_repetition_ratio": sum(f.repetition_ratio for f in flags) / n,
        "frac_hallucinated_numbers": sum(1 for f in flags if f.has_hallucinated_numbers) / n,
        "mean_extractive_overlap": sum(f.extractive_overlap for f in flags) / n,
        "mean_length_ratio": sum(f.length_ratio for f in flags) / n,
        "frac_morph_flag": sum(1 for f in flags if f.morpho_red_flag) / n,
    }
