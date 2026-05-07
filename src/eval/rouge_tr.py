"""Turkish-aware ROUGE.

Two modes:
- 'standard': rouge_score with a Turkish-aware tokenizer (lowercase + Unicode-aware
  word splitting, accent preservation). Used as the primary metric.
- 'stem': same tokenization plus a 5-character prefix stemmer to mitigate Turkish
  agglutination. Reported alongside the standard score so the report can argue
  that the gap between the two is informative.

Both modes return ROUGE-1, ROUGE-2, and ROUGE-L F1 averaged over examples.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

# Strip everything that is not a Unicode letter, number, or whitespace.
# \W in re.UNICODE keeps Turkish letters intact when LOCALE is unset.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


class TurkishTokenizer(Tokenizer):
    """Whitespace tokenizer with Unicode-aware lowercasing."""
    def tokenize(self, text: str) -> list[str]:
        return _normalize(text).split()


class TurkishStemTokenizer(Tokenizer):
    """Apply a 5-char prefix stemmer to soften morphological variation."""
    def __init__(self, prefix: int = 5):
        self.prefix = prefix

    def tokenize(self, text: str) -> list[str]:
        return [w[: self.prefix] for w in _normalize(text).split() if w]


@dataclass
class RougeResult:
    rouge1: float
    rouge2: float
    rougeL: float
    mode: str
    n: int

    def to_dict(self) -> dict:
        return {"rouge1": self.rouge1, "rouge2": self.rouge2, "rougeL": self.rougeL,
                "mode": self.mode, "n": self.n}


def compute_rouge(predictions: list[str], references: list[str], mode: str = "standard") -> RougeResult:
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: preds={len(predictions)} refs={len(references)}")
    if mode == "standard":
        tokenizer = TurkishTokenizer()
    elif mode == "stem":
        tokenizer = TurkishStemTokenizer(prefix=5)
    else:
        raise ValueError(f"Unknown rouge mode: {mode!r}")

    scorer = rouge_scorer.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=tokenizer,
    )

    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for pred, ref in zip(predictions, references):
        if not (pred and ref):
            continue
        s = scorer.score(ref, pred)
        sums["rouge1"] += s["rouge1"].fmeasure
        sums["rouge2"] += s["rouge2"].fmeasure
        sums["rougeL"] += s["rougeL"].fmeasure
        n += 1
    if n == 0:
        return RougeResult(0.0, 0.0, 0.0, mode, 0)
    return RougeResult(sums["rouge1"] / n, sums["rouge2"] / n, sums["rougeL"] / n, mode, n)
