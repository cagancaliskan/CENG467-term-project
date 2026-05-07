"""BERTScore wrapper using a multilingual encoder appropriate for Turkish."""
from __future__ import annotations

import gc
from dataclasses import dataclass

import torch


# xlm-roberta-large is the standard multilingual baseline cited by BERTScore authors.
DEFAULT_MODEL = "xlm-roberta-large"


@dataclass
class BertScoreResult:
    precision: float
    recall: float
    f1: float
    model_type: str
    n: int

    def to_dict(self) -> dict:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1,
                "model_type": self.model_type, "n": self.n}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = DEFAULT_MODEL,
    batch_size: int = 8,
    lang: str = "tr",
) -> BertScoreResult:
    """Compute BERTScore. Falls back to CPU if CUDA is unavailable.

    Memory note: xlm-roberta-large + batch_size=8 fits on a T4 alongside the
    student model offloaded to CPU.
    """
    from bert_score import score as bs_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F = bs_score(
        predictions,
        references,
        model_type=model_type,
        lang=lang,
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=False,   # keep raw scores; we report deltas across systems
        verbose=False,
    )
    out = BertScoreResult(
        precision=float(P.mean().item()),
        recall=float(R.mean().item()),
        f1=float(F.mean().item()),
        model_type=model_type,
        n=len(predictions),
    )
    # Free GPU mem aggressively so the same Colab session can score multiple systems.
    del P, R, F
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
