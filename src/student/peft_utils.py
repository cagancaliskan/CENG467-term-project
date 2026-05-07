"""PEFT/LoRA helper for mT5.

Centralized so the same target modules and dtype handling are used in train and
infer paths.
"""
from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel


# mT5 uses T5-style attention (q, v in self-attn; k optional). We target q/v which
# is the most common rank-frugal recipe and is well-tested for T5 family.
DEFAULT_TARGET_MODULES = ["q", "v"]


def wrap_with_lora(
    model: PreTrainedModel,
    rank: int = 8,
    alpha: int | None = None,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> PreTrainedModel:
    """Apply a LoRA adapter sized for mT5-small on a 15GB T4."""
    if alpha is None:
        # Hu et al. recommend alpha == rank or 2*rank; 2x scales updates more aggressively.
        alpha = rank * 2

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
    )
    return get_peft_model(model, cfg)
