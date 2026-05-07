"""Fine-tune mT5-small with LoRA adapters on Turkish summarization data.

Sized for Colab free-tier T4 (15GB VRAM):
- bf16 disabled (T4 lacks bf16); fp16 enabled.
- per_device_train_batch_size=4, gradient_accumulation_steps=4 -> effective batch 16.
- max_input_length=512, max_target_length=128 keeps tensor sizes modest.

Reads HF-style JSONL produced by src.data.prepare_synthetic with fields
{id, article, target}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.student.peft_utils import wrap_with_lora
from src.utils.io import dump_run_config, read_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("student.train")


def _load_split(path: str) -> Dataset:
    rows = list(read_jsonl(path))
    return Dataset.from_list(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Train mT5-small + LoRA on summarization JSONL.")
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="google/mt5-small")
    p.add_argument("--max-input-length", type=int, default=512)
    p.add_argument("--max-target-length", type=int, default=128)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-epochs", type=float, default=3.0)
    p.add_argument("--per-device-train-batch", type=int, default=4)
    p.add_argument("--per-device-eval-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=None,
                    help="Cap training rows after loading; useful for size ablations.")
    p.add_argument("--source-prefix", default="özetle: ",
                    help="mT5 was pretrained without task prefixes, but a short Turkish "
                          "prefix consistently helps generation quality.")
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_run_config(out_dir, vars(args))

    LOG.info("Loading tokenizer + model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model = wrap_with_lora(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    LOG.info("Trainable params: %d / %d (%.3f%%)", trainable, total, 100 * trainable / total)

    train_ds = _load_split(args.train_file)
    val_ds = _load_split(args.val_file)
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    LOG.info("Loaded train=%d val=%d", len(train_ds), len(val_ds))

    def preprocess(batch: dict) -> dict:
        inputs = [args.source_prefix + (a or "") for a in batch["article"]]
        targets = batch["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols_to_remove = [c for c in train_ds.column_names if c not in {"input_ids", "attention_mask", "labels"}]
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=cols_to_remove,
                              desc="Tokenizing train")
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=cols_to_remove,
                          desc="Tokenizing val")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.per_device_train_batch * args.grad_accum)))
    LOG.info("Estimated steps/epoch=%d total_steps=%d", steps_per_epoch,
              int(steps_per_epoch * args.num_epochs))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch,
        per_device_eval_batch_size=args.per_device_eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        predict_with_generate=False,   # we evaluate generation separately to keep training fast
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    LOG.info("Starting training")
    trainer.train()

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = trainer.evaluate()
    with open(out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOG.info("Final eval metrics: %s", metrics)


if __name__ == "__main__":
    main()
