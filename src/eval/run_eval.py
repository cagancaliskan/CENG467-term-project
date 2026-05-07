"""Score one or more system prediction files and emit a unified results table.

Each --pred argument is parsed as label=path, e.g.:
    --pred B1=outputs/predictions/B1_zeroshot.jsonl \
    --pred B2=outputs/predictions/B2_human.jsonl   \
    --pred S-gpt=outputs/predictions/S_gpt.jsonl

Each predictions JSONL must have fields {id, prediction, reference}.

Outputs:
    <out-json>      Aggregated metric table for all systems.
    <out-jsonl>     Per-example scores (only when --per-example is set).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.bertscore_eval import compute_bertscore
from src.eval.error_analysis import aggregate, detect
from src.eval.rouge_tr import compute_rouge
from src.utils.io import read_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("eval.run")


def _load(path: str) -> list[dict]:
    rows = list(read_jsonl(path))
    rows = [r for r in rows if r.get("prediction") and r.get("reference")]
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Score system predictions on a test set.")
    p.add_argument("--pred", action="append", required=True,
                    help="Repeatable. Format: label=path/to/predictions.jsonl")
    p.add_argument("--out-json", required=True, help="Aggregated metrics JSON.")
    p.add_argument("--out-jsonl", default=None, help="Optional per-example metrics JSONL.")
    p.add_argument("--metrics", nargs="+", default=["rouge", "bertscore", "errors"],
                    choices=["rouge", "bertscore", "errors"])
    p.add_argument("--bertscore-model", default="xlm-roberta-large")
    p.add_argument("--bertscore-batch", type=int, default=8)
    p.add_argument("--rouge-modes", nargs="+", default=["standard", "stem"],
                    choices=["standard", "stem"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    per_example: list[dict] = []

    for spec in args.pred:
        if "=" not in spec:
            raise ValueError(f"Bad --pred (need label=path): {spec!r}")
        label, path = spec.split("=", 1)
        rows = _load(path)
        LOG.info("[%s] %d examples from %s", label, len(rows), path)
        if not rows:
            LOG.warning("[%s] no scorable rows; skipping", label)
            continue

        preds = [r["prediction"] for r in rows]
        refs = [r["reference"] for r in rows]

        sysres: dict = {"label": label, "path": path, "n": len(rows)}

        if "rouge" in args.metrics:
            for mode in args.rouge_modes:
                r = compute_rouge(preds, refs, mode=mode)
                sysres[f"rouge_{mode}"] = r.to_dict()
                LOG.info("  ROUGE/%s | R1=%.4f R2=%.4f RL=%.4f", mode, r.rouge1, r.rouge2, r.rougeL)

        if "bertscore" in args.metrics:
            b = compute_bertscore(preds, refs,
                                    model_type=args.bertscore_model,
                                    batch_size=args.bertscore_batch)
            sysres["bertscore"] = b.to_dict()
            LOG.info("  BERTScore | P=%.4f R=%.4f F1=%.4f", b.precision, b.recall, b.f1)

        if "errors" in args.metrics:
            flags = [
                detect(article=r.get("article", ""), prediction=r["prediction"], reference=r["reference"])
                for r in rows
            ]
            sysres["errors"] = aggregate(flags)
            LOG.info("  Errors | rep=%.3f halluc#=%.3f extract=%.3f morph=%.3f",
                      sysres["errors"]["frac_repetition"],
                      sysres["errors"]["frac_hallucinated_numbers"],
                      sysres["errors"]["mean_extractive_overlap"],
                      sysres["errors"]["frac_morph_flag"])

            if args.out_jsonl:
                for r, f in zip(rows, flags):
                    per_example.append({
                        "system": label,
                        "id": r["id"],
                        "prediction": r["prediction"],
                        "reference": r["reference"],
                        **f.__dict__,
                    })

        results[label] = sysres

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOG.info("Wrote aggregated metrics -> %s", args.out_json)

    if args.out_jsonl and per_example:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for rec in per_example:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        LOG.info("Wrote per-example flags -> %s", args.out_jsonl)


if __name__ == "__main__":
    main()
