"""Per-example scoring driver for the camera-ready revision.

Differs from src/eval/run_eval.py in three ways that the revision needs:

1. Emits PER-EXAMPLE scores, which paired bootstrap CIs and paired
   significance tests require (run_eval.py only ever produced corpus means).
2. Reports ROUGE precision and recall SEPARATELY, not just F1.  A system with
   high precision and low recall is omitting content -- the most direct
   quantitative answer to the reviewer's point that the students' shorter,
   more source-anchored summaries may drop important information.
3. Handles empty predictions explicitly.  Stripping sentinels can empty out a
   degenerate prediction; run_eval.py silently DROPPED such rows, which would
   change n between the dirty and clean arms and break the pairing.  Here every
   row is scored (empty -> 0.0) and the legacy drop-empty aggregate is also
   reported so the published numbers stay comparable.

Usage::

    python -m src.eval.run_eval_v2 \
        --pred B1=outputs/predictions/clean/B1_zeroshot.jsonl \
        --pred B2=outputs/predictions/clean/B2_human.jsonl \
        --metrics rouge errors bertscore \
        --per-example-dir outputs/results/v2/per_example \
        --out-json outputs/results/v2/main_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

from src.eval.error_analysis import detect
from src.eval.rouge_tr import TurkishStemTokenizer, TurkishTokenizer
from src.eval.stats import bootstrap_ci_mean
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("eval.run_eval_v2")

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
TOKENIZERS = {"standard": TurkishTokenizer(), "stem": TurkishStemTokenizer(prefix=5)}


def _scorers() -> dict:
    return {
        mode: rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False, tokenizer=tok)
        for mode, tok in TOKENIZERS.items()
    }


def _score_rows(rows: list[dict], scorers: dict) -> list[dict]:
    out = []
    for r in rows:
        pred = (r.get("prediction") or "").strip()
        ref = (r.get("reference") or "").strip()
        art = r.get("article") or ""
        rec: dict = {"id": r.get("id"), "empty_pred": int(not pred), "empty_ref": int(not ref)}

        for mode, sc in scorers.items():
            if pred and ref:
                s = sc.score(ref, pred)
                for rt in ROUGE_TYPES:
                    rec[f"{mode}_{rt}_p"] = s[rt].precision
                    rec[f"{mode}_{rt}_r"] = s[rt].recall
                    rec[f"{mode}_{rt}_f"] = s[rt].fmeasure
            else:
                for rt in ROUGE_TYPES:
                    rec[f"{mode}_{rt}_p"] = 0.0
                    rec[f"{mode}_{rt}_r"] = 0.0
                    rec[f"{mode}_{rt}_f"] = 0.0

        f = detect(article=art, prediction=pred, reference=ref)
        rec["err_repetition"] = float(f.has_repetition)
        rec["err_halluc_num"] = float(f.has_hallucinated_numbers)
        rec["err_extract"] = f.extractive_overlap
        rec["err_len_ratio"] = f.length_ratio
        rec["err_morph"] = float(f.morpho_red_flag)
        rec["n_pred_tokens"] = len(pred.split())
        rec["n_ref_tokens"] = len(ref.split())
        out.append(rec)
    return out


def _add_bertscore(rows: list[dict], per_ex: list[dict], model_type: str, batch_size: int) -> None:
    import torch
    from bert_score import score as bs_score

    idx = [i for i, r in enumerate(rows)
           if (r.get("prediction") or "").strip() and (r.get("reference") or "").strip()]
    for rec in per_ex:
        rec["bs_p"] = rec["bs_r"] = rec["bs_f1"] = 0.0
    if not idx:
        return

    preds = [rows[i]["prediction"].strip() for i in idx]
    refs = [rows[i]["reference"].strip() for i in idx]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    P, R, F = bs_score(preds, refs, model_type=model_type, lang="tr", device=device,
                       batch_size=batch_size, rescale_with_baseline=False, verbose=False)
    for j, i in enumerate(idx):
        per_ex[i]["bs_p"] = float(P[j]); per_ex[i]["bs_r"] = float(R[j]); per_ex[i]["bs_f1"] = float(F[j])

    del P, R, F
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _aggregate(per_ex: list[dict], n_boot: int, seed: int) -> dict:
    if not per_ex:
        return {}
    keys = [k for k in per_ex[0]
            if k not in {"id"} and isinstance(per_ex[0][k], (int, float))]
    agg: dict = {"n": len(per_ex)}
    for k in keys:
        vals = [r[k] for r in per_ex]
        ci = bootstrap_ci_mean(vals, n_boot=n_boot, seed=seed)
        agg[k] = {"mean": ci.estimate, "ci_low": ci.low, "ci_high": ci.high, "ci_method": ci.method}

    # Legacy aggregate: mean over rows where BOTH pred and ref are non-empty,
    # i.e. exactly run_eval.py's filter, so v1 numbers remain comparable.
    keep = [r for r in per_ex if not r["empty_pred"] and not r["empty_ref"]]
    agg["legacy_drop_empty"] = {
        "n": len(keep),
        **{k: (sum(r[k] for r in keep) / len(keep) if keep else 0.0)
           for k in keys if k not in {"empty_pred", "empty_ref"}},
    }
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Per-example scoring with bootstrap CIs.")
    p.add_argument("--pred", action="append", required=True, help="label=path/to/preds.jsonl")
    p.add_argument("--out-json", required=True)
    p.add_argument("--per-example-dir", default=None,
                   help="Write <label>.jsonl of per-example scores here (needed for paired tests).")
    p.add_argument("--metrics", nargs="+", default=["rouge", "errors"],
                   choices=["rouge", "errors", "bertscore"])
    p.add_argument("--bertscore-model", default="xlm-roberta-large")
    p.add_argument("--bertscore-batch", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    scorers = _scorers()
    results: dict = {}

    for spec in args.pred:
        if "=" not in spec:
            raise ValueError(f"Bad --pred (need label=path): {spec!r}")
        label, path = spec.split("=", 1)
        rows = list(read_jsonl(path))
        LOG.info("[%s] %d rows from %s", label, len(rows), path)

        per_ex = _score_rows(rows, scorers)
        if "bertscore" in args.metrics:
            LOG.info("[%s] BERTScore (%s)...", label, args.bertscore_model)
            _add_bertscore(rows, per_ex, args.bertscore_model, args.bertscore_batch)

        if args.per_example_dir:
            outp = Path(args.per_example_dir) / f"{label}.jsonl"
            write_jsonl(outp, per_ex)
            LOG.info("[%s] per-example -> %s", label, outp)

        agg = _aggregate(per_ex, args.n_boot, args.seed)
        agg["label"] = label
        agg["path"] = path
        results[label] = agg

        r1 = agg["standard_rouge1_f"]["mean"]
        r1p = agg["standard_rouge1_p"]["mean"]
        r1r = agg["standard_rouge1_r"]["mean"]
        LOG.info("  R1 std F=%.4f [%.4f, %.4f]  P=%.4f  R=%.4f  empty=%d  halluc#=%.4f",
                 r1, agg["standard_rouge1_f"]["ci_low"], agg["standard_rouge1_f"]["ci_high"],
                 r1p, r1r, int(sum(r["empty_pred"] for r in per_ex)),
                 agg["err_halluc_num"]["mean"])

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOG.info("Wrote aggregate -> %s", args.out_json)


if __name__ == "__main__":
    main()
