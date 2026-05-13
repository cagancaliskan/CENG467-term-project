"""Export a stratified sample of 30 test predictions for hand-labeling.

Reads all six system prediction files from outputs/predictions/ and the source
articles + references from data/raw/mlsum_tr/test.jsonl. Stratifies by source
article length (10 short / 10 medium / 10 long) and produces a single CSV
where each row is one (article, system) pair with blank columns for the human
labeler to fill in five binary judgments.

Output columns:
    id, system, article_snippet, reference, prediction,
    factual_correct, completeness, fluency, morpho_correct, no_mode_collapse,
    notes

The labeler marks each binary column as 1 (good) or 0 (bad), and may add free
text in `notes`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from src.utils.io import read_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("eval.qual_export")

SYSTEMS = {
    "B1": "outputs/predictions/B1_zeroshot.jsonl",
    "B2": "outputs/predictions/B2_human.jsonl",
    "B3a": "outputs/predictions/B3a_gpt.jsonl",
    "B3b": "outputs/predictions/B3b_claude.jsonl",
    "S-gpt": "outputs/predictions/S_gpt.jsonl",
    "S-claude": "outputs/predictions/S_claude.jsonl",
}


def _len_bucket(n_chars: int) -> str:
    if n_chars < 800:
        return "short"
    if n_chars < 2000:
        return "medium"
    return "long"


def main() -> None:
    p = argparse.ArgumentParser(description="Export stratified qual-analysis CSV for hand-labeling.")
    p.add_argument("--test", default="data/raw/mlsum_tr/test.jsonl")
    p.add_argument("--out", default="outputs/results/qual_labels_blank.csv")
    p.add_argument("--n-per-bucket", type=int, default=10,
                    help="Articles per length bucket (short/medium/long). 10*3=30 total articles, x6 systems = 180 rows.")
    p.add_argument("--snippet-chars", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    # Load test articles by id
    test_rows = {r["id"]: r for r in read_jsonl(args.test)}
    LOG.info("Loaded %d test articles", len(test_rows))

    # Load all system predictions
    sys_preds: dict[str, dict[str, str]] = {}
    for label, path in SYSTEMS.items():
        rows = list(read_jsonl(path))
        sys_preds[label] = {r["id"]: r["prediction"] for r in rows}
        LOG.info("  %s: %d predictions", label, len(sys_preds[label]))

    # Restrict to articles that have predictions in all 6 systems
    common = set.intersection(*[set(d) for d in sys_preds.values()])
    LOG.info("Articles with predictions in all 6 systems: %d", len(common))

    # Stratify by article length
    buckets = {"short": [], "medium": [], "long": []}
    for aid in common:
        if aid not in test_rows:
            continue
        buckets[_len_bucket(len(test_rows[aid]["article"]))].append(aid)

    LOG.info("Buckets: short=%d medium=%d long=%d",
              len(buckets["short"]), len(buckets["medium"]), len(buckets["long"]))

    sampled = []
    for b in ("short", "medium", "long"):
        rng.shuffle(buckets[b])
        sampled.extend(buckets[b][: args.n_per_bucket])

    LOG.info("Sampled %d articles (10 per bucket)", len(sampled))

    # Build CSV rows
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "bucket", "system",
            "article_snippet", "reference", "prediction",
            "factual_correct", "completeness", "fluency",
            "morpho_correct", "no_mode_collapse",
            "notes",
        ])
        for aid in sampled:
            row = test_rows[aid]
            article = row["article"]
            snippet = (article[: args.snippet_chars] + "...") if len(article) > args.snippet_chars else article
            bucket = _len_bucket(len(article))
            for sys_label in SYSTEMS.keys():
                pred = sys_preds[sys_label].get(aid, "")
                w.writerow([
                    aid, bucket, sys_label,
                    snippet, row.get("reference", ""), pred,
                    "", "", "", "", "",  # blanks for labels (0/1 each)
                    "",  # notes
                ])

    LOG.info("Wrote %d rows -> %s", 30 * 6, out_path)
    LOG.info("\nLabel instructions (mark each column as 1 = good / 0 = bad):")
    LOG.info("  factual_correct   — no hallucinated entities/numbers/dates")
    LOG.info("  completeness      — captures the article's lede (who/what)")
    LOG.info("  fluency           — grammatical Turkish, no awkward phrasings")
    LOG.info("  morpho_correct    — correct noun cases and verb tenses")
    LOG.info("  no_mode_collapse  — no repeated phrases or filler text")


if __name__ == "__main__":
    main()
