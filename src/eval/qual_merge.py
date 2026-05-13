"""Merge an external judge's labels (just id, system, 5 axes, notes) with the
blank qual CSV that has the article snippets and predictions.

Used when the LLM judge runs outside this codebase (e.g., a chat session with
Claude Opus on claude.ai). Output is compatible with qual_aggregate.py.

Usage:
    python -m src.eval.qual_merge \\
        --blank outputs/results/qual_labels_blank.csv \\
        --judgments outputs/results/opus_judgments.csv \\
        --output outputs/results/qual_labels_filled.csv

The judgments CSV must have columns: id, system, factual_correct,
completeness, fluency, morpho_correct, no_mode_collapse, notes (in any order).
Extra columns are ignored.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.utils.logging import get_logger

LOG = get_logger("eval.qual_merge")

AXES = ["factual_correct", "completeness", "fluency", "morpho_correct", "no_mode_collapse"]


def main() -> None:
    p = argparse.ArgumentParser(description="Merge external judge labels into the blank qual CSV.")
    p.add_argument("--blank", default="outputs/results/qual_labels_blank.csv",
                    help="The original blank CSV (kept rows + columns).")
    p.add_argument("--judgments", required=True,
                    help="External judge CSV with id, system, 5 axis columns, notes.")
    p.add_argument("--output", default="outputs/results/qual_labels_filled.csv",
                    help="Final merged CSV in qual_aggregate.py-compatible format.")
    args = p.parse_args()

    # Load judgments keyed by (id, system)
    with open(args.judgments, "r", encoding="utf-8") as f:
        judge_rows = list(csv.DictReader(f))
    LOG.info("Loaded %d judgment rows from %s", len(judge_rows), args.judgments)

    judgments: dict[tuple[str, str], dict] = {}
    for r in judge_rows:
        key = (r.get("id", "").strip(), r.get("system", "").strip())
        judgments[key] = r

    # Load blank CSV and apply
    with open(args.blank, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    LOG.info("Loaded %d rows from blank CSV", len(rows))

    n_matched = 0
    n_missing = 0
    for r in rows:
        key = (r["id"], r["system"])
        j = judgments.get(key)
        if j is None:
            n_missing += 1
            continue
        for axis in AXES:
            v = j.get(axis, "").strip()
            r[axis] = v
        r["notes"] = j.get("notes", "").strip()
        n_matched += 1

    LOG.info("Matched %d rows, missing %d (rows without a judgment will be blank)", n_matched, n_missing)

    # Write merged CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    LOG.info("Wrote merged CSV -> %s", out_path)


if __name__ == "__main__":
    main()
