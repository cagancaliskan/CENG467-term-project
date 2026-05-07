"""Verify a teacher cache covers an input JSONL.

Usage:
    python -m scripts.check_cache \
        --input data/raw/mlsum_tr/train.jsonl \
        --cache-dir data/synthetic/openai/concise \
        --n 10000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.io import read_jsonl
from src.utils.logging import get_logger

LOG = get_logger("check_cache")


def main() -> None:
    p = argparse.ArgumentParser(description="Check a teacher cache directory against an input JSONL.")
    p.add_argument("--input", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--n", type=int, default=None,
                    help="Only check the first N input rows (matches generate.py --n).")
    args = p.parse_args()

    rows = list(read_jsonl(args.input))
    if args.n is not None:
        rows = rows[: args.n]
    expected_ids = {r["id"] for r in rows}

    cache_dir = Path(args.cache_dir)
    actual = list(cache_dir.glob("*.json"))
    actual_ids = {p.stem for p in actual if p.stem != "manifest"}
    # Filter manifest.jsonl out (not a per-article cache file).
    actual_ids = {i for i in actual_ids if i != "manifest"}

    present = expected_ids & actual_ids
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids

    LOG.info("expected n=%d  cached n=%d  present=%d  missing=%d  extra=%d",
              len(expected_ids), len(actual_ids), len(present), len(missing), len(extra))

    # Sanity: peek at a few cached files for non-empty summaries.
    peeks = list(cache_dir.glob("*.json"))[:5]
    for p_ in peeks:
        try:
            with open(p_, "r", encoding="utf-8") as f:
                rec = json.load(f)
            slen = len(rec.get("summary", "") or "")
            LOG.info("  peek %s: summary_chars=%d teacher=%s prompt=%s",
                      p_.name, slen, rec.get("teacher"), rec.get("prompt_variant"))
        except Exception as e:
            LOG.warning("  failed peek %s: %s", p_, e)

    if missing:
        # Persist the missing IDs so a follow-up generate.py call can complete the gap.
        out_path = cache_dir / "_missing_ids.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for aid in sorted(missing):
                f.write(aid + "\n")
        LOG.warning("Wrote %d missing IDs -> %s", len(missing), out_path)
        # Exit non-zero so CI / scripts can detect the partial run.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
