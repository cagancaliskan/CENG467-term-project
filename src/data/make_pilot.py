"""Carve a deterministic pilot subset for Day 3 prompt validation.

Reads MLSUM-TR train.jsonl (already filtered by load_mlsum.py) and selects N
articles whose combined token count fits a target window — biased toward median
length so the pilot reflects typical articles, not edge cases.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("data.pilot")


def main() -> None:
    p = argparse.ArgumentParser(description="Sample a pilot subset for prompt validation.")
    p.add_argument("--input", default="data/raw/mlsum_tr/train.jsonl")
    p.add_argument("--out", default="data/raw/mlsum_tr/pilot_100.jsonl")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--min-chars", type=int, default=800,
                    help="Skip very short articles (not representative).")
    p.add_argument("--max-chars", type=int, default=4000,
                    help="Skip very long articles (atypical).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    rows = list(read_jsonl(args.input))
    LOG.info("Loaded %d rows from %s", len(rows), args.input)
    typical = [r for r in rows if args.min_chars <= len(r["article"]) <= args.max_chars]
    LOG.info("  typical-length subset: %d", len(typical))
    rng.shuffle(typical)
    chosen = typical[: args.n]

    n = write_jsonl(args.out, chosen)
    char_lens = [len(r["article"]) for r in chosen]
    LOG.info("Wrote %d pilot rows -> %s | char-len min/median/max = %d/%d/%d",
              n, args.out, min(char_lens), sorted(char_lens)[len(char_lens) // 2], max(char_lens))


if __name__ == "__main__":
    main()
