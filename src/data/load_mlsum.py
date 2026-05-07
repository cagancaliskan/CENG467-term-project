"""MLSUM-TR loader.

Wraps `datasets.load_dataset("mlsum", "tu")` and emits clean JSONL with stable
SHA-1 article IDs we can use as cache keys for teacher API calls.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

from datasets import load_dataset

from src.utils.io import write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("data.mlsum")

# MLSUM-TR has a few near-empty rows; filter conservatively.
MIN_ARTICLE_CHARS = 200
MIN_SUMMARY_CHARS = 20


def _stable_id(text: str) -> str:
    """Deterministic article ID independent of HF row indexing."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _iter_split(split_name: str, n: int | None, seed: int) -> Iterable[dict]:
    LOG.info("Loading mlsum/tu split=%s ...", split_name)
    ds = load_dataset("mlsum", "tu", split=split_name)
    LOG.info("  raw size: %d", len(ds))

    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n * 2))  # over-sample, then filter

    kept = 0
    for row in ds:
        article = (row.get("text") or "").strip()
        summary = (row.get("summary") or "").strip()
        if len(article) < MIN_ARTICLE_CHARS or len(summary) < MIN_SUMMARY_CHARS:
            continue
        yield {
            "id": _stable_id(article),
            "article": article,
            "reference": summary,
            "topic": row.get("topic"),
            "url": row.get("url"),
            "date": row.get("date"),
            "split": split_name,
            "source": "mlsum_tr",
        }
        kept += 1
        if n is not None and kept >= n:
            break
    LOG.info("  kept %d records after filtering", kept)


def main() -> None:
    p = argparse.ArgumentParser(description="Download and clean MLSUM-TR splits.")
    p.add_argument("--out-dir", default="data/raw/mlsum_tr",
                    help="Output directory for JSONL files.")
    p.add_argument("--n-train", type=int, default=20000,
                    help="How many train rows to keep (None = all).")
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n in (("train", args.n_train), ("validation", args.n_val), ("test", args.n_test)):
        target = out_dir / f"{split}.jsonl"
        n_written = write_jsonl(target, _iter_split(split, n, args.seed))
        LOG.info("Wrote %d rows -> %s", n_written, target)


if __name__ == "__main__":
    main()
