"""TR-News loader for out-of-domain evaluation.

Falls back to the HuggingFace mirror; if that is unavailable, reads from a
locally provided CSV at --local-csv. Schema after normalization mirrors MLSUM-TR.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

from src.utils.io import write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("data.trnews")

MIN_ARTICLE_CHARS = 200
MIN_SUMMARY_CHARS = 20

# Common HF mirror IDs for TR-News, in order of preference.
HF_CANDIDATES = [
    ("batubayk/TR-News", None),
    ("mukayese/tr-news", None),
]


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize(row: dict, split: str) -> dict | None:
    # TR-News schemas vary by mirror — try the common field names.
    article = (row.get("content") or row.get("text") or row.get("article") or "").strip()
    summary = (row.get("abstract") or row.get("summary") or row.get("highlights") or "").strip()
    if len(article) < MIN_ARTICLE_CHARS or len(summary) < MIN_SUMMARY_CHARS:
        return None
    return {
        "id": _stable_id(article),
        "article": article,
        "reference": summary,
        "topic": row.get("topic") or row.get("category"),
        "url": row.get("url"),
        "split": split,
        "source": "tr_news",
    }


def _iter_hf(split_name: str, n: int | None, seed: int) -> Iterable[dict]:
    from datasets import load_dataset

    last_err: Exception | None = None
    for repo, config in HF_CANDIDATES:
        try:
            ds = load_dataset(repo, config, split=split_name, trust_remote_code=True) if config else load_dataset(repo, split=split_name, trust_remote_code=True)
            LOG.info("Loaded %s split=%s (%d rows)", repo, split_name, len(ds))
            if n is not None and n < len(ds):
                ds = ds.shuffle(seed=seed).select(range(n * 2))
            kept = 0
            for row in ds:
                norm = _normalize(row, split_name)
                if norm is None:
                    continue
                yield norm
                kept += 1
                if n is not None and kept >= n:
                    return
            return
        except Exception as e:
            LOG.warning("Failed %s: %s", repo, e)
            last_err = e
    if last_err is not None:
        raise last_err


def _iter_local(csv_path: Path, n: int | None) -> Iterable[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    LOG.info("Loaded local CSV %s (%d rows)", csv_path, len(df))
    kept = 0
    for _, row in df.iterrows():
        norm = _normalize(row.to_dict(), "test")
        if norm is None:
            continue
        yield norm
        kept += 1
        if n is not None and kept >= n:
            break


def main() -> None:
    p = argparse.ArgumentParser(description="Download or load TR-News for OOD evaluation.")
    p.add_argument("--out", default="data/raw/trnews/test.jsonl")
    p.add_argument("--n", type=int, default=1000, help="Number of articles for OOD eval.")
    p.add_argument("--split", default="test")
    p.add_argument("--local-csv", default=None,
                    help="Path to a local TR-News CSV if HF mirrors are unavailable.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    if args.local_csv:
        n_written = write_jsonl(args.out, _iter_local(Path(args.local_csv), args.n))
    else:
        n_written = write_jsonl(args.out, _iter_hf(args.split, args.n, args.seed))
    LOG.info("Wrote %d rows -> %s", n_written, args.out)


if __name__ == "__main__":
    main()
