"""MLSUM-TR loader with fallbacks.

Tries, in order:
  1. The original `mlsum` HF dataset with config 'tu' (requires datasets<3.0).
  2. A parquet-based community mirror, if available.
  3. csebuetnlp/xlsum, Turkish split — different dataset, but a viable Turkish
     summarization benchmark with the same article/summary schema. We log the
     fallback prominently so the report can disclose it.

Either way, output schema is:
  {id, article, reference, topic, url, date, split, source}
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

from src.utils.io import write_jsonl
from src.utils.logging import get_logger
from src.utils.seed import set_seed

LOG = get_logger("data.mlsum")

MIN_ARTICLE_CHARS = 200
MIN_SUMMARY_CHARS = 20

# Dataset candidates in priority order. Each entry: (repo, config_or_None, schema_kind, source_label).
CANDIDATES = [
    ("mlsum", "tu", "mlsum", "mlsum_tr"),
    ("mukayese/mlsum_tr", None, "mlsum", "mlsum_tr_mukayese"),
    ("csebuetnlp/xlsum", "turkish", "xlsum", "xlsum_tr_fallback"),
]


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _normalize(row: dict, kind: str, split: str, source: str) -> dict | None:
    if kind == "mlsum":
        article = (row.get("text") or row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()
        topic = row.get("topic")
        url = row.get("url")
        date = row.get("date")
    elif kind == "xlsum":
        article = (row.get("text") or "").strip()
        summary = (row.get("summary") or "").strip()
        topic = None
        url = row.get("url")
        date = None
    else:
        return None

    if len(article) < MIN_ARTICLE_CHARS or len(summary) < MIN_SUMMARY_CHARS:
        return None
    return {
        "id": _stable_id(article),
        "article": article,
        "reference": summary,
        "topic": topic,
        "url": url,
        "date": date,
        "split": split,
        "source": source,
    }


def _try_load_split(split_name: str, n: int | None, seed: int) -> tuple[Iterable[dict], str] | None:
    from datasets import load_dataset

    last_err: Exception | None = None
    for repo, cfg, kind, source in CANDIDATES:
        try:
            LOG.info("Trying %s%s split=%s ...", repo, f"/{cfg}" if cfg else "", split_name)
            ds = load_dataset(repo, cfg, split=split_name, trust_remote_code=True) if cfg else load_dataset(repo, split=split_name, trust_remote_code=True)
            LOG.info("  loaded %d rows from %s", len(ds), repo)
            if n is not None and n < len(ds):
                ds = ds.shuffle(seed=seed).select(range(n * 2))

            def _gen() -> Iterable[dict]:
                kept = 0
                for row in ds:
                    norm = _normalize(row, kind, split_name, source)
                    if norm is None:
                        continue
                    yield norm
                    kept += 1
                    if n is not None and kept >= n:
                        return
            return _gen(), source
        except Exception as e:
            LOG.warning("  failed %s: %s", repo, type(e).__name__)
            last_err = e

    if last_err is not None:
        raise last_err
    raise RuntimeError("No dataset candidate succeeded.")


def main() -> None:
    p = argparse.ArgumentParser(description="Download MLSUM-TR (with fallbacks).")
    p.add_argument("--out-dir", default="data/raw/mlsum_tr")
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # XLSum uses 'validation' but mlsum uses 'validation' too — same names. Good.
    for split, n in (("train", args.n_train), ("validation", args.n_val), ("test", args.n_test)):
        target = out_dir / f"{split}.jsonl"
        gen, source = _try_load_split(split, n, args.seed)
        n_written = write_jsonl(target, gen)
        LOG.info("Wrote %d rows from %s -> %s", n_written, source, target)


if __name__ == "__main__":
    main()
