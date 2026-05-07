"""Inspect teacher pilot outputs: length stats, error flags, side-by-side samples.

Designed for Day 3 manual prompt validation. Reads two cache directories
(typically <teacher>/<concise> and <teacher>/<detailed>) and prints:
  - per-prompt summary length distribution
  - aggregated error_analysis flags (repetition, hallucinated numbers, etc.)
  - K random side-by-side samples with article snippet, both summaries, flags

Usage:
    python -m scripts.inspect_outputs \
        --pilot data/raw/mlsum_tr/pilot_100.jsonl \
        --concise data/synthetic/openai/concise \
        --detailed data/synthetic/openai/detailed \
        --k 5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from src.eval.error_analysis import aggregate, detect
from src.utils.io import read_jsonl
from src.utils.logging import get_logger

LOG = get_logger("inspect")


def _load_cache(cache_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not cache_dir.exists():
        LOG.warning("Cache dir missing: %s", cache_dir)
        return out
    for p in cache_dir.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("summary"):
                out[rec.get("id", p.stem)] = rec["summary"]
        except Exception as e:
            LOG.warning("  skipping %s: %s", p, e)
    return out


def _stats(name: str, summaries: list[str]) -> None:
    if not summaries:
        print(f"  {name}: no outputs")
        return
    chars = [len(s) for s in summaries]
    words = [len(s.split()) for s in summaries]
    print(f"  {name}: n={len(summaries)}  chars min/med/max={min(chars)}/{int(statistics.median(chars))}/{max(chars)}  "
          f"words min/med/max={min(words)}/{int(statistics.median(words))}/{max(words)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect teacher pilot outputs.")
    p.add_argument("--pilot", required=True, help="Pilot JSONL with article + reference.")
    p.add_argument("--concise", required=True, help="Cache dir for concise prompt.")
    p.add_argument("--detailed", default=None, help="Cache dir for detailed prompt (optional).")
    p.add_argument("--k", type=int, default=5, help="Random side-by-side samples to print.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-snippet-chars", type=int, default=400)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pilot = list(read_jsonl(args.pilot))
    by_id = {r["id"]: r for r in pilot}

    concise = _load_cache(Path(args.concise))
    detailed = _load_cache(Path(args.detailed)) if args.detailed else {}

    print("== length stats ==")
    _stats("concise", list(concise.values()))
    if detailed:
        _stats("detailed", list(detailed.values()))

    def _flags(cache: dict[str, str]) -> dict:
        flags = []
        for aid, summary in cache.items():
            row = by_id.get(aid)
            if not row:
                continue
            flags.append(detect(article=row["article"], prediction=summary, reference=row.get("reference")))
        return aggregate(flags)

    print("\n== error_analysis aggregates ==")
    print("  concise :", _flags(concise))
    if detailed:
        print("  detailed:", _flags(detailed))

    print(f"\n== {args.k} side-by-side samples ==")
    common_ids = list(set(concise) & (set(detailed) if detailed else set(concise)))
    rng.shuffle(common_ids)
    for aid in common_ids[: args.k]:
        row = by_id.get(aid)
        if not row:
            continue
        print("\n" + "─" * 80)
        print(f"id: {aid}  topic: {row.get('topic')}")
        print("ARTICLE (snippet):")
        print("  ", (row["article"][: args.max_snippet_chars] + "…") if len(row["article"]) > args.max_snippet_chars else row["article"])
        if row.get("reference"):
            print("REFERENCE:")
            print("  ", row["reference"])
        print("CONCISE:")
        print("  ", concise.get(aid, "<missing>"))
        if detailed:
            print("DETAILED:")
            print("  ", detailed.get(aid, "<missing>"))


if __name__ == "__main__":
    main()
