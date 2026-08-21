"""Strip mT5 SentencePiece sentinel tokens from archived prediction files.

Applies EXACTLY the transformation that ``src/student/infer.py`` applies at
decode time::

    _SENTINEL_RE.sub("", text).strip()

so that re-scoring the cleaned files is mathematically identical to
regenerating them with the fixed inference code.  ``scripts/verify_sentinel_
equivalence.py`` proves this empirically.

The per-system sentinel incidence reported here is itself a generation-quality
finding: the rate at which each model falls back into mT5's span-corruption
mode.

Usage::

    python -m src.eval.clean_predictions \
        --in-dir  outputs/predictions/v1_raw \
        --out-dir outputs/predictions/clean \
        --stats-json outputs/results/v2/sentinel_incidence.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import get_logger

LOG = get_logger("eval.clean_predictions")

# Must stay byte-identical to src/student/infer.py::_SENTINEL_RE
SENTINEL_RE = re.compile(r"<extra_id_\d+>")


def strip_sentinels(text: str) -> str:
    """Identical to the post-decode step in src/student/infer.py."""
    return SENTINEL_RE.sub("", text or "").strip()


def _ntok(text: str) -> int:
    return len((text or "").split())


def clean_file(in_path: Path, out_path: Path) -> dict:
    rows = list(read_jsonl(in_path))
    n = len(rows)
    n_hit = 0
    n_sent = 0
    n_empty_after = 0
    tok_before = 0
    tok_after = 0
    out_rows = []

    for r in rows:
        raw = r.get("prediction") or ""
        hits = SENTINEL_RE.findall(raw)
        cleaned = strip_sentinels(raw)
        if hits:
            n_hit += 1
            n_sent += len(hits)
        if not cleaned:
            n_empty_after += 1
        tok_before += _ntok(raw)
        tok_after += _ntok(cleaned)
        new = dict(r)
        new["prediction"] = cleaned
        new["prediction_raw"] = raw
        new["n_sentinels"] = len(hits)
        out_rows.append(new)

    write_jsonl(out_path, out_rows)

    return {
        "file": in_path.name,
        "n": n,
        "n_with_sentinel": n_hit,
        "frac_with_sentinel": (n_hit / n) if n else 0.0,
        "n_sentinels_total": n_sent,
        "mean_sentinels_per_affected": (n_sent / n_hit) if n_hit else 0.0,
        "n_empty_after_clean": n_empty_after,
        "mean_tokens_before": (tok_before / n) if n else 0.0,
        "mean_tokens_after": (tok_after / n) if n else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Strip mT5 sentinel tokens from prediction JSONLs.")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--glob", default="*.jsonl")
    p.add_argument("--stats-json", default=None)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pth for pth in in_dir.glob(args.glob) if not pth.name.endswith(".gen_config.json"))
    if not files:
        raise SystemExit(f"No files matched {in_dir}/{args.glob}")

    stats = []
    for f in files:
        st = clean_file(f, out_dir / f.name)
        stats.append(st)
        LOG.info("%-26s n=%5d  sentinel_rows=%5d (%5.1f%%)  total=%6d  empty_after=%3d  tok %.1f->%.1f",
                 st["file"], st["n"], st["n_with_sentinel"], 100 * st["frac_with_sentinel"],
                 st["n_sentinels_total"], st["n_empty_after_clean"],
                 st["mean_tokens_before"], st["mean_tokens_after"])

    if args.stats_json:
        outp = Path(args.stats_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump({s["file"]: s for s in stats}, fh, indent=2, ensure_ascii=False)
        LOG.info("Wrote sentinel incidence -> %s", outp)


if __name__ == "__main__":
    main()
