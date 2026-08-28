"""Consolidate the per-article teacher cache into one JSONL per (teacher, variant).

Two problems this fixes.

1. **The paper's Sec. VII-C claims the synthetic caches are publicly available
   at the GitHub URL. They are not** -- `git ls-files data/synthetic` returns
   nothing. The cache lives only in Google Drive as ~22k individual .json files.
   Consolidating each (teacher, prompt) pair into a single JSONL makes it small
   enough to actually commit, so the claim becomes true.

2. Copying 22k tiny files out of Drive into Colab takes tens of minutes; copying
   four JSONLs takes seconds.

The originals are left untouched. `prepare_synthetic.py` prefers the JSONL when
it exists and falls back to the per-article directory otherwise, so nothing
that already works stops working.

Usage::

    python scripts/consolidate_synthetic.py --root data/synthetic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate per-article teacher caches into JSONL.")
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--out-root", default=None, help="Defaults to --root.")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_root or args.root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"{root} does not exist")

    total = 0
    for teacher_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for variant_dir in sorted(p for p in teacher_dir.iterdir() if p.is_dir()):
            files = sorted(variant_dir.glob("*.json"))
            if not files:
                continue
            out = out_root / f"{teacher_dir.name}_{variant_dir.name}.jsonl"
            n, skipped = 0, 0
            with open(out, "w", encoding="utf-8") as fo:
                for p in files:
                    try:
                        rec = json.load(open(p, encoding="utf-8"))
                    except Exception:
                        skipped += 1
                        continue
                    summary = (rec.get("summary") or "").strip()
                    if not summary:
                        skipped += 1
                        continue
                    fo.write(json.dumps({"id": rec.get("id") or p.stem,
                                         "summary": summary,
                                         "teacher": teacher_dir.name,
                                         "prompt_variant": variant_dir.name},
                                        ensure_ascii=False) + "\n")
                    n += 1
            mb = out.stat().st_size / 1e6
            print(f"{teacher_dir.name}/{variant_dir.name}: {n} summaries "
                  f"({skipped} skipped) -> {out} [{mb:.1f} MB]")
            total += n
    print(f"\nTotal: {total} summaries consolidated under {out_root}")


if __name__ == "__main__":
    main()
