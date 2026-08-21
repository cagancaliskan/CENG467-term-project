"""Gate check A: is the `article` field in the archived predictions the FULL
source article, or the truncated model input?

Every faithfulness metric in the revision is computed against this field.  If
it is truncated, entity/number grounding silently re-creates the 600-character
judging bug at scale -- and it biases *against* the verbose teachers, i.e.
against exactly the systems the hallucination claim compares to.

Run this BEFORE any faithfulness work.

Usage::

    python scripts/audit_articles.py --in-dir outputs/predictions/v1_raw
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def audit(path: Path) -> dict:
    lens, n = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lens.append(len((r.get("article") or "")))
            n += 1
    lens.sort()
    top = Counter(lens).most_common(3)
    # A hard truncation cap shows up as a large spike at one exact length.
    spike_len, spike_n = (top[0] if top else (0, 0))
    return {
        "file": path.name,
        "n": n,
        "min": lens[0] if lens else 0,
        "p25": _pct(lens, 0.25),
        "median": _pct(lens, 0.50),
        "p75": _pct(lens, 0.75),
        "p95": _pct(lens, 0.95),
        "max": lens[-1] if lens else 0,
        "mean": (sum(lens) / n) if n else 0.0,
        "modal_length": spike_len,
        "modal_count": spike_n,
        "modal_frac": (spike_n / n) if n else 0.0,
        "n_at_3000": sum(1 for x in lens if x == 3000),
        "n_ge_3000": sum(1 for x in lens if x >= 3000),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit the `article` field length distribution.")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--glob", default="*.jsonl")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    files = sorted(Path(args.in_dir).glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {args.in_dir}/{args.glob}")

    rows = [audit(f) for f in files]

    hdr = f"{'file':<26} {'n':>5} {'min':>6} {'med':>6} {'p95':>7} {'max':>7} {'mode':>7} {'mode%':>6} {'>=3000':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['file']:<26} {r['n']:>5} {r['min']:>6} {r['median']:>6} {r['p95']:>7} "
              f"{r['max']:>7} {r['modal_length']:>7} {100*r['modal_frac']:>5.1f}% {r['n_ge_3000']:>7}")

    print()
    verdict_ok = True
    for r in rows:
        if r["modal_frac"] > 0.05 and r["modal_length"] > 0:
            print(f"  !! {r['file']}: {100*r['modal_frac']:.1f}% of articles are EXACTLY "
                  f"{r['modal_length']} chars -> looks like a hard truncation cap.")
            verdict_ok = False
    if verdict_ok:
        print("  OK: no hard truncation spike detected. `article` looks like full source text.")
        print("      -> faithfulness metrics may be computed against this field.")
    else:
        print("  ACTION: recover full articles from MLSUM-TR by id before running WP-D.")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            json.dump({r["file"]: r for r in rows}, fh, indent=2)
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
