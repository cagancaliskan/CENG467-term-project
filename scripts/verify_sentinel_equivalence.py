"""Prove that post-hoc sentinel stripping == regenerating with the fixed decoder.

The revision re-scores ARCHIVED predictions after applying the sentinel regex,
rather than re-running inference for every system.  A reviewer is entitled to
ask whether that is legitimate.  It is, because the strip in
src/student/infer.py happens AFTER `tokenizer.batch_decode`, so it is a pure
string post-processing step on the very same decoded text.

Check 1 (deterministic, decisive): generate the same articles twice with the
same checkpoint, once with --keep-sentinels and once without, then verify

    regex_strip(dirty_i) == clean_i   for every i.

This cannot fail for reasons of GPU/library drift, because both arms come from
the same decode call pattern; it isolates exactly the claim being made.

Check 2 (reproducibility, informative): compare the freshly generated dirty
output against the archived v1 predictions.  Small disagreement here is beam
search sensitivity to library/GPU versions, NOT a flaw in Check 1.

Usage::

    python scripts/verify_sentinel_equivalence.py \
        --dirty    outputs/predictions/_verify/dirty.jsonl \
        --clean    outputs/predictions/_verify/clean.jsonl \
        --archived outputs/predictions/v1_raw/S_gpt.jsonl \
        --out-json outputs/results/v2/sentinel_equivalence.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SENTINEL_RE = re.compile(r"<extra_id_\d+>")


def _load(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["id"]] = r.get("prediction") or ""
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Verify sentinel-strip equivalence.")
    p.add_argument("--dirty", required=True, help="Fresh run WITH --keep-sentinels.")
    p.add_argument("--clean", required=True, help="Fresh run WITHOUT --keep-sentinels.")
    p.add_argument("--archived", default=None, help="Archived v1 prediction file (optional).")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    dirty, clean = _load(args.dirty), _load(args.clean)
    common = sorted(set(dirty) & set(clean))
    if not common:
        raise SystemExit("No shared ids between --dirty and --clean.")

    match, mismatches = 0, []
    n_with_sentinel = 0
    for k in common:
        if SENTINEL_RE.search(dirty[k]):
            n_with_sentinel += 1
        if SENTINEL_RE.sub("", dirty[k]).strip() == clean[k].strip():
            match += 1
        elif len(mismatches) < 5:
            mismatches.append({"id": k, "dirty": dirty[k][:200], "clean": clean[k][:200]})

    res = {
        "check1_postprocess_equals_inpipeline": {
            "n_compared": len(common),
            "n_exact_match": match,
            "match_rate": match / len(common),
            "n_dirty_with_sentinel": n_with_sentinel,
            "frac_dirty_with_sentinel": n_with_sentinel / len(common),
            "examples_of_mismatch": mismatches,
            "passed": match == len(common),
        }
    }

    if args.archived:
        arch = _load(args.archived)
        shared = sorted(set(arch) & set(dirty))
        same = sum(1 for k in shared if arch[k].strip() == dirty[k].strip())
        res["check2_archived_reproduces"] = {
            "n_compared": len(shared),
            "n_identical": same,
            "identity_rate": (same / len(shared)) if shared else 0.0,
            "note": ("Disagreement here reflects beam-search sensitivity to "
                     "library/GPU versions, not the validity of check 1."),
        }

    c1 = res["check1_postprocess_equals_inpipeline"]
    print(f"CHECK 1  post-processing == in-pipeline strip")
    print(f"  compared           : {c1['n_compared']}")
    print(f"  with >=1 sentinel  : {c1['n_dirty_with_sentinel']} ({100*c1['frac_dirty_with_sentinel']:.1f}%)")
    print(f"  exact match        : {c1['n_exact_match']} ({100*c1['match_rate']:.2f}%)")
    print(f"  PASSED             : {c1['passed']}")
    if "check2_archived_reproduces" in res:
        c2 = res["check2_archived_reproduces"]
        print(f"\nCHECK 2  archived v1 predictions reproduce")
        print(f"  compared  : {c2['n_compared']}")
        print(f"  identical : {c2['n_identical']} ({100*c2['identity_rate']:.2f}%)")

    if args.out_json:
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {outp}")

    raise SystemExit(0 if c1["passed"] else 1)


if __name__ == "__main__":
    main()
