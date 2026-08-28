"""Build the dirty -> clean artifact-impact table (report Section VI).

Takes two aggregate JSONs produced by src.eval.run_eval_v2 (one scored on the
archived predictions, one on the sentinel-stripped predictions) and reports the
delta per system per metric, with the paired-bootstrap CI of the difference
where per-example scores are available.

Usage::

    python scripts/artifact_impact.py \
        --dirty outputs/results/v2/main_eval_DIRTY.json \
        --clean outputs/results/v2/main_eval_CLEAN.json \
        --per-example-dirty outputs/results/v2/per_example_dirty \
        --per-example-clean outputs/results/v2/per_example_clean \
        --out-json outputs/results/v2/artifact_impact.json \
        --out-md   outputs/results/v2/artifact_impact.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run as `python scripts/artifact_impact.py`, sys.path[0] is scripts/, so
# `from src.eval.stats import ...` fails and every paired CI silently degrades
# to "-". Put the repo root on the path first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS = [
    ("standard_rouge1_f", "R1 std F1"),
    ("standard_rouge2_f", "R2 std F1"),
    ("standard_rougeL_f", "RL std F1"),
    ("stem_rouge1_f", "R1 stem F1"),
    ("standard_rouge1_p", "R1 std P"),
    ("standard_rouge1_r", "R1 std R"),
    ("bs_f1", "BERTScore F1"),
    ("err_extract", "extract"),
    ("err_len_ratio", "len ratio"),
    ("err_halluc_num", "halluc#"),
]


def _load(p): return json.load(open(p, encoding="utf-8"))


def _per_ex(d, label):
    if not d:
        return None
    p = Path(d) / f"{label}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dirty vs clean artifact impact table.")
    ap.add_argument("--dirty", required=True)
    ap.add_argument("--clean", required=True)
    ap.add_argument("--per-example-dirty", default=None)
    ap.add_argument("--per-example-clean", default=None)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    D, C = _load(args.dirty), _load(args.clean)
    labels = [l for l in C if l in D]
    out, md = {}, []

    md.append("# Artifact impact: sentinel-contaminated vs cleaned predictions\n")
    md.append("Delta = clean - dirty. Positive means the published (dirty) number "
              "UNDERSTATED the system.\n")
    md.append("| System | Metric | dirty | clean | delta | paired 95% CI of delta |")
    md.append("| --- | --- | ---: | ---: | ---: | --- |")

    for label in labels:
        out[label] = {}
        pd_ = _per_ex(args.per_example_dirty, label)
        pc = _per_ex(args.per_example_clean, label)
        for key, pretty in METRICS:
            if key not in D[label] or key not in C[label]:
                continue
            dv = D[label][key]["mean"]
            cv = C[label][key]["mean"]
            row = {"dirty": dv, "clean": cv, "delta": cv - dv}
            ci_txt = "-"
            if pd_ and pc and len(pd_) == len(pc):
                try:
                    from src.eval.stats import paired_diff_ci
                    ci = paired_diff_ci([r[key] for r in pc], [r[key] for r in pd_])
                    row["delta_ci_low"], row["delta_ci_high"] = ci.low, ci.high
                    ci_txt = f"[{ci.low:+.4f}, {ci.high:+.4f}]"
                except Exception as e:  # pragma: no cover
                    row["ci_error"] = str(e)
                    print(f"  !! CI failed for {label}/{key}: {e}", file=sys.stderr)
            out[label][key] = row
            md.append(f"| {label} | {pretty} | {dv:.4f} | {cv:.4f} | {cv-dv:+.4f} | {ci_txt} |")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote {args.out_json}")
    if args.out_md:
        Path(args.out_md).write_text("\n".join(md), encoding="utf-8")
        print(f"Wrote {args.out_md}")
    print("\n".join(md[3:]))


if __name__ == "__main__":
    main()
