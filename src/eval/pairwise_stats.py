"""Paired significance and equivalence tests on the cleaned per-example scores.

R1: "no confidence intervals or statistical significance tests were reported."
R2: "the paper claims that the systems are equivalent, that teacher selection
has little effect."

"The difference is small" is not evidence of equivalence. TOST (Lakens 2017) is
the test that licenses an equivalence claim; the margin is pre-registered in
PREREGISTRATION.md at delta = 0.010 ROUGE-1, derived from the published v1 gaps
before any new run.

CPU-only: runs on the per-example scores run_eval_v2 already wrote.

Caveat printed with the output: these tests capture TEST-SET sampling
uncertainty only. Training-seed variance is a separate source, not estimated
here; that needs the multi-seed runs.
"""
from __future__ import annotations

import argparse, itertools, json
from pathlib import Path

from src.eval.stats import (bootstrap_ci_mean, holm_bonferroni, paired_diff_ci,
                            paired_permutation_test, tost_paired)

# Equivalence is claimed only for the two pairs the paper calls interchangeable.
# Testing every pair for equivalence would be fishing.
EQUIV_PAIRS = [("B3a", "B3b"), ("S-gpt", "S-claude")]


def load(d: Path, metric: str) -> dict[str, dict[str, float]]:
    out = {}
    for p in sorted(d.glob("*.jsonl")):
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        out[p.stem] = {r["id"]: r[metric] for r in rows if metric in r}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired tests + TOST on per-example scores.")
    ap.add_argument("--per-example-dir", required=True)
    ap.add_argument("--metric", default="standard_rouge1_f")
    ap.add_argument("--delta", type=float, default=0.010)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    data = load(Path(args.per_example_dir), args.metric)
    if not data:
        raise SystemExit(f"no per-example files with metric {args.metric!r}")
    labels = list(data)
    common = sorted(set.intersection(*[set(v) for v in data.values()]))
    vec = {s: [data[s][i] for i in common] for s in labels}
    print(f"metric={args.metric}  systems={len(labels)}  paired items={len(common)}\n")

    res = {"metric": args.metric, "n_items": len(common), "delta": args.delta,
           "per_system": {}, "pairwise": {}, "equivalence": {}}

    for s in labels:
        ci = bootstrap_ci_mean(vec[s], n_boot=args.n_boot)
        res["per_system"][s] = ci.to_dict()
        print(f"{s:<12}{ci.estimate:>9.4f}   [{ci.low:.4f}, {ci.high:.4f}]")
    print()

    pairs = list(itertools.combinations(labels, 2))
    raw = []
    for a, b in pairs:
        t = paired_permutation_test(vec[a], vec[b], n_perm=args.n_perm)
        d = paired_diff_ci(vec[a], vec[b], n_boot=args.n_boot)
        res["pairwise"][f"{a} vs {b}"] = {"mean_diff": t["mean_diff"],
                                          "diff_ci": [d.low, d.high], "p_raw": t["p_value"]}
        raw.append(t["p_value"])
    holm = holm_bonferroni(raw)
    for (a, b), padj, rej in zip(pairs, holm["p_adjusted"], holm["reject"]):
        k = f"{a} vs {b}"
        res["pairwise"][k]["p_holm"] = padj
        res["pairwise"][k]["significant"] = rej
        r = res["pairwise"][k]
        print(f"{k:<22}{r['mean_diff']:>+9.4f}   [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}]"
              f"{padj:>11.4f}   {'yes' if rej else 'no'}")
    print()

    print(f"Equivalence (TOST, pre-registered delta = {args.delta}):")
    for a, b in EQUIV_PAIRS:
        if a in vec and b in vec:
            t = tost_paired(vec[a], vec[b], delta=args.delta)
            res["equivalence"][f"{a} vs {b}"] = t
            print(f"  {a} vs {b}: diff={t['mean_diff']:+.4f}  90% CI "
                  f"[{t['ci_1_minus_2alpha'][0]:+.4f}, {t['ci_1_minus_2alpha'][1]:+.4f}]  "
                  f"p_TOST={t['p_tost']:.2g}\n      -> {t['verdict']}")
    print("\nNOTE: test-set sampling uncertainty only. Training-seed variance is a")
    print("      separate source and is NOT estimated here.")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
