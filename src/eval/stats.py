"""Statistical machinery for the camera-ready revision.

Addresses reviewer comments R1 ("only a single training run ... no confidence
intervals or statistical significance tests") and R2 ("the paper claims that
the systems are equivalent").

Two independent sources of uncertainty are reported separately:

1. Test-set sampling variance -- paired bootstrap over the fixed test articles.
   Applies to every system, including the API teachers, which have no seed.
2. Training seed variance -- Student t interval over the seeds (df = m-1).
   With m=3 seeds the multiplier is 4.30, which is honest; a bare standard
   deviation over 3 points looks misleadingly tight.

Significance tests run on PER-EXAMPLE scores that have already been averaged
over seeds, so a p-value is not conditional on one lucky training run.

Equivalence ("the two teachers are interchangeable") is tested with TOST
(Lakens 2017), not by failing to reject a difference. The margin delta is
pre-registered in PREREGISTRATION.md before the multi-seed runs.

References:
  Dror et al. (2018) The Hitchhiker's Guide to Testing Statistical
    Significance in NLP. ACL.
  Riezler & Maxwell (2005) On some pitfalls in automatic evaluation and
    significance testing for MT.
  Lakens (2017) Equivalence tests. Soc. Psychol. Personal. Sci.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from statistics import NormalDist
from typing import Sequence

import numpy as np

_N01 = NormalDist()


# --------------------------------------------------------------------------
# point estimates + confidence intervals
# --------------------------------------------------------------------------

@dataclass
class Interval:
    estimate: float
    low: float
    high: float
    method: str
    n: int

    def to_dict(self) -> dict:
        return asdict(self)

    def fmt(self, nd: int = 4) -> str:
        return f"{self.estimate:.{nd}f} [{self.low:.{nd}f}, {self.high:.{nd}f}]"


def _boot_means(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return x[idx].mean(axis=1)


def bootstrap_ci_mean(values: Sequence[float], n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 42) -> Interval:
    """BCa bootstrap CI for a mean. Falls back to percentile if BCa degenerates."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, "empty", 0)
    if n == 1:
        return Interval(float(x[0]), float(x[0]), float(x[0]), "n=1", 1)

    rng = np.random.default_rng(seed)
    theta = float(x.mean())
    boots = _boot_means(x, n_boot, rng)

    # bias-correction
    prop = float((boots < theta).mean())
    prop = min(max(prop, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    z0 = _N01.inv_cdf(prop)

    # acceleration via closed-form jackknife of the mean
    jack = (x.sum() - x) / (n - 1)
    d = jack.mean() - jack
    denom = 6.0 * (float((d ** 2).sum()) ** 1.5)
    a = float((d ** 3).sum()) / denom if denom > 0 else 0.0

    def _adj(p: float) -> float:
        z = _N01.inv_cdf(p)
        num = z0 + z
        den = 1.0 - a * num
        if den == 0:
            return p
        return _N01.cdf(z0 + num / den)

    lo_p, hi_p = _adj(alpha / 2), _adj(1.0 - alpha / 2)
    if not (0.0 < lo_p < hi_p < 1.0):  # BCa degenerate -> percentile
        lo_p, hi_p, method = alpha / 2, 1.0 - alpha / 2, "percentile"
    else:
        method = "BCa"

    lo, hi = np.quantile(boots, [lo_p, hi_p])
    return Interval(theta, float(lo), float(hi), f"bootstrap-{method}", n)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                   alpha: float = 0.05, seed: int = 42) -> Interval:
    """CI for mean(a) - mean(b) on PAIRED observations (same test articles)."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y):
        raise ValueError(f"paired inputs must match: {len(x)} vs {len(y)}")
    return bootstrap_ci_mean(x - y, n_boot=n_boot, alpha=alpha, seed=seed)


def seed_interval(values: Sequence[float], alpha: float = 0.05) -> Interval:
    """Student t interval across training seeds. With m=3, t(.975,2)=4.303."""
    from math import sqrt
    v = np.asarray(values, float)
    m = len(v)
    if m < 2:
        return Interval(float(v[0]) if m else 0.0, float("nan"), float("nan"), "seeds<2", m)
    mean = float(v.mean())
    se = float(v.std(ddof=1) / sqrt(m))
    try:
        from scipy import stats as _st
        t = float(_st.t.ppf(1 - alpha / 2, m - 1))
    except Exception:
        t = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(m - 1, 1.96)
    return Interval(mean, mean - t * se, mean + t * se, f"t-interval(df={m-1})", m)


# --------------------------------------------------------------------------
# significance tests
# --------------------------------------------------------------------------

def paired_permutation_test(a: Sequence[float], b: Sequence[float],
                            n_perm: int = 10000, seed: int = 42) -> dict:
    """Two-sided approximate randomization test on paired per-example scores.

    H0: the sign of each paired difference is exchangeable.
    p = (#{|mean(d*)| >= |mean(d)|} + 1) / (n_perm + 1)   [Riezler & Maxwell]
    """
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y):
        raise ValueError(f"paired inputs must match: {len(x)} vs {len(y)}")
    d = x - y
    n = len(d)
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null = np.abs((signs * d).mean(axis=1))
    p = (int((null >= obs - 1e-15).sum()) + 1) / (n_perm + 1)
    return {"mean_diff": float(d.mean()), "abs_mean_diff": obs,
            "p_value": float(p), "n": n, "n_perm": n_perm,
            "test": "paired approximate randomization (two-sided)"}


def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05) -> dict:
    """Holm step-down correction. Returns adjusted p-values and reject flags."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return {"p_raw": p.tolist(), "p_adjusted": adj.tolist(),
            "reject": (adj < alpha).tolist(), "alpha": alpha, "m": m}


def tost_paired(a: Sequence[float], b: Sequence[float], delta: float,
                alpha: float = 0.05) -> dict:
    """Two One-Sided Tests for equivalence of paired means within +/- delta.

    Equivalence is DECLARED only if both one-sided nulls are rejected, i.e.
    p_tost < alpha, equivalently the (1-2*alpha) CI lies inside (-delta, delta).
    Failing this returns 'no evidence of a difference, equivalence NOT shown'.
    """
    from math import sqrt
    x, y = np.asarray(a, float), np.asarray(b, float)
    if len(x) != len(y):
        raise ValueError(f"paired inputs must match: {len(x)} vs {len(y)}")
    d = x - y
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / sqrt(n))
    if se == 0:
        se = 1e-12
    df = n - 1
    try:
        from scipy import stats as _st
        cdf = lambda t: float(_st.t.cdf(t, df))
        tcrit = float(_st.t.ppf(1 - alpha, df))
    except Exception:
        cdf = _N01.cdf
        tcrit = _N01.inv_cdf(1 - alpha)

    t_lower = (mean + delta) / se     # H0: mean <= -delta
    t_upper = (mean - delta) / se     # H0: mean >=  delta
    p_lower = 1.0 - cdf(t_lower)
    p_upper = cdf(t_upper)
    p_tost = max(p_lower, p_upper)

    lo, hi = mean - tcrit * se, mean + tcrit * se     # (1-2alpha) CI = 90% at alpha=.05
    equivalent = bool(p_tost < alpha)
    return {
        "mean_diff": mean, "se": se, "n": n, "delta": delta, "alpha": alpha,
        "p_lower": float(p_lower), "p_upper": float(p_upper), "p_tost": float(p_tost),
        "ci_1_minus_2alpha": [lo, hi],
        "equivalent": equivalent,
        "verdict": ("equivalence supported: the difference lies within the "
                    f"pre-registered margin of +/-{delta}")
                   if equivalent else
                   ("equivalence NOT established; absence of a significant "
                    "difference is not evidence of equivalence"),
    }


# --------------------------------------------------------------------------
# binary outcomes (judge pass rates, hallucination flags, human labels)
# --------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return Interval(0.0, 0.0, 1.0, "wilson", 0)
    z = _N01.inv_cdf(1 - alpha / 2)
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / den
    return Interval(phat, max(0.0, centre - half), min(1.0, centre + half), "wilson", n)


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test on paired binary outcomes (same items, two systems)."""
    x, y = np.asarray(a, int), np.asarray(b, int)
    if len(x) != len(y):
        raise ValueError(f"paired inputs must match: {len(x)} vs {len(y)}")
    n01 = int(((x == 0) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0,
                "test": "exact McNemar"}
    try:
        from scipy.stats import binomtest
        p = float(binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import comb
        k = min(n01, n10)
        p = min(1.0, 2.0 * sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc))
    return {"n01": n01, "n10": n10, "n_discordant": n_disc,
            "p_value": p, "test": "exact McNemar"}


def min_detectable_difference(n_per_group: int, p_baseline: float = 0.10,
                              alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate MDE for two independent proportions -- report this whenever a
    small human-annotation sample fails to find a difference."""
    z_a = _N01.inv_cdf(1 - alpha / 2)
    z_b = _N01.inv_cdf(power)
    pbar = p_baseline
    return float((z_a + z_b) * ((2 * pbar * (1 - pbar) / n_per_group) ** 0.5))


def self_test() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.30, 0.10, 2000)
    b = a + rng.normal(0.0, 0.02, 2000)          # nearly identical system
    c = a + rng.normal(0.05, 0.02, 2000)         # clearly better system

    ci = bootstrap_ci_mean(a)
    assert ci.low < ci.estimate < ci.high, ci
    assert abs(ci.estimate - 0.30) < 0.02

    same = paired_permutation_test(a, b, n_perm=2000)
    diff = paired_permutation_test(a, c, n_perm=2000)
    assert same["p_value"] > 0.05, same
    assert diff["p_value"] < 0.01, diff

    eq = tost_paired(a, b, delta=0.010)
    ne = tost_paired(a, c, delta=0.010)
    assert eq["equivalent"] is True, eq
    assert ne["equivalent"] is False, ne

    h = holm_bonferroni([0.001, 0.04, 0.5])
    assert h["reject"] == [True, False, False], h

    w = wilson_ci(3, 40)
    assert 0.0 < w.low < 0.075 < w.high < 0.30, w

    s = seed_interval([0.261, 0.258, 0.264])
    assert s.low < s.estimate < s.high and s.method.startswith("t-interval"), s

    m = mcnemar_exact([1, 1, 0, 0, 1, 0], [1, 0, 0, 1, 1, 1])
    assert 0.0 <= m["p_value"] <= 1.0, m

    print("stats.py self-test PASSED")
    print(f"  bootstrap BCa CI on N(0.30,0.10), n=2000 : {ci.fmt()}")
    print(f"  TOST equivalent pair (delta=0.010)       : p_tost={eq['p_tost']:.3g} -> {eq['equivalent']}")
    print(f"  TOST different pair  (delta=0.010)       : p_tost={ne['p_tost']:.3g} -> {ne['equivalent']}")
    print(f"  seed t-interval over 3 seeds             : {s.fmt()}")
    print(f"  MDE at n=40/group, p=0.10                : {min_detectable_difference(40):.3f}")


if __name__ == "__main__":
    self_test()
