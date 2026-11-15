"""Level-2 validation on the real universe: flag rates per test and group, and detection of
anomalies INJECTED into real series. No parameter is retuned here (0.10, +30, 0.05 stay as they
are); there are no labelled real fraud cases, so a flag rate is an upper bound on the
false-positive rate, never a measured one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import universe as UN
from context import analyse

RETUNED = False          # parameters 0.10 and +30 are NOT retuned on these data
SEED = 2026
SCEN = {"smoothing s=0.3": ("smooth", 0.3), "smoothing s=0.6": ("smooth", 0.6),
        "losses shifted above zero": ("shift", 0.7)}
RELEVANT = {"smooth": "autocorrelation", "shift": "near_zero"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))


def _rate(d, test):
    ok = d[f"status_{test}"] == "ok" if f"status_{test}" in d else pd.Series(False, index=d.index)
    n, k = int(ok.sum()), int((d[f"flag_{test}"] & ok).sum()) if f"flag_{test}" in d else 0
    lo, hi = wilson(k, n)
    return dict(n_tested=n, n_flagged=k, flag_pct=100 * k / n if n else np.nan, ci95_low=lo, ci95_high=hi)


def flag_rates(F, source):
    """Flag rate per test for: all funds, each group, each pricing type, dead vs live funds."""
    F = F.copy()
    F["status_group"] = np.where(F.is_dead, "dead/merged", "live")
    F["group"] = F.asset_class + " x " + F.region
    parts = [("all funds", "all", F)]
    for lev, col in (("group", "group"), ("pricing", "pricing"), ("dead vs live", "status_group")):
        parts += [(lev, str(k), d) for k, d in F.groupby(col)]
    rows = []
    for lev, key, d in parts:
        for t in UN.TESTS:
            rows.append(dict(source=source, level=lev, key=key, test=t, n_funds=len(d), **_rate(d, t)))
        anyf = d[[f"flag_{t}" for t in UN.TESTS]].any(axis=1) if len(d) else pd.Series(dtype=bool)
        lo, hi = wilson(int(anyf.sum()), len(d))
        rows.append(dict(source=source, level=lev, key=key, test="any of the four", n_funds=len(d), n_tested=len(d),
                         n_flagged=int(anyf.sum()), flag_pct=100 * anyf.mean() if len(d) else np.nan, ci95_low=lo, ci95_high=hi))
    return pd.DataFrame(rows)


def reading_status_counts(F, source):
    rows = []
    for k in UN.READING_KEYS:
        vc = F[f"reading_{k}"].value_counts()
        for lvl, n in vc.items():
            rows.append(dict(source=source, reading=k, status=lvl, n_funds=int(n), pct=100 * n / len(F)))
    return pd.DataFrame(rows)


def survivorship_warning(U, F):
    n_dead = int(F.is_dead.sum())
    if n_dead == 0:
        return ("No dead or merged fund in this universe: only funds that still exist are observed, so past "
                "failures are missing and flag rates are likely understated for the problems that kill funds.")
    return (f"{n_dead} dead/merged fund(s) of {len(F)} are included. A universe with a few dead funds still "
            "under-represents all closures; read flag rates as indicative.")


def _smooth(r, s):
    out = r.copy().astype(float)
    for i in range(1, len(r)):
        out.iloc[i] = (1 - s) * r.iloc[i] + s * out.iloc[i - 1]
    return out


def _shift(r, frac, rng):
    x = r.copy()
    m = (x < 0) & (x > -0.01) & (rng.random(len(x)) < frac)
    x[m] = rng.uniform(0.0, 0.005, int(m.sum()))
    return x


def injection(U, max_funds=60, seed=SEED):
    """Inject graded anomalies into REAL series (seeded), run the same engine against the same
    reference, and report how often the relevant test flags. Same funds unmodified = baseline."""
    ids = [f for f in sorted(U.returns) if U.profiles[f].scoring_supported and len(U.returns[f]) >= 60]
    rng = np.random.default_rng(seed)
    if len(ids) > max_funds:
        ids = sorted(rng.choice(ids, max_funds, replace=False).tolist())
    rows = []
    for name, (kind, par) in [("unmodified (baseline)", (None, None))] + list(SCEN.items()):
        res = []
        for i, fid in enumerate(ids):
            s = U.returns[fid]
            if kind == "smooth":
                s = _smooth(s, par)
            elif kind == "shift":
                s = _shift(s, par, np.random.default_rng(seed + i))
            a = analyse(s.rename(fid), U.profiles[fid], U.factors, U.reference)
            if a.score is None:
                continue
            t = a.engine.all_tests()
            res.append(dict(fid=fid, score=a.score["score"],
                            **{k: bool(t[k]["flag"] and t[k]["status"] == "ok") for k in UN.TESTS}))
        d = pd.DataFrame(res)
        tgt = RELEVANT.get(kind)
        rel = d[tgt] if tgt else d[list(UN.TESTS)].any(axis=1)
        k, n = int(rel.sum()), len(d)
        lo, hi = wilson(k, n)
        rows.append(dict(scenario=name, n_funds=n, target_test=tgt or "any of the four", n_flagged=k,
                         detection_pct=100 * k / n if n else np.nan, ci95_low=lo, ci95_high=hi,
                         median_score=float(d.score.median()), pct_score_below_50=100 * float((d.score < 50).mean()),
                         pct_any_flag=100 * float(d[list(UN.TESTS)].any(axis=1).mean())))
    return pd.DataFrame(rows)


def run(U, F, source, with_injection=True):
    out = dict(flag_rates=flag_rates(F, source), reading_counts=reading_status_counts(F, source),
               survivorship=survivorship_warning(U, F), retuned=RETUNED)
    if with_injection:
        out["injection"] = injection(U)
        out["injection"].insert(0, "source", source)
    return out
