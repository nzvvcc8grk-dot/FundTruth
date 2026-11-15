"""Synthetic data with KNOWN anomalies (all numbers are generator assumptions,
not market data). Reproducible through explicit seeds.

(a) smoothed Madoff-style series   (b) losses shifted just above zero
(c) closet indexer                 (d) honest control
plus synthetic_reference(): the frozen placeholder peer set (version v0).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from features import compute_features

END = "2025-12-31"
# (mean, sd) of the random exposures / annual alpha / monthly residual sd range
CATEGORY_PARAMS = {
    "equity": dict(beta=(1.00, 0.08), smb=(0.10, 0.20), hml=(0.00, 0.20),
                   alpha=(-0.010, 0.020), eps=(0.008, 0.022)),
    "bond": dict(beta=(0.10, 0.05), smb=(0.00, 0.05), hml=(0.00, 0.05),
                 alpha=(0.015, 0.010), eps=(0.003, 0.008)),
    "multi-asset": dict(beta=(0.55, 0.08), smb=(0.05, 0.10), hml=(0.00, 0.10),
                        alpha=(-0.005, 0.015), eps=(0.004, 0.012)),
}


def make_factors(n, seed, end=END):
    """Synthetic monthly factors (decimal): Mkt-RF, SMB, HML, RF."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=end, periods=n, freq="ME")
    rf = np.clip(0.002 + 0.0007 * rng.standard_normal(n), 0.0, None)
    return pd.DataFrame({
        "Mkt-RF": rng.normal(0.007, 0.045, n),
        "SMB": rng.normal(0.002, 0.025, n),
        "HML": rng.normal(0.002, 0.027, n),
        "RF": rf}, index=idx)


def honest_fund(category, seed, n=96, name=None):
    """Honest fund: factor model + iid Gaussian residual. Returns (series, factors)."""
    p = CATEGORY_PARAMS[category]
    rng = np.random.default_rng(seed)
    f = make_factors(n, seed + 10_000_000)
    b = rng.normal(*p["beta"])
    s = rng.normal(*p["smb"])
    h = rng.normal(*p["hml"])
    a = rng.normal(*p["alpha"])
    sd = rng.uniform(*p["eps"])
    r = (f["RF"] + a / 12 + b * f["Mkt-RF"] + s * f["SMB"] + h * f["HML"]
         + rng.normal(0, sd, n))
    return r.rename(name or f"HONEST_{category}_{seed}"), f


def smoothed_fund(seed, n=96, theta=(0.40, 0.30, 0.30)):
    """(a) Reported return = weighted average of current and 2 lagged true
    returns (Getmansky-Lo-Makarov smoothing): autocorrelated, lower vol."""
    true, f = honest_fund("multi-asset", seed, n + 2)
    th = np.asarray(theta, dtype=float) / np.sum(theta)
    rep = th[0] * true + th[1] * true.shift(1) + th[2] * true.shift(2)
    return rep.iloc[2:].rename(f"SYN_A_smoothed_{seed}"), f.iloc[2:]


def near_zero_fund(seed, n=96):
    """(b) Losses in [-0.5%, 0) are reported as gains uniform in [0, 0.5%)."""
    r, f = honest_fund("multi-asset", seed, n)
    rng = np.random.default_rng(seed + 20_000_000)
    mask = (r >= -0.005) & (r < 0.0)
    r = r.copy()
    r[mask] = rng.uniform(0.0, 0.005, int(mask.sum()))
    return r.rename(f"SYN_B_nearzero_{seed}"), f


def closet_fund(seed, n=96, noise=0.001, fee_monthly=0.00008):
    """(c) Market (Mkt-RF + RF) return plus tiny noise, minus a tiny fee."""
    f = make_factors(n, seed + 10_000_000)
    rng = np.random.default_rng(seed + 30_000_000)
    r = f["RF"] + f["Mkt-RF"] - fee_monthly + rng.normal(0, noise, n)
    return r.rename(f"SYN_C_closet_{seed}"), f


def control_fund(seed, n=96):
    """(d) Honest equity control."""
    r, f = honest_fund("equity", seed, n)
    return r.rename(f"SYN_D_control_{seed}"), f


def synthetic_set(seed, n=96):
    """The four labelled series of the validation rule: {label: (series, factors)}."""
    return {"a_smoothed": smoothed_fund(seed, n), "b_nearzero": near_zero_fund(seed, n),
            "c_closet": closet_fund(seed, n), "d_control": control_fund(seed, n)}


@lru_cache(maxsize=4)
def _reference(n_per_category, n_obs, seed):
    rows = []
    for cat in CATEGORY_PARAMS:
        for i in range(n_per_category):
            r, f = honest_fund(cat, 100_000_000 + seed * 10_000 + i, n_obs)
            feats = compute_features(r, f)
            feats.update(category=cat, style="active")
            rows.append(feats)
    df = pd.DataFrame(rows)
    df.index = [f"REF{i:04d}" for i in range(len(df))]
    return df


def synthetic_reference(n_per_category=1000, n_obs=96, seed=2026):
    """Frozen placeholder reference (version 'synthetic-v0'). Replace with a
    reference built from real funds (reference_from_returns in data.py)."""
    return _reference(n_per_category, n_obs, seed).copy()
