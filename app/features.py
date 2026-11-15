"""Per-fund statistics shared by the tests and by the reference builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from validation import FACTOR_COLS, RF_COL

FEATURE_COLS = ["vol", "skew", "ac1", "ac2", "ac3", "lb_q",
                "alpha", "alpha_t", "r2", "te"]


def autocorrelations(r, nlags=3):
    """Sample autocorrelations rho_1..rho_nlags (biased estimator, as acf)."""
    x = np.asarray(r, dtype=float)
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 0.0:
        return np.zeros(nlags)
    return np.array([np.dot(x[k:], x[:-k]) / den for k in range(1, nlags + 1)])


def ljung_box_q(r, nlags=3):
    """Ljung-Box Q statistic and asymptotic chi2 p-value."""
    n = len(r)
    rho = autocorrelations(r, nlags)
    q = n * (n + 2) * float(np.sum(rho ** 2 / (n - np.arange(1, nlags + 1))))
    return q, float(stats.chi2.sf(q, nlags))


def regress_factors(r, factors):
    """OLS of excess returns on Mkt-RF, SMB, HML (statsmodels).

    Returns dict(alpha, alpha_t, r2, te, n): alpha and te annualised (x12,
    x sqrt(12)), nan values when the regression is not defined.
    """
    idx = r.index.intersection(factors.index)
    out = dict(alpha=np.nan, alpha_t=np.nan, r2=np.nan, te=np.nan, n=len(idx))
    if len(idx) < 8:
        return out
    y = r.loc[idx] - (factors.loc[idx, RF_COL] if RF_COL in factors else 0.0)
    X = sm.add_constant(factors.loc[idx, FACTOR_COLS])
    if float(np.var(y.to_numpy())) == 0.0:
        return out
    res = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
    resid_sd = float(np.sqrt(res.ssr / res.df_resid))
    out["alpha"] = float(res.params[0]) * 12.0
    out["r2"] = float(res.rsquared)
    out["te"] = resid_sd * np.sqrt(12.0)
    # exact replication -> zero residual: t-stat undefined, set to 0 (no skill)
    out["alpha_t"] = float(res.tvalues[0]) if resid_sd > 1e-10 else 0.0
    return out


def autocorr_basis(r, factors=None):
    """Series used for the autocorrelation statistics and its label.

    Returns in EXCESS of the risk-free rate when factors with an RF column
    cover every month of r (a slowly varying risk-free rate or carry is
    persistent but is not smoothing); raw returns otherwise.
    """
    if factors is not None and RF_COL in factors and len(r) \
            and r.index.isin(factors.index).all():
        return r - factors[RF_COL].reindex(r.index), "excess over RF"
    return r, "raw returns"


def compute_features(r, factors=None):
    """All statistics used by the tests, as a flat dict (floats)."""
    x = r.to_numpy(dtype=float)
    xa = autocorr_basis(r, factors)[0].to_numpy(dtype=float)
    rho = autocorrelations(xa, 3)
    q, _ = ljung_box_q(xa, 3)
    feats = dict(
        vol=float(np.std(x, ddof=1) * np.sqrt(12.0)) if len(x) > 1 else np.nan,
        skew=float(stats.skew(x, bias=False)) if len(x) > 2 and np.std(x) > 0 else np.nan,
        ac1=float(rho[0]), ac2=float(rho[1]), ac3=float(rho[2]),
        # direction-aware: only positive (smoothing-type) autocorrelation counts
        lb_q=float(q) if rho.sum() > 0 else 0.0,
        alpha=np.nan, alpha_t=np.nan, r2=np.nan, te=np.nan)
    if factors is not None:
        reg = regress_factors(r, factors)
        feats.update({k: reg[k] for k in ("alpha", "alpha_t", "r2", "te")})
    return feats
