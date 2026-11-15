"""Diagnostics shown NEXT TO the score. None of them feeds the score.

All functions take plain pandas objects (decimal monthly returns, month-end
index) and depend only on the fund's own inputs, except peer percentiles, which
compare against a peer table that is passed in explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from validation import FACTOR_COLS, RF_COL

# PROVISIONAL (not calibrated): a fund is index-like against its declared benchmark
# when R2 >= CLOSET_R2-style 0.95 and annual tracking error <= 2%.
INDEX_R2 = 0.95
INDEX_TE = 0.02
SUBPERIODS = {"2020": ("2020-01-01", "2020-12-31"), "2022-2023": ("2022-01-01", "2023-12-31")}
MIN_SUB = 12


def excess(r, factors=None):
    """Returns in excess of the risk-free rate when it covers every month, else raw."""
    if factors is not None and RF_COL in factors and len(r) and r.index.isin(factors.index).all():
        return r - factors[RF_COL].reindex(r.index)
    return r


def rolling_autocorr(x, window=36):
    """Rolling lag-1 autocorrelation over `window` months (NaN until the window fills)."""
    x = pd.Series(x, dtype=float)
    out = x.rolling(window).apply(lambda w: pd.Series(w).autocorr(lag=1), raw=True)
    return out.dropna()


def desmoothed_vol(r, factors=None):
    """Reported vs de-smoothed annualised volatility.

    Method (simple AR(1) unsmoothing, Geltner/Okunev-White style): with a = lag-1
    autocorrelation of excess returns (floored at 0), true_t = (r_t - a r_{t-1}) / (1 - a).
    Limits: assumes the whole autocorrelation is smoothing (true returns are iid),
    so it overstates the correction for funds with genuinely persistent returns;
    the full Getmansky-Lo-Makarov MA(2) maximum-likelihood fit is NOT implemented.
    A ratio well above 1 is a reading, not a measurement of hidden risk.
    """
    x = excess(r, factors).to_numpy(dtype=float)
    if len(x) < 24 or np.std(x) < 1e-12:
        return dict(reported=np.nan, desmoothed=np.nan, ratio=np.nan, a=np.nan)
    d = x - x.mean()
    a = float(np.clip(np.dot(d[1:], d[:-1]) / np.dot(d, d), 0.0, 0.95))
    u = (x[1:] - a * x[:-1]) / (1.0 - a)
    rep = float(np.std(x, ddof=1) * np.sqrt(12))
    des = float(np.std(u, ddof=1) * np.sqrt(12))
    return dict(reported=rep, desmoothed=des, ratio=des / rep if rep > 0 else np.nan, a=a)


def drawdown(r):
    """Drawdown series (<= 0) of the cumulative return path."""
    w = (1.0 + pd.Series(r, dtype=float)).cumprod()
    return w / w.cummax() - 1.0


def risk_shape(r):
    x = np.asarray(r, dtype=float)
    return dict(max_drawdown=float(drawdown(r).min()),
                skew=float(stats.skew(x, bias=False)) if len(x) > 2 and x.std() > 0 else np.nan,
                excess_kurtosis=float(stats.kurtosis(x, bias=False)) if len(x) > 3 and x.std() > 0 else np.nan)


def capture_ratios(r, m):
    """Up/down capture vs a market return series: mean fund return over mean market
    return in months when the market was up (down). Arithmetic means; NaN if no such months."""
    d = pd.DataFrame({"r": r, "m": m}).dropna()
    up, dn = d[d.m > 0], d[d.m < 0]
    return dict(up=float(up.r.mean() / up.m.mean()) if len(up) >= 6 else np.nan,
                down=float(dn.r.mean() / dn.m.mean()) if len(dn) >= 6 else np.nan,
                n_up=len(up), n_down=len(dn))


def factor_loadings(r, factors):
    """OLS of excess returns on the three factors: betas with 95% intervals, R2,
    annualised alpha with t-stat, annualised tracking error (residual vol). None if undefined."""
    if factors is None:
        return None
    idx = r.index.intersection(factors.index)
    if len(idx) < 24:
        return None
    y = r.loc[idx] - (factors.loc[idx, RF_COL] if RF_COL in factors else 0.0)
    if float(np.var(y.to_numpy())) < 1e-18:
        return None
    X = sm.add_constant(factors.loc[idx, FACTOR_COLS])
    res = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
    ci = res.conf_int(0.05)
    out = dict(n=len(idx), r2=float(res.rsquared), alpha=float(res.params[0] * 12),
               alpha_t=float(res.tvalues[0]),
               te=float(np.sqrt(res.ssr / res.df_resid) * np.sqrt(12)))
    for i, c in enumerate(FACTOR_COLS, start=1):
        out[f"beta_{c}"], out[f"lo_{c}"], out[f"hi_{c}"] = (float(res.params[i]),
                                                          float(ci[i, 0]), float(ci[i, 1]))
    return out


def rolling_beta(r, factors, window=36):
    """Rolling market beta (OLS slope on Mkt-RF over `window` months)."""
    if factors is None or "Mkt-RF" not in factors:
        return pd.Series(dtype=float)
    d = pd.DataFrame({"y": r - (factors[RF_COL].reindex(r.index) if RF_COL in factors else 0.0),
                      "m": factors["Mkt-RF"].reindex(r.index)}).dropna()
    if len(d) < window:
        return pd.Series(dtype=float)
    cov = d.y.rolling(window).cov(d.m)
    return (cov / d.m.rolling(window).var()).dropna()


def alpha_subperiods(r, factors, periods=SUBPERIODS, min_obs=MIN_SUB):
    """Annualised alpha (factor model re-fitted in each sub-period). Sub-periods with
    fewer than `min_obs` months are reported as insufficient, with n."""
    out = {}
    for name, (a, b) in periods.items():
        rr = r.loc[a:b]
        fl = factor_loadings(rr, factors.loc[a:b]) if factors is not None and len(rr) >= min_obs else None
        # factor_loadings needs 24 months; for 12-23 months use a plain fit
        if fl is None and factors is not None and len(rr) >= min_obs:
            idx = rr.index.intersection(factors.index)
            if len(idx) >= min_obs:
                y = rr.loc[idx] - factors.loc[idx, RF_COL]
                X = sm.add_constant(factors.loc[idx, FACTOR_COLS])
                res = sm.OLS(y.to_numpy(), X.to_numpy()).fit()
                fl = dict(n=len(idx), alpha=float(res.params[0] * 12), alpha_t=float(res.tvalues[0]))
        out[name] = dict(n=len(rr), alpha=fl["alpha"] if fl else np.nan,
                         alpha_t=fl["alpha_t"] if fl else np.nan)
    return out


def benchmark_fit(r, bench):
    """R2, annualised tracking error, beta and months vs a DECLARED benchmark."""
    idx = r.index.intersection(bench.index)
    if len(idx) < 24:
        return dict(n=len(idx), r2=np.nan, te=np.nan, beta=np.nan)
    a, b = r.loc[idx].to_numpy(), bench.loc[idx].to_numpy()
    if np.std(b) < 1e-12:
        return dict(n=len(idx), r2=np.nan, te=np.nan, beta=np.nan)
    c = np.corrcoef(a, b)[0, 1]
    return dict(n=len(idx), r2=float(c * c), te=float(np.std(a - b, ddof=1) * np.sqrt(12)),
                beta=float(np.cov(a, b)[0, 1] / np.var(b, ddof=1)))


def sharpe(r, factors=None):
    x = excess(r, factors)
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan


def peer_percentile(value, peer_values):
    """Share (0-1) of peers with a value <= `value`; NaN when no peers or value."""
    p = np.asarray(peer_values, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0 or not np.isfinite(value):
        return np.nan
    return float((p <= value).mean())


def fund_stats(r, factors=None):
    """Flat per-fund statistics used as peer statistics."""
    fl = factor_loadings(r, factors) or {}
    x = excess(r, factors)
    return dict(vol=float(r.std(ddof=1) * np.sqrt(12)), sharpe=sharpe(r, factors),
                ac1=float(pd.Series(x).autocorr(lag=1)) if len(x) > 2 else np.nan,
                r2=fl.get("r2", np.nan), alpha=fl.get("alpha", np.nan), te=fl.get("te", np.nan),
                beta_mkt=fl.get("beta_Mkt-RF", np.nan), beta_smb=fl.get("beta_SMB", np.nan),
                beta_hml=fl.get("beta_HML", np.nan), max_drawdown=float(drawdown(r).min()))


def build_peer_stats(returns_by_fund, profiles, factors_by_region):
    """Per-fund statistics table (no prices) used for peer percentiles and medians."""
    from fund_profile import factor_set_for
    rows = []
    for fid, s in returns_by_fund.items():
        if len(s) < 36:
            continue
        p = profiles[fid]
        fac = factors_by_region.get(factor_set_for(p).key)
        d = fund_stats(s, fac)
        d.update(fund=fid, asset_class=p.asset_class, region=p.region,
                 expense_ratio=p.expense_ratio)
        rows.append(d)
    return pd.DataFrame(rows).set_index("fund") if rows else pd.DataFrame()
