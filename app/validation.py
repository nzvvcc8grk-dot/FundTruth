"""Input validation for FundTruth (contract v2).

Hard errors (ValueError) are raised for problems that would silently corrupt
a score: wrong units, non month-end dates, duplicated dates, non-finite values.
Soft problems (NaN, gaps, constant series) are returned as warnings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBS = 36                 # contract: minimum observations per test
MAX_ABS_RETURN = 0.50        # monthly return outside [-50%, +50%] -> error
PCT_MEDIAN_ABS = 0.20        # median |r| above 20% -> looks like percent
PCT_STD = 0.25               # monthly std above 25% -> looks like percent
FACTOR_COLS = ["Mkt-RF", "SMB", "HML"]
RF_COL = "RF"


def percent_to_decimal(df):
    """Explicit, single conversion percent -> decimal (e.g. Fama-French)."""
    return df / 100.0


def to_month_end(obj):
    """Move a DatetimeIndex to calendar month-end (normalised, no time)."""
    out = obj.copy()
    out.index = pd.DatetimeIndex(out.index).normalize() + pd.offsets.MonthEnd(0)
    return out


def _check_index(idx, label):
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{label}: index must be a DatetimeIndex (month end).")
    if idx.has_duplicates:
        raise ValueError(f"{label}: duplicated dates in the index.")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{label}: index must be sorted in increasing order.")
    if len(idx) and not bool(np.all(idx.is_month_end)):
        raise ValueError(
            f"{label}: dates must be month-end; use validation.to_month_end().")


def _check_units(x, label, what):
    """Raise ValueError when values look like percentages or are implausible."""
    if x.size == 0:
        return
    if np.nanmax(np.abs(x)) > MAX_ABS_RETURN:
        raise ValueError(
            f"{label}: {what} outside [-0.5, +0.5] (max |x|="
            f"{np.nanmax(np.abs(x)):.3f}). Probably percent instead of decimal "
            "(divide once, explicitly) or a data error. No score produced.")
    if np.nanmedian(np.abs(x)) > PCT_MEDIAN_ABS or np.nanstd(x) > PCT_STD:
        raise ValueError(
            f"{label}: {what} look like percentages (median |x| or std too "
            "large). Convert to decimal explicitly. No score produced.")


def valida_dati(returns, factors=None):
    """Validate returns (and optional factors).

    Returns (clean_returns, clean_factors, warnings). Raises ValueError on
    hard errors (see module docstring). Series shorter than MIN_OBS are NOT an
    error here: every test then returns status "insufficient_data".
    """
    warnings: list[str] = []
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series.")
    label = str(returns.name) if returns.name is not None else "returns"
    r = pd.to_numeric(returns, errors="raise").astype(float)
    _check_index(r.index, label)
    if np.isinf(r.to_numpy()).any():
        raise ValueError(f"{label}: infinite values in returns.")
    n_nan = int(r.isna().sum())
    if n_nan:
        warnings.append(f"{n_nan} missing value(s) dropped.")
        r = r.dropna()
    _check_units(r.to_numpy(), label, "monthly returns")
    if len(r) > 1:
        full = pd.date_range(r.index[0], r.index[-1], freq="ME")
        n_gap = len(full.difference(r.index))
        if n_gap:
            warnings.append(f"{n_gap} calendar month(s) missing inside the span.")
    if len(r) and float(r.std(ddof=0)) == 0.0:
        warnings.append("constant series: most tests are not applicable.")

    f = None
    if factors is not None:
        if not isinstance(factors, pd.DataFrame):
            raise ValueError("factors must be a pandas DataFrame.")
        miss = [c for c in FACTOR_COLS if c not in factors.columns]
        if miss:
            raise ValueError(f"factors: missing column(s) {miss}.")
        _check_index(factors.index, "factors")
        cols = FACTOR_COLS + ([RF_COL] if RF_COL in factors.columns else [])
        f = factors[cols].astype(float).dropna()
        if len(f):
            arr = f["Mkt-RF"].to_numpy()
            if np.max(np.abs(f.to_numpy())) > MAX_ABS_RETURN or np.std(arr) > PCT_STD:
                raise ValueError(
                    "factors: values look like percent (Fama-French files are "
                    "in percent). Apply validation.percent_to_decimal() once.")
        if RF_COL not in f.columns:
            warnings.append("no RF column: regression uses raw (not excess) returns.")
    return r, f, warnings
