"""CSV connector (Bloomberg-style exports): long-format returns + metadata.

returns.csv : date, fund_id, value, kind     kind = "return" (monthly decimal) or "nav"
meta.csv    : fund_id [, region, currency, asset_class, pricing, style,
              benchmark_id, inception, expense_ratio, aum]   (all but fund_id optional)

Validation is explicit: hard problems reject the fund (reason in `rejected`),
soft problems become warnings. Percentages are rejected with a clear message,
never silently converted. Real vendor data must never be committed to the
repository (licence): only aggregated results with their citation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fund_profile import build_profile
from validation import MIN_OBS, valida_dati

RETURN_COLS = ["date", "fund_id", "value", "kind"]
META_COLS = ["fund_id", "region", "currency", "asset_class", "pricing", "style",
             "benchmark_id", "inception", "expense_ratio", "aum",
             "domicile", "is_dead", "last_date", "category_label"]
REQUIRED_FOR_PIPELINE = ["fund_id", "region", "currency", "asset_class"]
KINDS = ("return", "nav")
MAX_EXPENSE_RATIO = 0.20       # above this the value looks like percent


@dataclass
class LoadResult:
    returns: dict = field(default_factory=dict)     # fund_id -> monthly return Series
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    rejected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["fund_id", "reason"]))
    warnings: dict = field(default_factory=dict)    # fund_id or "_file" -> [str]

    def profiles(self, user_input=None):
        """FundProfile per loaded fund (metadata first, then user input, else unknown)."""
        out = {}
        for fid in self.returns:
            row = self.meta.loc[fid].to_dict() if fid in self.meta.index else {}
            ui = (user_input or {}).get(fid)
            p = build_profile(fid, row, ui)
            if fid not in self.meta.index and not ui:
                p.data_warnings.append("No metadata row for this fund_id.")
            out[fid] = p
        return out


def _read(source):
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return pd.read_csv(source, dtype=str, keep_default_na=False)


def _clean_columns(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def read_meta_csv(source):
    """Read and validate meta.csv. Returns (meta indexed by fund_id, warnings)."""
    df = _clean_columns(_read(source))
    warns = []
    if "fund_id" not in df.columns:
        raise ValueError("meta.csv: required column 'fund_id' is missing.")
    df["fund_id"] = df["fund_id"].astype(str).str.strip()
    if (df["fund_id"] == "").any():
        raise ValueError("meta.csv: empty fund_id found.")
    dup = df["fund_id"][df["fund_id"].duplicated()].unique()
    if len(dup):
        raise ValueError(f"meta.csv: duplicated fund_id {list(dup)}; one row per fund "
                         "(share classes need distinct fund_id).")
    extra = [c for c in df.columns if c not in META_COLS]
    if extra:
        warns.append(f"meta.csv: unknown column(s) ignored: {extra}.")
    for c in META_COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[META_COLS].set_index("fund_id")
    er = pd.to_numeric(df["expense_ratio"].replace("", np.nan), errors="coerce")
    bad = er > MAX_EXPENSE_RATIO
    if bad.any():
        warns.append(f"expense_ratio above {MAX_EXPENSE_RATIO} for {list(df.index[bad])}: "
                     "looks like percent (expect decimal, e.g. 0.0075); value ignored.")
        er[bad] = np.nan
    df["expense_ratio"] = er
    df["aum"] = pd.to_numeric(df["aum"].replace("", np.nan), errors="coerce")
    insc = pd.to_datetime(df["inception"].replace("", np.nan), errors="coerce")
    badi = insc.isna() & (df["inception"].astype(str).str.strip() != "")
    if badi.any():
        warns.append(f"inception not a date for {list(df.index[badi])}; ignored.")
    df["inception"] = insc.dt.strftime("%Y-%m-%d")
    return df, warns


def _nav_to_returns(g, fid, warns):
    """NAV rows of one fund -> monthly returns on month-ends. Returns spanning a
    missing month are NOT computed (NaN) and counted as gaps."""
    nav = g.groupby("me")["value"].last()
    if (g.groupby("me").size() > 1).any():
        warns.append("several NAV dates per month: last value of each month used.")
    nav = nav.sort_index()
    if (nav <= 0).any() or not np.isfinite(nav.to_numpy()).all():
        raise ValueError("NAV must be positive and finite.")
    full = pd.date_range(nav.index[0], nav.index[-1], freq="ME")
    nav = nav.reindex(full)
    gaps = int(nav.isna().sum())
    if gaps:
        warns.append(f"{gaps} month(s) without NAV: returns spanning a gap are not "
                     "computed (left missing), not spread over the gap.")
    r = nav / nav.shift(1) - 1.0
    return r.iloc[1:]


def read_returns_csv(source):
    """Read returns.csv. Returns (dict fund_id -> Series, rejected DataFrame, warnings dict)."""
    df = _clean_columns(_read(source))
    missing = [c for c in RETURN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"returns.csv: required column(s) missing: {missing}; "
                         f"expected {RETURN_COLS}.")
    df = df[RETURN_COLS].copy()
    df["fund_id"] = df["fund_id"].astype(str).str.strip()
    df["kind"] = df["kind"].astype(str).str.strip().str.lower()
    if (df["fund_id"] == "").any():
        raise ValueError("returns.csv: empty fund_id found.")
    badk = sorted(set(df["kind"]) - set(KINDS))
    if badk:
        raise ValueError(f"returns.csv: kind must be one of {KINDS}; found {badk}.")
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except (ValueError, TypeError) as e:
        raise ValueError(f"returns.csv: unparseable date ({e}).") from None
    raw = df["value"].astype(str).str.strip()
    val = pd.to_numeric(raw.replace("", np.nan), errors="coerce")
    nonnum = (val.isna()) & (raw != "") & (raw.str.lower() != "nan")
    if nonnum.any():
        raise ValueError(f"returns.csv: non-numeric value in row(s) "
                         f"{list(df.index[nonnum][:5] + 2)} (first 5 shown).")
    df["value"] = val
    df["me"] = df["date"].dt.normalize() + pd.offsets.MonthEnd(0)

    out, rej, warns = {}, [], {}
    for fid, g in df.groupby("fund_id", sort=False):
        w = []
        try:
            if g["kind"].nunique() > 1:
                raise ValueError("mixed kinds (return and nav) for one fund_id.")
            n_nan = int(g["value"].isna().sum())
            if n_nan:
                w.append(f"{n_nan} empty value(s) dropped.")
                g = g.dropna(subset=["value"])
            if g.empty:
                raise ValueError("no values.")
            exact = g.duplicated(subset=["date", "value"])
            if exact.any():
                w.append(f"{int(exact.sum())} exact duplicate row(s) removed.")
                g = g[~exact]
            if g["date"].duplicated().any():
                raise ValueError("same date with different values: conflicting duplicates. "
                                 "Share classes must have distinct fund_id.")
            kind = g["kind"].iloc[0]
            if kind == "nav":
                r = _nav_to_returns(g, fid, w)
            else:
                if g["me"].duplicated().any():
                    raise ValueError("more than one return in a calendar month: monthly "
                                     "series required (use kind=nav for higher frequency).")
                r = g.set_index("me")["value"].sort_index()
            r = r.rename(fid)
            r.index.name = None
            clean, _, vw = valida_dati(r)      # raises on percent units, inf, etc.
            w.extend(vw)
            if len(clean) < MIN_OBS:
                w.append(f"short history: {len(clean)} months (< {MIN_OBS}); tests return "
                         "'insufficient data' and the fund is excluded from references.")
            out[fid] = clean.rename(fid)
        except ValueError as e:
            rej.append(dict(fund_id=fid, reason=str(e)))
        if w:
            warns[fid] = w
    return out, pd.DataFrame(rej, columns=["fund_id", "reason"]), warns


def load_csv_universe(returns_source, meta_source=None):
    """Load a universe (returns + optional metadata) with explicit validation."""
    res = LoadResult()
    res.returns, res.rejected, res.warnings = read_returns_csv(returns_source)
    if meta_source is not None:
        res.meta, mw = read_meta_csv(meta_source)
        if mw:
            res.warnings["_file"] = mw
        unknown = [f for f in res.meta.index if f not in set(res.returns) | set(res.rejected.fund_id)]
        if unknown:
            res.warnings.setdefault("_file", []).append(
                f"meta rows without returns: {unknown}.")
    else:
        res.meta = pd.DataFrame(columns=META_COLS[1:]).rename_axis("fund_id")
    return res


def read_benchmarks_csv(source):
    """benchmarks.csv: date, benchmark_id, value, kind (same rules as returns.csv).
    Returns (dict benchmark_id -> monthly return Series, rejected, warnings)."""
    df = _clean_columns(_read(source))
    if "benchmark_id" not in df.columns:
        raise ValueError("benchmarks.csv: required column 'benchmark_id' is missing.")
    return read_returns_csv(df.rename(columns={"benchmark_id": "fund_id"}))


def read_field_log(source):
    """field_log.csv: field mnemonic used, retrieval date, universe definition (for sourcing)."""
    df = _clean_columns(_read(source))
    need = ["field", "retrieval_date", "universe"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"field_log.csv: required column(s) missing: {miss}.")
    return df[need]


def anonymise(fund_ids, prefix="F"):
    """Sequential anonymous labels (F01, F02, ...) for reports. Keep the mapping
    private: never commit it or paste it in a deliverable."""
    return {fid: f"{prefix}{i + 1:02d}" for i, fid in enumerate(fund_ids)}
