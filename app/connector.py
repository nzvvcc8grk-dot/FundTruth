"""Data connector: fetch any fund's monthly returns and the Fama-French factors
on demand, so the user picks a fund and FundTruth scores it.

Only the two public sources already used by data.py are contacted: Yahoo
Finance (yfinance) and the Kenneth French Data Library. Each request carries
only the ticker. Downloads are cached in .cache/ so a repeated analysis is
instant and funds already fetched still work offline.

Nothing is invented: a fund that cannot be downloaded or validated raises
ConnectorError with the reason.
"""
from __future__ import annotations

import os
import re
import time

import pandas as pd

from data import build_dataset, download_prices, load_ff
from fundtruth import CATEGORIES, FundTruth

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("FUNDTRUTH_CACHE", os.path.join(HERE, ".cache"))
PRICE_TTL_DAYS = 7
FACTOR_TTL_DAYS = 30
DEFAULT_START = "2016-01-01"
STYLES = ("active", "passive")
_TICKER = re.compile(r"^[A-Z0-9.\-]{1,12}$")


class ConnectorError(Exception):
    """A fund or factor download that cannot be used, with a readable reason."""


def _fresh(path, days):
    return os.path.exists(path) and time.time() - os.path.getmtime(path) < days * 86400


def _read(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)


def clean_ticker(ticker):
    t = str(ticker).strip().upper()
    if not _TICKER.match(t):
        raise ConnectorError(f"'{ticker}' is not a valid ticker (letters, digits, '.' or '-').")
    return t


# ------------------------------------------------------------------ universe
def load_universe():
    """Built-in list of 40 funds and ETFs (tickers.csv): category and style known."""
    return pd.read_csv(os.path.join(HERE, "tickers.csv"))


def lookup(ticker):
    """(category, style) of a ticker in the built-in list, else None."""
    u = load_universe()
    row = u[u["ticker"] == str(ticker).strip().upper()]
    if row.empty:
        return None
    return row.iloc[0]["category"], row.iloc[0]["style"]


def _yahoo_search(query, limit):
    import yfinance as yf
    return yf.Search(query, max_results=limit).quotes


def search(query, limit=10):
    """Funds matching `query`: the built-in list first, then Yahoo Finance
    (mutual funds and ETFs only). Falls back to the built-in list if offline."""
    q = str(query).strip()
    if not q:
        return []
    u = load_universe()
    hits = u[u["ticker"].str.contains(q, case=False, regex=False)]
    out = [dict(ticker=r.ticker, name=f"{r.category}, {r.style} ({r.type})",
                source="built-in list") for r in hits.itertuples()]
    try:
        for h in _yahoo_search(q, limit):
            if h.get("quoteType") in ("MUTUALFUND", "ETF") and h.get("symbol"):
                out.append(dict(ticker=h["symbol"].upper(),
                                name=h.get("shortname") or h.get("longname") or "",
                                source="Yahoo Finance"))
    except Exception:
        pass                      # offline or API changed: built-in list only
    seen, uniq = set(), []
    for o in out:
        if o["ticker"] not in seen:
            seen.add(o["ticker"])
            uniq.append(o)
    return uniq[:limit]


# ------------------------------------------------------------------ downloads
def get_factors(cache_dir=None, ff_file=None):
    """Fama-French monthly factors (decimal). Cached for 30 days; a stale cache
    or a data/factors.csv built by data.py is used if the download fails."""
    cache_dir = cache_dir or CACHE_DIR
    path = os.path.join(cache_dir, "factors.csv")
    if ff_file is None and _fresh(path, FACTOR_TTL_DAYS):
        return _read(path)
    try:
        f = load_ff(ff_file)
    except Exception as e:
        local = os.path.join(HERE, "data", "factors.csv")
        for p in (path, local):
            if os.path.exists(p):
                return _read(p)
        raise ConnectorError(
            "Could not download the Fama-French factors (no internet?). "
            "Run 'python data.py' once online, or pass a manually downloaded file."
        ) from e
    os.makedirs(cache_dir, exist_ok=True)
    f.to_csv(path, index_label="date")
    return f


def get_prices(ticker, start=DEFAULT_START, cache_dir=None):
    """Adjusted prices of one ticker from Yahoo Finance, cached for 7 days."""
    cache_dir = cache_dir or CACHE_DIR
    t = clean_ticker(ticker)
    path = os.path.join(cache_dir, f"prices_{t}_{str(start)[:4]}.csv")
    if _fresh(path, PRICE_TTL_DAYS):
        return _read(path)
    try:
        p = download_prices([t], start)
    except Exception as e:
        if os.path.exists(path):
            return _read(path)
        raise ConnectorError(f"Could not download prices for {t}: {e}") from e
    if p is None or p.empty or t not in p.columns or p[t].dropna().empty:
        raise ConnectorError(
            f"No price data found for '{t}'. Check the ticker (mutual funds and "
            "ETFs listed on Yahoo Finance work best).")
    p = p[[t]].dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    p.to_csv(path, index_label="date")
    return p


def fetch_returns(ticker, category="equity", style="active", start=DEFAULT_START,
                  cache_dir=None, factors=None):
    """Monthly total returns (decimal, month-end), cleaned and validated with the
    same pipeline as data.py (aligned to the factor months, at least 36 months)."""
    if category not in CATEGORIES:
        raise ConnectorError(f"category must be one of {CATEGORIES}.")
    if style not in STYLES:
        raise ConnectorError(f"style must be one of {STYLES}.")
    t = clean_ticker(ticker)
    factors = get_factors(cache_dir) if factors is None else factors
    prices = get_prices(t, start, cache_dir)
    meta = pd.DataFrame([dict(ticker=t, category=category, style=style, fund_id=t)])
    wide, _, rej = build_dataset(meta, prices, factors)
    if wide.empty or t not in wide:
        reason = rej.iloc[0]["reason"] if len(rej) else "unknown reason"
        raise ConnectorError(f"{t} cannot be scored: {reason}.")
    return wide[t]


# ------------------------------------------------------------------ scoring
def load_reference(exclude=None):
    """Real-fund peer reference built by data.py (data/reference_real_v1.csv),
    or None if it has not been built on this machine."""
    path = os.path.join(HERE, "data", "reference_real_v1.csv")
    if not os.path.exists(path):
        return None
    ref = pd.read_csv(path, index_col="fund")
    if exclude is not None and exclude in ref.index:
        ref = ref.drop(index=exclude)       # leave-one-out, as in score_funds.py
    return ref


def analyse(ticker, category="equity", style="active", start=DEFAULT_START,
            use_real_reference=False, cache_dir=None):
    """Download, validate and score one fund. Returns (FundTruth, info dict)."""
    t = clean_ticker(ticker)
    factors = get_factors(cache_dir)
    r = fetch_returns(t, category, style, start, cache_dir, factors)
    peers = load_reference(t) if use_real_reference else None
    ft = FundTruth(r.rename(t), factors, peers=peers, category=category, style=style)
    info = dict(ticker=t, months=len(r), first=r.index[0].date(), last=r.index[-1].date(),
                reference=(f"real funds (n={len(peers)}, leave-one-out)"
                           if peers is not None else "synthetic placeholder (v0)"))
    return ft, info


def cached_scores():
    """Scores of the 40 built-in funds already computed (offline view), or None."""
    path = os.path.join(HERE, "results", "real_funds_scores_real.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ------------------------------------------------------------------ context-aware (Level 1b)
def get_region_factors(region, cache_dir=None, ff_file=None):
    """3-factor file of a region (US, Europe, Global=Developed proxy), cached 30 days.
    Returns None when no file is mapped (EM, Other, unknown): the factor test is skipped."""
    from data import FF_REGION_FILES
    if region not in FF_REGION_FILES:
        return None
    if region == "US":
        return get_factors(cache_dir, ff_file)
    cache_dir = cache_dir or CACHE_DIR
    path = os.path.join(cache_dir, f"factors_{region}.csv")
    if ff_file is None and _fresh(path, FACTOR_TTL_DAYS):
        return _read(path)
    try:
        f = load_ff(ff_file, region)
    except Exception as e:
        if os.path.exists(path):
            return _read(path)
        raise ConnectorError(f"Could not download the {region} factors (no internet?). "
                             "Download the file from the Kenneth French Data Library and "
                             "pass it with ff_file.") from e
    os.makedirs(cache_dir, exist_ok=True)
    f.to_csv(path, index_label="date")
    return f


def fetch_series(ticker, start=DEFAULT_START, cache_dir=None):
    """Monthly total returns of one ticker (decimal, month-end), validated, NOT aligned
    to any factor set (alignment happens in the analysis, per region)."""
    from data import monthly_returns
    from validation import valida_dati
    t = clean_ticker(ticker)
    prices = get_prices(t, start, cache_dir)
    s = monthly_returns(prices)[t].dropna().rename(t)
    try:
        clean, _, _ = valida_dati(s)
    except ValueError as e:
        raise ConnectorError(f"{t} cannot be used: {e}") from e
    return clean


# Proposed benchmark proxies (ETFs on Yahoo Finance), editable by the user. They are
# proposals, not facts about the fund's declared benchmark.
BENCHMARK_PROPOSALS = {("equity", "US"): "SPY", ("bond", "US"): "AGG",
                       ("equity", "Europe"): "VGK", ("equity", "Global"): "VT"}


def propose_benchmark(asset_class, region):
    return BENCHMARK_PROPOSALS.get((asset_class, region))
