"""Data pipeline: fund returns (yfinance) + Fama-French factors -> clean CSVs.

Runs on the group's machine (the build sandbox cannot reach Yahoo or the
Kenneth French library). Nothing is invented: every number comes from the
downloads; funds failing validation are written to rejected.csv with a reason.

Usage:
  python data.py --tickers tickers.csv --start 2016-01-01 --out data
  python data.py --tickers tickers.csv --ff-file F-F_Research_Data_Factors_CSV.zip
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from datetime import date

import pandas as pd

from features import compute_features
from validation import MIN_OBS, percent_to_decimal, to_month_end, valida_dati

# TO VERIFY: official location of the monthly 3-factor file (Kenneth R. French
# Data Library, Tuck School). Use --ff-file with a manual download if it fails.
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_CSV.zip")
FF_COLS = ["Mkt-RF", "SMB", "HML", "RF"]
_ROW = re.compile(r"^\s*(\d{6})\s*,(.*)$")


def parse_ff_factors(text):
    """Parse the monthly block of the French 3-factor CSV. Returns decimals
    (percent divided by 100 exactly once), month-end index."""
    rows, started = [], False
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            started = True
            vals = [v.strip() for v in m.group(2).split(",")]
            if len(vals) < 4:
                raise ValueError(f"unexpected factor row: {line!r}")
            rows.append([m.group(1)] + vals[:4])
        elif started:
            break                      # end of the monthly block (annual follows)
    if not rows:
        raise ValueError("no monthly factor rows found (format changed?).")
    df = pd.DataFrame(rows, columns=["ym"] + FF_COLS)
    df.index = pd.to_datetime(df.pop("ym"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.where(df > -99.0)          # -99.99 / -999 are missing-value codes
    return percent_to_decimal(df).dropna()


FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/ftp/"
# Monthly 3-factor files; US, Developed and Europe names verified on the library
# page (2026-11-13). Currency of the international files: to verify on the library.
FF_REGION_FILES = {"US": "F-F_Research_Data_Factors_CSV.zip",
                   "Europe": "Europe_3_Factors_CSV.zip",
                   "Global": "Developed_3_Factors_CSV.zip"}


def ff_url(region="US"):
    """Download URL of the monthly 3-factor file of a region."""
    if region not in FF_REGION_FILES:
        raise ValueError(f"no factor file mapped for region {region!r}; "
                         f"known: {sorted(FF_REGION_FILES)}")
    return FF_BASE + FF_REGION_FILES[region]


def check_ff_layout(text):
    """Summary of a downloaded French factor file, to confirm the parser reads
    the real layout (run it on the real file, not on a fixture)."""
    fac = parse_ff_factors(text)
    return dict(n_months=len(fac), first=str(fac.index[0].date()),
                last=str(fac.index[-1].date()), columns=list(fac.columns),
                max_abs=float(fac.abs().max().max()),
                has_annual_block=bool(re.search(r"(?im)^\s*annual", text)),
                header_lines=[ln.strip() for ln in text.splitlines()[:6]])


def load_ff(source=None, region="US"):
    """Read factors from a local .zip/.csv path, or download the region's file
    (default US, unchanged behaviour)."""
    if source is None:
        import urllib.request
        url = FF_URL if region == "US" else ff_url(region)
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
    else:
        with open(source, "rb") as fh:
            raw = fh.read()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".csv", ".txt")))
            raw = z.read(name)
    return parse_ff_factors(raw.decode("latin-1"))


def monthly_returns(prices):
    """Total-return monthly series from dividend/split-adjusted prices
    (yfinance auto_adjust=True). Index = calendar month-end."""
    p = prices.copy()
    p.index = pd.DatetimeIndex(p.index).tz_localize(None)
    m = p.resample("ME").last()
    return m.pct_change(fill_method=None).iloc[1:]


def download_prices(tickers, start):
    import yfinance as yf
    d = yf.download(list(tickers), start=start, auto_adjust=True,
                    progress=False, group_by="column")
    close = d["Close"]
    return close.to_frame(tickers[0]) if isinstance(close, pd.Series) else close


def build_dataset(meta, prices, factors, min_obs=MIN_OBS):
    """Return (returns_wide, meta_ok, rejected). `meta` needs columns
    ticker, category, style, fund_id. One share class per fund_id."""
    rets = monthly_returns(prices)
    good, ok_meta, rejected, seen = {}, [], [], set()
    for _, row in meta.iterrows():
        t = row["ticker"]
        if row["fund_id"] in seen:
            rejected.append((t, "duplicate share class of an included fund"))
            continue
        if t not in rets or rets[t].dropna().empty:
            rejected.append((t, "no price data downloaded"))
            continue
        s = rets[t].dropna().rename(t)
        s = s.loc[s.index.intersection(factors.index)]
        try:
            clean, _, warns = valida_dati(s, factors)
        except ValueError as e:
            rejected.append((t, f"validation error: {e}"))
            continue
        if len(clean) < min_obs:
            rejected.append((t, f"only {len(clean)} months with factors (< {min_obs})"))
            continue
        if clean.std() == 0:
            rejected.append((t, "constant series"))
            continue
        good[t] = clean
        seen.add(row["fund_id"])
        ok_meta.append(dict(row, n_obs=len(clean), first=clean.index[0].date(),
                            last=clean.index[-1].date(), warnings="; ".join(warns)))
    wide = pd.DataFrame(good).sort_index()
    return wide, pd.DataFrame(ok_meta), pd.DataFrame(rejected, columns=["ticker", "reason"])


def reference_from_returns(wide, meta, factors):
    """Feature table used as the frozen peer reference (one row per fund)."""
    rows = []
    for _, m in meta.iterrows():
        s = wide[m["ticker"]].dropna()
        f = compute_features(s, factors.loc[factors.index.intersection(s.index)])
        f.update(category=m["category"], style=m["style"])
        rows.append(pd.Series(f, name=m["ticker"]))
    return pd.DataFrame(rows)


def write_note(path, meta, rejected, factors, start):
    txt = f"""FundTruth data note (generated {date.today().isoformat()})
Sources: Yahoo Finance via yfinance (adjusted close, dividends reinvested);
Kenneth R. French Data Library, monthly Fama/French 3 factors (US, percent in the file, divided by 100 once here).
Period: requested start {start}; factors cover {factors.index[0].date()} to {factors.index[-1].date()}; funds are aligned to the common months.
Funds kept: {len(meta)}; rejected: {len(rejected)} (see rejected.csv).
Limitations: survivorship bias (only funds that exist today); Yahoo prices are
not audited fund NAVs and may be net of fees only through the price; some
tickers are share classes with loads not reflected; US factors are weak
explanatory variables for bond funds; private/hedge funds not covered.
Every figure derived from this file must cite these sources.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="tickers.csv")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--ff-file", default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--check-ff", default=None,
                    help="print the layout summary of a downloaded French factor file and exit")
    a = ap.parse_args()
    if a.check_ff:
        raw = open(a.check_ff, "rb").read()
        if raw[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                raw = z.read(next(n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))))
        print(json.dumps(check_ff_layout(raw.decode("latin-1")), indent=1))
        return
    import os
    os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.tickers)
    factors = load_ff(a.ff_file)
    prices = download_prices(list(meta["ticker"]), a.start)
    wide, ok, rej = build_dataset(meta, prices, factors)
    fac = factors.loc[factors.index.intersection(wide.index)]
    wide.to_csv(f"{a.out}/fund_returns.csv", index_label="date")
    fac.to_csv(f"{a.out}/factors.csv", index_label="date")
    ok.to_csv(f"{a.out}/funds_meta.csv", index=False)
    rej.to_csv(f"{a.out}/rejected.csv", index=False)
    ref = reference_from_returns(wide, ok, fac)
    ref.to_csv(f"{a.out}/reference_real_v1.csv", index_label="fund")
    with open(f"{a.out}/reference_real_v1.json", "w") as fh:
        json.dump({"version": f"real-v1-{date.today().isoformat()}",
                   "n_funds": len(ref)}, fh)
    write_note(f"{a.out}/data_note.txt", ok, rej, factors, a.start)
    print(f"kept {len(ok)} funds, rejected {len(rej)}; files in {a.out}/")


if __name__ == "__main__":
    main()
