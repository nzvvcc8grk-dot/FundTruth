"""Turn the files downloaded by scarica_dati.sh (public sources only) into the data/ folder.

    python public_data.py --raw ../fundtruth_dati/raw --out data

Inputs (raw/):
  yahoo/<TICKER>.json   Yahoo Finance chart API, daily, adjusted close (dividends reinvested)
  ff/F-F_Research_Data_Factors_CSV.zip   Kenneth R. French Data Library, monthly 3 factors
  holdings/*.xlsx|*.csv  ETF holdings from the issuers (optional)
  nbim_exclusions.csv    Norges Bank exclusion list via OpenSanctions (optional)
Outputs (data/):
  fund_returns.csv, factors.csv, funds_meta.csv, rejected.csv, reference_real_v1.csv/json,
  data_note.txt                 -> same files as data.py (FundTruth, 40 funds)
  esg_returns.csv               -> monthly total returns of the ESG-module funds, parents, energy proxies
  holdings/<ticker>.csv         -> normalised holdings (name, ticker, weight, sector)
  nbim_exclusions.csv           -> copied as is
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import zipfile
from datetime import date

import pandas as pd

import data as D

HERE = os.path.dirname(os.path.abspath(__file__))


def yahoo_daily(path):
    """Adjusted daily close from one Yahoo chart JSON file; None if the file is not usable."""
    try:
        js = json.load(open(path))
        res = js["chart"]["result"][0]
        ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
        adj = res["indicators"]["adjclose"][0]["adjclose"]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    s = pd.Series(adj, index=ts, dtype="float64").dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s if len(s) > 20 else None


def load_prices(raw):
    """Daily adjusted prices: Yahoo chart JSON files, or raw/yahoo_prices.csv written by yfinance
    (auto_adjust=True) when Yahoo refuses direct downloads."""
    prices, missing = {}, []
    for f in sorted(glob.glob(os.path.join(raw, "yahoo", "*.json"))):
        t = os.path.basename(f)[:-5]
        s = yahoo_daily(f)
        (missing.append(t) if s is None else prices.__setitem__(t, s))
    csv = os.path.join(raw, "yahoo_prices.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        for t in df.columns:
            s = df[t].dropna()
            if len(s) > 20 and t not in prices:
                prices[t] = s
                missing = [m for m in missing if m != t]
            elif t not in prices:
                missing.append(t)
    return pd.DataFrame(prices).sort_index(), sorted(set(missing))


def load_factors(raw):
    z = os.path.join(raw, "ff", "F-F_Research_Data_Factors_CSV.zip")
    with zipfile.ZipFile(z) as zz:
        name = next(n for n in zz.namelist() if n.lower().endswith((".csv", ".txt")))
        return D.parse_ff_factors(zz.read(name).decode("latin-1"))


def read_holdings(path):
    """Issuer holdings file (State Street .xlsx or iShares .csv) -> name, ticker, weight (0-1), sector."""
    if path.endswith(".xlsx"):
        raw = pd.read_excel(path, header=None)
    else:
        raw = pd.read_csv(path, header=None, skip_blank_lines=False, on_bad_lines="skip",
                          engine="python", names=range(30))
    hdr = None
    for i in range(min(40, len(raw))):
        cells = [str(c).strip().lower() for c in raw.iloc[i].tolist()]
        if any(c.startswith("weight") for c in cells) and ("name" in cells or "ticker" in cells):
            hdr = i
            break
    if hdr is None:
        raise ValueError(f"{os.path.basename(path)}: no header row with Name/Ticker and Weight")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr].tolist()]
    df = df.loc[:, [c for c in df.columns if c and c.lower() != "nan"]]
    col = lambda *keys: next((c for c in df.columns if any(c.lower().startswith(k) for k in keys)), None)
    out = pd.DataFrame({"name": df[col("name")].astype(str).str.strip() if col("name") else "",
                        "ticker": df[col("ticker")].astype(str).str.strip() if col("ticker") else "",
                        "weight": pd.to_numeric(df[col("weight")].astype(str).str.replace(",", "").str.rstrip("%"),
                                                errors="coerce"),
                        "sector": df[col("sector")].astype(str).str.strip() if col("sector") else ""})
    if "asset class" in [c.lower() for c in df.columns]:
        ac = df[next(c for c in df.columns if c.lower() == "asset class")].astype(str).str.lower()
        out = out[ac.values == "equity"]
    out = out.dropna(subset=["weight"])
    out = out[out["name"].str.len() > 0]
    out = out[~out["name"].str.lower().isin(["nan", "cash", "us dollar", "usd cash"])]
    out = out[out["weight"] > 0]
    tot = out["weight"].sum()
    if tot > 1.5:                                  # file in percent
        out["weight"] = out["weight"] / 100.0
    return out.reset_index(drop=True)


def add_uk_prices(prices, raw):
    """London-listed ETFs in GBP. USD-quoted series (and IXC) are converted with GBP/USD;
    UKENERGY = equal-weight daily index of Shell and BP (kept for UK-equity funds)."""
    p = os.path.join(raw, "yahoo_prices_uk.csv")
    if not os.path.exists(p):
        return prices
    uk = pd.read_csv(p, index_col=0, parse_dates=True)
    uk.index = pd.DatetimeIndex(uk.index).tz_localize(None).normalize()
    cur = {}
    cp = os.path.join(raw, "yahoo_currency_uk.csv")
    if os.path.exists(cp):
        cur = dict(pd.read_csv(cp).fillna("").values)
    fx = uk.pop("GBPUSD=X").ffill() if "GBPUSD=X" in uk else None   # USD per GBP
    eur = uk.pop("EURGBP=X").ffill() if "EURGBP=X" in uk else None  # GBP per EUR
    out = prices.copy()
    for t in uk.columns:
        s = uk[t].dropna()
        if len(s) < 20:
            continue
        c = str(cur.get(t, "")).upper()
        if c == "USD":
            if fx is None:
                continue
            s = (s / fx.reindex(s.index, method="ffill")).dropna()
        elif c == "EUR":
            if eur is None:
                print(f"{t}: quoted in EUR but no EUR/GBP rate downloaded: skipped")
                continue
            s = (s * eur.reindex(s.index, method="ffill")).dropna()
        out[t] = s
    if fx is not None and "IXC" in prices:
        ixc = prices["IXC"].dropna()
        out["IXC_GBP"] = (ixc / fx.reindex(ixc.index, method="ffill")).dropna()
    if {"SHEL.L", "BP.L"} <= set(uk.columns):
        r = uk[["SHEL.L", "BP.L"]].pct_change(fill_method=None).mean(axis=1, skipna=False).dropna()
        out["UKENERGY"] = (1 + r).cumprod()
    return out.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "fundtruth_dati", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--start", default="2016-01-01")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prices, missing = load_prices(a.raw)
    factors = load_factors(a.raw)
    print(f"prices for {prices.shape[1]} tickers; missing: {missing or 'none'}")

    # FundTruth: the 40 funds, same treatment as data.py
    meta = pd.read_csv(os.path.join(HERE, "tickers.csv"))
    pr = prices.loc[prices.index >= pd.Timestamp(a.start)]          # same window as data.py: first return = Feb 2016
    wide, ok, rej = D.build_dataset(meta, pr[[t for t in meta.ticker if t in pr]], factors)
    for t in meta.ticker:
        if t not in pr and t not in set(rej.ticker):
            rej.loc[len(rej)] = (t, "no price data downloaded")
    fac = factors.loc[factors.index.intersection(wide.index)]
    wide.to_csv(f"{a.out}/fund_returns.csv", index_label="date")
    fac.to_csv(f"{a.out}/factors.csv", index_label="date")
    ok.to_csv(f"{a.out}/funds_meta.csv", index=False)
    rej.to_csv(f"{a.out}/rejected.csv", index=False)
    ref = D.reference_from_returns(wide, ok, fac)
    ref.to_csv(f"{a.out}/reference_real_v1.csv", index_label="fund")
    json.dump({"version": f"real-v1-{date.today().isoformat()}", "n_funds": len(ref)},
              open(f"{a.out}/reference_real_v1.json", "w"))
    D.write_note(f"{a.out}/data_note.txt", ok, rej, factors, a.start)
    print(f"FundTruth: kept {len(ok)} funds, rejected {len(rej)}")

    # ESG module: funds, parents and energy proxies, full history
    claims = json.load(open(os.path.join(HERE, "esg_claims.json")))["funds"]
    need = sorted({t for k, v in claims.items() for t in (k, v["parent"], v["energy_proxy"])})
    prices = add_uk_prices(prices, a.raw)
    have = [t for t in need if t in prices]
    esg = D.monthly_returns(prices[have]).dropna(how="all")
    esg = esg.loc[esg.index <= factors.index.max()] if len(factors) else esg
    esg.to_csv(f"{a.out}/esg_returns.csv", index_label="date")
    print(f"ESG module: returns for {len(have)}/{len(need)} tickers "
          f"({', '.join(t for t in need if t not in have) or 'none missing'})")

    hdir = os.path.join(a.out, "holdings")
    os.makedirs(hdir, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.raw, "holdings", "*"))):
        t = os.path.splitext(os.path.basename(f))[0].upper()
        try:
            h = read_holdings(f)
            h.to_csv(os.path.join(hdir, f"{t}.csv"), index=False)
            print(f"holdings {t}: {len(h)} lines, weights sum {h.weight.sum():.3f}")
        except Exception as e:                      # noqa: BLE001 - report and continue
            print(f"holdings {t}: NOT USABLE ({e})")
    src = os.path.join(a.raw, "nbim_exclusions.csv")
    if os.path.exists(src) and os.path.getsize(src) > 200:
        shutil.copy(src, os.path.join(a.out, "nbim_exclusions.csv"))
        print("NBIM exclusion list copied")


if __name__ == "__main__":
    main()
