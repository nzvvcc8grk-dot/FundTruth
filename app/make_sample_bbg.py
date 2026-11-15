"""SYNTHETIC but realistic-looking Bloomberg-style sample (no real data) -> sample_data/bbg
and sample_data/bbg_bad (one deliberate wrong-unit fund, for the stop test).

Groups: US equity 24, Europe equity 22, US bond 22, appraised-NAV funds 12 (property, private
credit, fund of hedge funds), 2 dead funds, and four funds that ingestion must EXCLUDE (short
history, a gap, repeated zeros, constant). IDs are anonymised codes F001... Factors: synthetic,
one realisation per region (US seed 0, Europe seed 1, Global seed 2), saved as factors_<region>.csv.
"""
import os
import shutil

import numpy as np
import pandas as pd

from synthetic import CATEGORY_PARAMS, make_factors

N = 120
FAC = {"US": make_factors(N, 0), "Europe": make_factors(N, 1), "Global": make_factors(N, 2)}
rows, meta, nid = [], [], [0]


def new_id():
    nid[0] += 1
    return f"F{nid[0]:03d}"


def gen(category, rng, F, indexlike=False, smooth=None, alpha=None):
    p = CATEGORY_PARAMS[category]
    b = rng.normal(*p["beta"]); s = rng.normal(*p["smb"]); h = rng.normal(*p["hml"])
    a = rng.normal(*p["alpha"]) if alpha is None else alpha
    sd = 0.0008 if indexlike else rng.uniform(*p["eps"])
    if indexlike:
        b, s, h, a = 1.0, 0.0, 0.0, -0.001
    r = F["RF"] + a / 12 + b * F["Mkt-RF"] + s * F["SMB"] + h * F["HML"] + rng.normal(0, sd, len(F))
    if smooth:
        th = np.asarray(smooth, float) / np.sum(smooth)
        r = th[0] * r + th[1] * r.shift(1) + th[2] * r.shift(2)
        r = r.iloc[2:]
    return r


def add(r, region, ccy, ac, pricing, bench, exp, kind="return", dead=False, label="", style="", drop=None, **extra):
    fid = new_id()
    s = r.copy()
    if dead:
        s = s.iloc[:-30]
    if kind == "nav":
        s = 100 * (1 + s).cumprod()
        first = s.index[0] - pd.offsets.MonthEnd(1)
        s = pd.concat([pd.Series([100.0], index=[first]), s])
    if drop is not None:
        s = s.drop(s.index[drop])
    rows.extend(dict(date=d.strftime("%Y-%m-%d"), fund_id=fid, value=f"{v:.8f}", kind=kind) for d, v in s.items())
    meta.append(dict(fund_id=fid, region=region, currency=ccy, asset_class=ac, pricing=pricing, benchmark_id=bench,
                     expense_ratio="" if exp is None else f"{exp:.4f}", inception="2008-01-31", aum="",
                     domicile={"US": "US", "Europe": "LU", "Global": "LU"}[region],
                     is_dead="True" if dead else "False", last_date=s.index[-1].strftime("%Y-%m-%d") if dead else "",
                     category_label=label))
    return fid


rng = np.random.default_rng(2026)
# US equity: 8 exchange-traded (4 index-like), 16 NAV from listed holdings (2 closet-like), 1 dead
for i in range(24):
    idx = i < 4
    closet = i in (8, 9)
    r = gen("equity", rng, FAC["US"], indexlike=idx or closet)
    add(r, "US", "USD", "equity", "exchange_traded" if i < 8 else "nav_listed_holdings", "BMK_US",
        0.0004 if idx else 0.0110 if closet else float(rng.uniform(0.004, 0.012)), dead=(i == 23),
        label="US Large Cap Blend")
# Europe equity: 22 funds, mixed currency declared, 6 given as NAV
for i in range(22):
    ccy = "EUR" if i < 14 else "GBP" if i < 19 else "USD"
    add(gen("equity", rng, FAC["Europe"]), "Europe", ccy, "equity", "exchange_traded" if 14 <= i < 18 else "nav_listed_holdings",
        "BMK_EU", float(rng.uniform(0.004, 0.015)), kind="nav" if i % 4 == 0 else "return", dead=(i == 21),
        label="Europe Large Cap")
# US bonds
for i in range(22):
    add(gen("bond", rng, FAC["US"]), "US", "USD", "bond", "exchange_traded" if i < 8 else "nav_listed_holdings",
        "BMK_AGG", float(rng.uniform(0.002, 0.008)), label="US Aggregate Bond")
# Appraised NAV: property (real_assets), private credit (other), fund of hedge funds (multi-asset)
for i in range(12):
    ac, lab = (("real_assets", "Property"), ("other", "Private Credit"), ("multi-asset", "Fund of Hedge Funds"))[i // 4]
    reg = "Europe" if i % 2 else "Global"
    add(gen("multi-asset", rng, FAC[reg], smooth=(0.5, 0.3, 0.2) if i % 3 else None, alpha=0.01), reg, "USD" if reg == "Global" else "EUR",
        ac, "appraised_nav", "", float(rng.uniform(0.008, 0.02)), kind="nav", label=lab)
# funds that must be EXCLUDED by ingestion (each for a different reason)
base = gen("equity", rng, FAC["US"])
add(base.iloc[-24:], "US", "USD", "equity", "nav_listed_holdings", "BMK_US", 0.008, label="short history")
add(base, "US", "USD", "equity", "nav_listed_holdings", "BMK_US", 0.008, drop=[40, 41, 42], label="gap")
z = base.copy(); z.iloc[50:56] = 0.0
add(z, "US", "USD", "equity", "nav_listed_holdings", "BMK_US", 0.008, label="stale zeros")
add(pd.Series(0.004, index=base.index), "US", "USD", "bond", "nav_listed_holdings", "BMK_AGG", 0.003, label="constant")

M = pd.DataFrame(meta)
R = pd.DataFrame(rows)
bm_rows = []
for bid, reg in (("BMK_US", "US"), ("BMK_EU", "Europe"), ("BMK_AGG", "US")):
    F = FAC[reg]
    r = (F["RF"] + F["Mkt-RF"]) if bid != "BMK_AGG" else (F["RF"] + 0.1 * F["Mkt-RF"] + 0.002)
    r = r + np.random.default_rng(abs(hash(bid)) % 1000).normal(0, 0.001, len(F))
    bm_rows += [dict(date=d.strftime("%Y-%m-%d"), benchmark_id=bid, value=f"{v:.8f}", kind="return") for d, v in r.items()]
FL = pd.DataFrame(dict(field=["TOTAL RETURN / NAV (SYNTHETIC EXAMPLE)", "FUND METADATA (SYNTHETIC EXAMPLE)"],
                       retrieval_date=["1900-01-01", "1900-01-01"], universe=["synthetic sample, not a real universe"] * 2))


def write(folder, R, M):
    os.makedirs(folder, exist_ok=True)
    R.to_csv(f"{folder}/returns.csv", index=False)
    M.to_csv(f"{folder}/meta.csv", index=False)
    pd.DataFrame(bm_rows).to_csv(f"{folder}/benchmarks.csv", index=False)
    FL.to_csv(f"{folder}/field_log.csv", index=False)
    for reg, F in FAC.items():
        F.to_csv(f"{folder}/factors_{reg}.csv", index_label="date")


write("sample_data/bbg", R, M)
# bad copy: one fund delivered in PERCENT (deliberate wrong unit)
bad = R.copy()
fid = M.fund_id.iloc[3]
m = bad.fund_id == fid
bad.loc[m, "value"] = (bad.loc[m, "value"].astype(float) * 100).map("{:.6f}".format)
write("sample_data/bbg_bad", bad, M)
print("written", R.fund_id.nunique(), "funds; bad-unit fund:", fid)
