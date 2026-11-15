"""Write a SYNTHETIC sample in the connector schema (no real data) to sample_data/.

All funds share ONE synthetic factor realisation (make_factors(96, 0)), the same one the
pipeline uses with --synthetic-factors, so factor regressions in the dry run are coherent."""
import os
import shutil

import numpy as np
import pandas as pd

from synthetic import CATEGORY_PARAMS, make_factors

F0 = make_factors(96, 0)
os.makedirs("sample_data", exist_ok=True)
rows, meta = [], []


def fund(category, seed, alpha=None, smooth=None):
    p = CATEGORY_PARAMS[category]
    rng = np.random.default_rng(seed)
    b, s, h = rng.normal(*p["beta"]), rng.normal(*p["smb"]), rng.normal(*p["hml"])
    a = rng.normal(*p["alpha"]) if alpha is None else alpha
    sd = rng.uniform(*p["eps"])
    r = F0["RF"] + a / 12 + b * F0["Mkt-RF"] + s * F0["SMB"] + h * F0["HML"] + rng.normal(0, sd, len(F0))
    if smooth:
        th = np.asarray(smooth, float) / np.sum(smooth)
        r = (th[0] * r + th[1] * r.shift(1) + th[2] * r.shift(2)).iloc[2:]
    return r


def add(fid, r, kind="return", drop=None, **m):
    s = r.copy()
    if kind == "nav":
        s = 100 * (1 + s).cumprod()
    if drop is not None:
        s = s.drop(s.index[drop])
    rows.extend(dict(date=d.strftime("%Y-%m-%d"), fund_id=fid, value=f"{v:.8f}", kind=kind)
                for d, v in s.items())
    meta.append(dict(fund_id=fid, **m))


rng = np.random.default_rng(5)
closet = F0["RF"] + F0["Mkt-RF"] - 0.00008 + rng.normal(0, 0.001, len(F0))
add("SYN_US_ETF", fund("equity", 1), region="US", currency="USD", asset_class="equity",
    pricing="exchange_traded", style="passive", expense_ratio=0.0003)
add("SYN_EU_EQ", fund("equity", 2), kind="nav", drop=[40], region="Europe", currency="EUR",
    asset_class="equity", pricing="nav_listed_holdings", style="active", expense_ratio=0.0075)
add("SYN_APPRAISED", fund("multi-asset", 3, smooth=(0.4, 0.3, 0.3)), region="Global", currency="USD",
    asset_class="multi-asset", pricing="appraised_nav", style="active", expense_ratio=0.012)
add("SYN_BOND", fund("bond", 4), region="US", currency="USD", asset_class="bond",
    pricing="exchange_traded", style="active")
add("SYN_CLOSET", closet, region="US", currency="USD", asset_class="equity",
    pricing="nav_listed_holdings", style="active", expense_ratio=0.0100)
add("SYN_SHORT", fund("equity", 6).iloc[-24:], region="US", currency="USD", asset_class="equity",
    pricing="nav_listed_holdings", style="active")
pd.DataFrame(rows).to_csv("sample_data/returns_sample.csv", index=False)
pd.DataFrame(meta).to_csv("sample_data/meta_sample.csv", index=False)

os.makedirs("sample_data/bbg_small", exist_ok=True)
shutil.copy("sample_data/returns_sample.csv", "sample_data/bbg_small/returns.csv")
m = pd.DataFrame(meta)
m["benchmark_id"] = "BMK_SYN"
m.to_csv("sample_data/bbg_small/meta.csv", index=False)
bench = F0["RF"] + F0["Mkt-RF"] + np.random.default_rng(99).normal(0, 0.004, len(F0))
pd.DataFrame(dict(date=bench.index.strftime("%Y-%m-%d"), benchmark_id="BMK_SYN",
                  value=bench.to_numpy(), kind="return")).to_csv("sample_data/bbg_small/benchmarks.csv", index=False)
pd.DataFrame(dict(field=["TOTAL_RETURN (SYNTHETIC EXAMPLE)"], retrieval_date=["1900-01-01"],
                  universe=["synthetic sample, not a real universe"])).to_csv(
    "sample_data/bbg_small/field_log.csv", index=False)
F0.to_csv("sample_data/bbg_small/factors_sample.csv", index_label="date")
print("written", len(rows), "rows,", len(meta), "funds")
