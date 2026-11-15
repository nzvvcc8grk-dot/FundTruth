"""Validation of the derived closet-indexing rule on synthetic data (series (c)).

Scenarios for the closet series: expense ratio unknown (check informational),
below peer median (index-fund-like, informational), above peer median (flag allowed).
Control: honest equity series. The synthetic reference gets a constant peer
expense ratio of 0.60% (an ASSUMPTION of this exercise, not an observed value).
"""
import sys
import warnings
import numpy as np
import pandas as pd
from context import analyse
from fund_profile import build_profile
from synthetic import closet_fund, control_fund, synthetic_reference

warnings.filterwarnings("ignore")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ref = synthetic_reference(n_per_category=300).copy()
ref["expense_ratio"] = 0.006
rows = []
for label, fn, er, usebm in [("c_closet, expense unknown", closet_fund, None, False),
                      ("c_closet, expense 0.30% (below median)", closet_fund, 0.003, False),
                      ("c_closet, expense 1.00% (above median)", closet_fund, 0.010, False),
                      ("d_control (honest), expense 1.00%", control_fund, 0.010, False),
                      ("c_closet, expense 1.00%, declared benchmark = market", closet_fund, 0.010, True),
                      ("c_closet, expense unknown, declared benchmark = market", closet_fund, None, True),
                      ("d_control (honest), expense 1.00%, declared benchmark = market", control_fund, 0.010, True)]:
    low = flag = 0
    for seed in range(N):
        r, f = fn(seed)
        p = build_profile(f"X{seed}", dict(region="US", currency="USD", asset_class="equity",
                                           pricing="exchange_traded", expense_ratio=er))
        bm = (f["RF"] + f["Mkt-RF"]) if usebm else None
        res = analyse(r, p, {"US": f}, ref, benchmark=bm)
        low += res.score["score"] < 50
        flag += res.engine.test_factor_regression()["flag"]
    rows.append(dict(scenario=label, n=N, factor_test_flag_pct=100 * flag / N,
                     score_below_50_pct=100 * low / N))
out = pd.DataFrame(rows)
out.to_csv("results/closet_rule_validation.csv", index=False)
print(out.round(1).to_string(index=False))
