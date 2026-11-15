"""Negative-control benchmark under the app's own rule.

Honest simulated funds (equity, bond, multi-asset; mutual-fund and ETF pricing) are run through exactly the
pipeline used for real funds: same frozen reference set (which is pooled across categories while every
category has fewer than 20 funds), same readings and same question rule as the fund cards. The share that
reaches 'unusual' is the like-for-like comparison for the 0 of 40 real liquid funds that reach it.
('Worth a look' rates depend strongly on how each fund type is simulated, so they are reported, not compared.)

Run: python negative_control.py [funds per cell]      -> results/negative_control.csv
"""
import os
import sys
import warnings

import pandas as pd

import fund_report as fr
import panel as pn
from fund_profile import build_profile, factor_set_for
from synthetic import honest_fund

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))


def run(n=50, months=129):
    ctx = fr.load_context()
    UV = ctx["UV"]
    rows = []
    for cat in ("equity", "bond", "multi-asset"):
        for pricing in ("nav_listed_holdings", "exchange_traded"):
            for i in range(n):
                r, f = honest_fund(cat, 7_000_000 + i + 1000 * len(rows), months)
                prof = build_profile(f"H{i}", dict(region="US", currency="USD", asset_class=cat, pricing=pricing))
                p = pn.analyse_panel(r, prof, {factor_set_for(prof).key: f}, UV.reference, UV.reference_label,
                                     None, "", UV.peer_stats, None)
                q = fr.questions(p, None)
                rows.append(dict(category=cat, pricing=pricing, question=len(q) > 0,
                                 unusual=any(x.level == "unusual" and x.tag != "informational" for x in p.readings)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = run(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    s = d.groupby(["category", "pricing"])[["question", "unusual"]].mean().mul(100).round(1).reset_index()
    s.loc[len(s)] = ["all", "all", round(100 * d.question.mean(), 1), round(100 * d.unusual.mean(), 1)]
    s["funds"] = [int(((d.category == c) & (d.pricing == p)).sum()) if c != "all" else len(d)
                  for c, p in zip(s.category, s.pricing)]
    s.rename(columns={"question": "question_pct", "unusual": "unusual_pct"}).to_csv(
        os.path.join(HERE, "results", "negative_control.csv"), index=False)
    print(s.to_string(index=False))
