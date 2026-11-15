"""Score every fund in data/ and write results/real_funds_scores.csv.

  python score_funds.py                      # against the synthetic reference
  python score_funds.py --reference real     # against the real-fund reference,
                                             # each fund scored leave-one-out

Real-reference caveat: with N reference funds the smallest empirical p-value
is about 1/(N+1), so scores cannot go very low. Report both modes.
"""
from __future__ import annotations

import argparse
import os
import warnings

import pandas as pd

from fundtruth import FundTruth

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--reference", choices=["synthetic", "real"], default="synthetic")
    a = ap.parse_args()
    wide = pd.read_csv(f"{a.data}/fund_returns.csv", index_col="date", parse_dates=True)
    fac = pd.read_csv(f"{a.data}/factors.csv", index_col="date", parse_dates=True)
    meta = pd.read_csv(f"{a.data}/funds_meta.csv").set_index("ticker")
    ref = (pd.read_csv(f"{a.data}/reference_real_v1.csv", index_col="fund")
           if a.reference == "real" else None)
    rows = []
    for t in wide.columns:
        s = wide[t].dropna().rename(t)
        peers = ref.drop(index=t) if ref is not None else None
        ft = FundTruth(s, fac, peers=peers, category=meta.loc[t, "category"],
                       style=meta.loc[t, "style"])
        sc = ft.score()
        rows.append(dict(ticker=t, category=meta.loc[t, "category"],
                         style=meta.loc[t, "style"], n_obs=len(s),
                         score=sc["score"], reliability=sc["reliability"],
                         cap=sc["cap_applied"],
                         **{f"sub_{k}": v for k, v in sc["subscores"].items()},
                         flagged=", ".join(k for k, r in ft.all_tests().items() if r["flag"])))
    out = pd.DataFrame(rows).sort_values("score")
    os.makedirs("results", exist_ok=True)
    out.to_csv(f"results/real_funds_scores_{a.reference}.csv", index=False)
    print(out.round(1).to_string(index=False))
    print("\nA low score is a statistical signal to investigate, "
          "not proof of irregularity.")


if __name__ == "__main__":
    main()
