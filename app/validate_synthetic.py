"""Synthetic validation: power, false positives, +/-20% weight sensitivity.

Run: python validate_synthetic.py [n_seeds]
Writes results/validation_summary.csv, results/sensitivity.csv, results/scores.csv

Caveat (stated in the report): honest series come from the SAME generator that
built the synthetic reference, so false-positive rates are optimistic.
Real-fund false positives can only be assessed with real data.
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

from fundtruth import DEFAULT_WEIGHTS, FundTruth
from synthetic import honest_fund, synthetic_set

warnings.filterwarnings("ignore")
BANDS = {"high_concern": 50.0, "review": 70.0}   # PROVISIONAL score bands
TESTS4 = ["autocorrelation", "factor_regression", "near_zero", "peer_anomaly"]
CAT = {"a_smoothed": "multi-asset", "b_nearzero": "multi-asset",
       "c_closet": "equity", "d_control": "equity"}


def band(x):
    return ("high_concern" if x < BANDS["high_concern"]
            else "review" if x < BANDS["review"] else "ok")


def reweighted(base, key, factor):
    """Scale one weight by `factor` (+/-20%) and renormalise the rest to 100."""
    w = dict(base)
    w[key] *= factor
    tot = sum(w[k] for k in TESTS4)
    return {k: (100.0 * w[k] / tot if k in TESTS4 else 0.0) for k in w}


def parameter_grid(df):
    """Re-score stored p-values under alternative PROVISIONAL parameters
    (P_SCALE, cap offset, band cut). Shows the power / false-positive trade-off."""
    out = []
    for scale, cap in itertools.product((0.05, 0.10, 0.20), (20.0, 30.0, 40.0)):  # 0.10 = adopted default
        finals = []
        for _, row in df.iterrows():
            subs, ws = {}, {}
            for k in TESTS4:
                if row[f"st_{k}"] != "ok":
                    continue
                pv = row[f"p_{k}"]
                subs[k] = (row[f"sub_{k}"] if k == "peer_anomaly"
                           else 100.0 * min(1.0, pv / scale))
                ws[k] = DEFAULT_WEIGHTS[k]
            tot = sum(ws.values())
            avg = sum(ws[k] * subs[k] for k in subs) / tot
            finals.append(min(avg, min(subs.values()) + cap))
        d = df.assign(final=finals)
        for cut in (50.0, 60.0):
            hc = d.final < cut
            honest = d[d.label.str.startswith(("d_", "honest"))]
            out.append(dict(
                p_scale=scale, cap=cap, cut=cut,
                power_a=100 * hc[d.label == "a_smoothed"].mean(),
                power_b=100 * hc[d.label == "b_nearzero"].mean(),
                power_c=100 * hc[d.label == "c_closet"].mean(),
                fp_honest=100 * (honest.final < cut).mean()))
    g = pd.DataFrame(out)
    return g.round({c: 1 for c in g if c not in ("p_scale",)})


def main(n_seeds=300):
    rows = []
    for seed in range(1, n_seeds + 1):
        for label, (r, f) in synthetic_set(seed).items():
            rows.append((label, seed, FundTruth(r, f, category=CAT[label])))
        for cat in ("equity", "bond", "multi-asset"):
            r, f = honest_fund(cat, 7_000 + seed)
            rows.append((f"honest_{cat}", seed, FundTruth(r, f, category=cat)))
    recs, sens = [], []
    variants = {"base": (DEFAULT_WEIGHTS, None)}
    for k, fac in itertools.product(TESTS4, (0.8, 1.2)):
        variants[f"w_{k}_x{fac}"] = (reweighted(DEFAULT_WEIGHTS, k, fac), None)
    variants["cap_24"], variants["cap_36"] = (DEFAULT_WEIGHTS, 24.0), (DEFAULT_WEIGHTS, 36.0)
    for label, seed, ft in rows:
        base_band = None
        for vname, (w, cap) in variants.items():
            sc = ft.score(w, cap_offset=cap)
            b = band(sc["score"])
            if vname == "base":
                base_band = b
                recs.append(dict(label=label, seed=seed, score=sc["score"],
                                 band=b, cap=sc["cap_applied"], rel=sc["reliability"],
                                 **{f"sub_{k}": v for k, v in sc["subscores"].items()},
                                 **{f"flag_{k}": ft.all_tests()[k]["flag"] for k in TESTS4 + ["benford"]},
                                 **{f"p_{k}": ft.all_tests()[k]["p_value"] for k in TESTS4},
                                 **{f"st_{k}": ft.all_tests()[k]["status"] for k in TESTS4}))
            sens.append(dict(label=label, variant=vname, changed=b != base_band,
                             high=b == "high_concern", score=sc["score"]))
    df, sv = pd.DataFrame(recs), pd.DataFrame(sens)
    summ = df.groupby("label").agg(
        n=("score", "size"), mean_score=("score", "mean"),
        median_score=("score", "median"),
        pct_high_concern=("band", lambda s: 100 * (s == "high_concern").mean()),
        pct_review_or_worse=("band", lambda s: 100 * (s != "ok").mean()),
        pct_cap=("cap", lambda s: 100 * s.mean())).round(1)
    flags = df.groupby("label")[[c for c in df if c.startswith("flag_")]].mean().mul(100).round(1)
    sens_tab = (sv[sv.variant != "base"].groupby(["label", "variant"])
                .agg(pct_band_changed=("changed", lambda s: 100 * s.mean()),
                     pct_high_concern=("high", lambda s: 100 * s.mean())).round(1))
    grid = parameter_grid(df)
    os.makedirs("results", exist_ok=True)
    grid.to_csv("results/parameter_grid.csv", index=False)
    summ.to_csv("results/validation_summary.csv")
    flags.to_csv("results/flag_rates_per_test.csv")
    sens_tab.to_csv("results/sensitivity.csv")
    df.to_csv("results/scores.csv", index=False)
    print(f"Seeds: {n_seeds}. Bands (provisional): <50 high concern, <70 review.\n")
    print(summ.to_string(), "\n\nFlag rate per test (%), p<0.05 / peer<5th pct:")
    print(flags.to_string(), "\n\nSensitivity: max % of series changing band per variant")
    print(sens_tab.groupby("label")["pct_band_changed"].agg(["mean", "max"]).round(1).to_string())
    print("\nParameter grid (provisional parameters; % of series below the cut):")
    print(grid.to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
