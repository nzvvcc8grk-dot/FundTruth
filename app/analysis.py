"""Real-fund diagnostics used in the report (all numbers computed from data/).

near_zero_audit : calibration check of the near-zero test on real funds.
carry_check     : autocorrelation of raw vs excess (over RF) returns for one fund.
weight_scenarios: effect of alternative weights / cap rules on synthetic and real scores.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from features import autocorrelations, ljung_box_q
from fundtruth import FundTruth, NEAR_ZERO_W

warnings.filterwarnings("ignore")
K = ["autocorrelation", "factor_regression", "near_zero", "peer_anomaly"]


def near_zero_audit(w, meta, factors):
    """Per-fund near-zero results plus pooled checks.

    The pooled z treats funds as independent and overstates the evidence
    (funds share the same months); the month-clustered t is the one to quote.
    """
    rows = []
    gain_ex = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    for t in w.columns:
        ft = FundTruth(w[t].rename(t), factors, category=meta.loc[t, "category"],
                       style=meta.loc[t, "style"])
        res = ft.test_near_zero()
        if res["status"] != "ok":
            rows.append(dict(ticker=t, status=res["status"]))
            continue
        d = ft.details["near_zero"]
        x = w[t]
        gain = ((x >= 0) & (x < NEAR_ZERO_W)).astype(float)
        loss = ((x >= -NEAR_ZERO_W) & (x < 0)).astype(float)
        gain_ex[t] = gain - d["p0"] * (gain + loss)
        rows.append(dict(ticker=t, status="ok", n_loss=d["n_loss"], n_gain=d["n_gain"],
                         p0=d["p0"], p=res["p_value"]))
    df = pd.DataFrame(rows)
    ok = df[df.status == "ok"]
    n = ok.n_loss + ok.n_gain
    exp_share = float((n * ok.p0).sum() / n.sum())
    tot = gain_ex.sum(axis=1)
    t_cl = float(tot.mean() / (tot.std(ddof=1) / np.sqrt(len(tot))))
    flagged = int((ok.p < 0.05).sum())
    return dict(
        table=df, applicable=len(ok), flagged=flagged,
        expected_flagged=0.05 * len(ok),
        p_at_least=float(stats.binom.sf(flagged - 1, len(ok), 0.05)),
        median_in_window=float(n.median()),
        share_gains=float(ok.n_gain.sum() / n.sum()), expected_share=exp_share,
        t_clustered=t_cl, p_clustered=float(stats.t.sf(t_cl, len(tot) - 1)))


def carry_check(w, factors, ticker):
    """Lag 1-3 autocorrelation and Ljung-Box p, raw vs excess over RF."""
    out = {}
    for lab, s in (("raw", w[ticker]), ("excess", w[ticker] - factors["RF"].reindex(w.index))):
        x = s.to_numpy()
        q, p = ljung_box_q(x, 3)
        out[lab] = dict(rho=autocorrelations(x, 3), q=q, p=p)
    return out


def _final(subs, weights, cap, nz_in_cap):
    w = np.asarray(weights, float)
    ok = ~np.isnan(subs)
    ww = w * ok
    if ww.sum() == 0 or w[~ok].sum() > 100 / 3 + 1e-9:
        return np.nan
    avg = (ww * np.nan_to_num(subs)).sum() / ww.sum()
    mask = ok & (w > 0) & np.array([True, True, nz_in_cap, True])
    worst = subs[mask].min() if mask.any() else 100.0
    return min(avg, worst + cap)


SCENARIOS = {
    "Current: 30/30/20/20": ((30, 30, 20, 20), True),
    "Near-zero 15 (35/30/15/20)": ((35, 30, 15, 20), True),
    "Near-zero 10 (35/35/10/20)": ((35, 35, 10, 20), True),
    "Near-zero not in cap (30/30/20/20)": ((30, 30, 20, 20), False),
    "Near-zero weight 0 (40/35/0/25)": ((40, 35, 0, 25), False),
}


def weight_scenarios(synth, real, tickers=("SPY", "FBNDX", "AOM"), cap=30.0):
    """Detection of series (b), honest false positives, and scores of selected real funds."""
    rows = []
    cols = [f"sub_{k}" for k in K]
    for name, (w, nzc) in SCENARIOS.items():
        fs = np.array([_final(r, w, cap, nzc) for r in synth[cols].to_numpy(float)])
        fr = np.array([_final(r, w, cap, nzc) for r in real[cols].to_numpy(float)])
        s = synth.assign(f=fs)
        hon = s[s.label.str.startswith(("d_", "honest"))]
        row = {"Scenario": name,
               "Detect (b)": 100 * (s[s.label == "b_nearzero"].f < 50).mean(),
               "Honest below 50": 100 * (hon.f < 50).mean()}
        for t in tickers:
            row[t] = float(fr[real.ticker.to_numpy() == t][0])
        rows.append(row)
    return pd.DataFrame(rows)


def power_curve(n_seeds=200, strengths=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
                real_ref=None):
    """Sensitivity to graded smoothing, with and without the significance gate.

    Reported return = (1-s) r_t + s/2 r_{t-1} + s/2 r_{t-2} (s=0.6 is series (a)).
    Columns: share of series (in %) where the autocorrelation test flags with the
    gate (current), without the gate (empirical p only), and where the final
    score is below 50 (current). Also returns the mean lag-1 autocorrelation.
    """
    from synthetic import synthetic_reference, smoothed_fund
    refs = {"synthetic reference (1000/category)": synthetic_reference()}
    if real_ref is not None:
        refs["real reference (40 funds)"] = real_ref
    rows = []
    for rname, ref in refs.items():
        for s in strengths:
            th = (1 - s, s / 2, s / 2)
            fl_gate = fl_nogate = low = rev = 0
            ac1 = []
            for seed in range(n_seeds):
                r, f = smoothed_fund(seed, theta=th)
                ft = FundTruth(r, f, peers=ref, category="multi-asset")
                out = ft.test_autocorrelation()
                d = ft.details.get("autocorrelation", {})
                fl_gate += bool(out["flag"])
                fl_nogate += d.get("p_emp", 1.0) < 0.05
                sc_ = ft.score()["score"]
                low += sc_ < 50
                rev += sc_ < 70
                ac1.append(d["ac"][0])
            rows.append(dict(reference=rname, strength=s, mean_ac1=np.mean(ac1),
                             flag_gated=100 * fl_gate / n_seeds,
                             flag_ungated=100 * fl_nogate / n_seeds,
                             score_below_50=100 * low / n_seeds,
                             score_below_70=100 * rev / n_seeds))
    return pd.DataFrame(rows)
