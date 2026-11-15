"""Learning curve: how the checks improve as the comparison set grows.

Every fund a user uploads can join the comparison (reference) set. This experiment measures what that is
worth, on simulated funds with known truth: for reference sets of 10, 20, 40, 80 and 160 honest funds it
records how often the checks find smoothed prices (strong and mild Getmansky-Lo-Makarov smoothing) and how
often they raise a question about an honest fund. The same test funds are used for every size, and each
smaller reference set is nested in the larger one, so only the size changes. The engine is used unchanged.

Run: python learning_curve.py [repetitions]      -> results/learning_curve.csv, results/learning_curve.png
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from features import compute_features
from fundtruth import FundTruth
from synthetic import honest_fund, smoothed_fund

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (10, 20, 40, 80, 160)
N_OBS = 96                                   # 8 years of monthly returns
N_TEST = 30                                  # test funds of each kind per repetition
MILD = (0.70, 0.20, 0.10)                    # mild smoothing; the default (0.4, 0.3, 0.3) is strong
TESTS = ("autocorrelation", "factor_regression", "near_zero", "peer_anomaly")
QUESTION_P, FLAG_P = 0.10, 0.05              # 'worth a look' and 'unusual' in the app


def reference(rep, n):
    rows = []
    for i in range(n):
        r, f = honest_fund("multi-asset", 500_000_000 + rep * 1_000 + i, N_OBS)
        x = compute_features(r, f)
        x.update(category="multi-asset", style="active")
        rows.append(x)
    df = pd.DataFrame(rows)
    df.index = [f"REF{rep}_{i}" for i in range(n)]
    return df


def pvalues(series, factors, ref):
    ft = FundTruth(series, factors, peers=ref, category="multi-asset", style="active")
    out = {}
    for t in TESTS:
        res = getattr(ft, f"test_{t}")()
        out[t] = res["p_value"] if res["status"] == "ok" and res["p_value"] is not None else np.nan
    return out


def run(reps=20):
    rows = []
    for rep in range(reps):
        full = reference(rep, max(SIZES))
        tests = []
        for j in range(N_TEST):
            s = 900_000_000 + rep * 1_000 + j
            tests.append(("honest",) + honest_fund("multi-asset", s, N_OBS))
            tests.append(("strong smoothing",) + smoothed_fund(s + 300, N_OBS))
            tests.append(("mild smoothing",) + smoothed_fund(s + 600, N_OBS, theta=MILD))
        for n in SIZES:
            ref = full.iloc[:n]
            for kind, r, f in tests:
                p = pvalues(r, f, ref)
                rows.append(dict(rep=rep, n_ref=n, kind=kind, **{f"p_{t}": p[t] for t in TESTS}))
        print(f"repetition {rep + 1}/{reps} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    out = []
    for (n, kind), g in df.groupby(["n_ref", "kind"]):
        ac = g["p_autocorrelation"]
        anyq = g[[f"p_{t}" for t in TESTS]].lt(QUESTION_P).any(axis=1)
        anyf = g[[f"p_{t}" for t in TESTS]].lt(FLAG_P).any(axis=1)
        peer_na = g["p_peer_anomaly"].isna().mean()
        k = len(g)
        out.append(dict(n_ref=n, kind=kind, funds=k,
                        smoothing_question_pct=100 * ac.lt(QUESTION_P).mean(),
                        smoothing_unusual_pct=100 * ac.lt(FLAG_P).mean(),
                        any_question_pct=100 * anyq.mean(), any_unusual_pct=100 * anyf.mean(),
                        peer_check_available_pct=100 * (1 - peer_na),
                        min_possible_p_vs_peers=1 / (n + 1)))
    s = pd.DataFrame(out)
    # 95% interval of a proportion (Wilson), for the two headline rates
    for col in ("smoothing_question_pct", "smoothing_unusual_pct", "any_question_pct"):
        p, k = s[col] / 100, s["funds"]
        z = 1.96
        c = (p + z * z / (2 * k)) / (1 + z * z / k)
        h = z * np.sqrt(p * (1 - p) / k + z * z / (4 * k * k)) / (1 + z * z / k)
        s[col + "_lo"], s[col + "_hi"] = 100 * (c - h), 100 * (c + h)
    return s


def chart(s, path, today=(10, 14, 16)):
    """Price-regularity check at the 'unusual' level (p < 0.05): found vs false alarms, by comparison-set size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.4, 3.9), dpi=200)
    ax.axvspan(min(today) * 0.93, max(today) * 1.07, color="#EDF0F3", zorder=0)
    ax.text(np.sqrt(min(today) * max(today)), 99, "same-type groups\ntoday: 10 to 16\n(so all 40 are\nused together)", ha="center",
            va="top", fontsize=8, color="#5F6B7A")
    styles = {"strong smoothing": ("#1E4D78", "Strong smoothing flagged"),
              "mild smoothing": ("#6E9BC5", "Mild smoothing flagged"),
              "honest": ("#A86A12", "Honest funds flagged by mistake")}
    col = "smoothing_unusual_pct"
    for kind, (c, lab) in styles.items():
        g = s[s.kind == kind].sort_values("n_ref")
        ax.fill_between(g["n_ref"], g[col + "_lo"], g[col + "_hi"], color=c, alpha=.13, linewidth=0)
        ax.plot(g["n_ref"], g[col], "-o", color=c, lw=2, ms=4)
        y = g[col].iloc[-1]
        ax.annotate(f"{lab}: {y:.1f}%" if y < 10 else f"{lab}: {y:.0f}%", (g["n_ref"].iloc[-1], y), xytext=(8, 0), textcoords="offset points",
                    va="center", color=c, fontsize=8.5)
    ax.annotate("with 10 funds the smallest\npossible p-value is 1/11 = 0.09:\nnothing can reach 'Unusual' (p < 0.05)",
                (10, 0), xytext=(24, 22), fontsize=8, color="#14213D",
                arrowprops=dict(arrowstyle="-", color="#5F6B7A", lw=.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.set_xlim(8.5, 420)
    ax.set_xlabel("Funds of the same type in the comparison set (log scale)")
    ax.set_ylabel("% of simulated funds flagged 'Unusual'")
    ax.set_ylim(-3, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DCE2E4", lw=.8)
    ax.set_title("More comparable funds, more smoothing found: the gain stops at about 40",
                 loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chart":            # redraw from the saved raw results
        s = summarise(pd.read_csv(os.path.join(HERE, "results", "learning_curve_raw.csv")))
        s.to_csv(os.path.join(HERE, "results", "learning_curve.csv"), index=False)
        chart(s, os.path.join(HERE, "results", "learning_curve.png"))
        sys.exit(0)
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    raw = run(reps)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    raw.to_csv(os.path.join(HERE, "results", "learning_curve_raw.csv"), index=False)
    s = summarise(raw)
    s.to_csv(os.path.join(HERE, "results", "learning_curve.csv"), index=False)
    chart(s, os.path.join(HERE, "results", "learning_curve.png"))
    print(s.round(1).to_string(index=False))
