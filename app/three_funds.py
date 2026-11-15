"""Comparison card of three funds (US exchange-traded equity ETF, European equity fund,
appraised-NAV fund): a table (CSV) and a chart (PNG) for the report. Aggregated readings only:
no series, no dates. Selection is deterministic (first anonymised ID matching the slot, at least
60 months) unless the three IDs are given explicitly."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diagnostics as dg
import readings as rd
import universe as UN

SLOTS = [("US exchange-traded equity ETF", lambda p: p.region == "US" and p.asset_class == "equity" and p.pricing == "exchange_traded"),
         ("European equity fund", lambda p: p.region == "Europe" and p.asset_class == "equity"),
         ("Appraised-NAV fund", lambda p: p.pricing == "appraised_nav")]
COL = {rd.NOTHING: "#2e7d5b", rd.LOOK: "#c58a00", rd.UNUSUAL: "#b33a3a", rd.DESC: "#6b7785", rd.NA: "#9aa5b1"}
TITLES = dict(price_regularity="Price regularity", style="Style and alpha", alpha="Alpha over time",
              risk_shape="Shape of risk", distribution="Small gains and losses", peers="Position among peers")


def pick(U, ids=None, min_months=60):
    if ids:
        return list(zip([s[0] for s in SLOTS], ids))
    out = []
    for name, rule in SLOTS:
        c = [f for f in sorted(U.returns) if rule(U.profiles[f]) and len(U.returns[f]) >= min_months]
        c = [f for f in c if not (name.startswith("European") and U.profiles[f].currency != "EUR")] or c
        out.append((name, c[0] if c else None))
    return out


def build(U, ids=None):
    cards, missing = [], []
    for slot, fid in pick(U, ids):
        if fid is None:
            missing.append(slot)
            continue
        p = UN.analyse_fund(U, fid)
        own = dg.fund_stats(p.returns, p.factors)
        cards.append((slot, fid, p, own))
    return cards, missing


def table(cards):
    rows = {}
    for slot, fid, p, own in cards:
        pr = p.res.profile
        t = p.technical
        col = {"Slot": slot, "Fund": fid, "Region": pr.region, "Asset class": pr.asset_class,
               "Pricing type": pr.pricing, "Currency": pr.currency, "Months analysed": len(p.returns),
               "Factor set": p.res.factor_set.label if p.res.factor_set.usable else "none (factor test skipped)",
               "Peer set": (p.res.peer_selection.note if p.res.peer_selection else "none"),
               "Peer set size": p.res.peer_selection.n if p.res.peer_selection else 0,
               "Headline": p.headline}
        for x in p.readings:
            col[f"{TITLES.get(x.key, x.key)}"] = x.level + (f" ({x.tag})" if x.tag else "")
        col["Priority index (technical)"] = round(t["priority_index"], 1) if t else "n/a"
        col["Reliability (technical)"] = t.get("reliability", "n/a") if t else "n/a"
        col["Lag-1 autocorrelation (fund)"] = round(own["ac1"], 3)
        col["Lag-1 autocorrelation (peer 5th-95th pct)"] = (f"{p.peer_band_ac1[0]:.2f} to {p.peer_band_ac1[1]:.2f}"
                                                           if p.peer_band_ac1 else "no peer band")
        col["Number of warnings"] = len(p.notes)
        rows[fid] = col
    return pd.DataFrame(rows).T.reset_index(drop=True)


def chart(cards, path):
    n = len(cards)
    if n == 0:
        return False
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.7, 1]))
    keys = [k for k in rd_keys(cards)]
    for j, (slot, fid, p, own) in enumerate(cards):
        lv = {x.key: x for x in p.readings}
        for i, k in enumerate(keys):
            x = lv.get(k)
            lvl = x.level if x else rd.NA
            a.add_patch(plt.Rectangle((j, -i - 1), 0.94, 0.9, color=COL[lvl]))
            lab = {rd.NOTHING: "nothing\nunusual", rd.LOOK: "worth\na look", rd.UNUSUAL: "unusual",
                   rd.DESC: "descriptive", rd.NA: "not\napplicable"}[lvl]
            a.text(j + .47, -i - .55, lab + (f"\n[{x.tag}]" if x and x.tag else ""), ha="center", va="center",
                   color="white", fontsize=7.5, fontweight="bold")
    a.set_xlim(0, n); a.set_ylim(-len(keys) - .1, 0)
    a.set_xticks([j + .47 for j in range(n)])
    a.set_xticklabels([f"{c[1]}\n{c[0]}" for c in cards], fontsize=8)
    a.xaxis.tick_top()
    a.set_yticks([-i - .55 for i in range(len(keys))]); a.set_yticklabels([TITLES.get(k, k) for k in keys], fontsize=8.5)
    for s in a.spines.values():
        s.set_visible(False)
    a.tick_params(length=0)
    a.set_title("Reading status (role in brackets: primary or informational)", fontsize=9, loc="left", pad=28)
    for j, (slot, fid, p, own) in enumerate(cards):
        if p.peer_band_ac1:
            b.plot([p.peer_band_ac1[0], p.peer_band_ac1[1]], [j, j], color="#9ec5b0", lw=9, solid_capstyle="butt")
        b.plot(own["ac1"], j, "o", color="#1f4e79", ms=8)
    b.axvline(0, color="#999", lw=.8)
    b.set_yticks(range(n)); b.set_yticklabels([c[1] for c in cards], fontsize=8.5); b.invert_yaxis()
    b.set_xlabel("Lag-1 autocorrelation (green band: peers 5th-95th percentile)", fontsize=8)
    b.set_title("Month-to-month echo vs peers", fontsize=9, loc="left")
    for s in ("top", "right"):
        b.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "A flag is a signal to investigate, never proof. Aggregated readings only; no raw series.", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def rd_keys(cards):
    ks = []
    for *_, p, own in cards:
        ks += [x.key for x in p.readings if x.key not in ks]
    return [k for k in UN.READING_KEYS if k in ks]
