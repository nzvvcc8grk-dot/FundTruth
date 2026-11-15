"""Panel: everything the app shows for one fund, built without Streamlit (testable).

Order of work: context.analyse (frozen engine, factors and peers chosen by the profile)
-> readings (evidence cards) -> chart data. The 0-100 score is computed by the engine and
kept in `technical`, labelled as a priority index; it is not the headline.
"""
from __future__ import annotations

import html
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import diagnostics as dg
import readings as rd
from context import analyse
from fund_profile import factor_set_for

PRIORITY_LABEL = "priority index, not calibrated against known cases"
SCREENING = ("FundTruth is a screening tool: it shows where to look, never who is at fault. "
             "There are no labelled fraud cases, so no check has been validated on real "
             "misreporting; a signal is a reason to ask questions, never proof.")


@dataclass
class Panel:
    res: object
    readings: list
    headline: str
    ctx: str
    returns: pd.Series
    factors: pd.DataFrame | None
    benchmark: pd.Series | None
    benchmark_label: str
    market: pd.Series | None
    peer_median_returns: pd.Series | None
    peer_band_ac1: tuple | None
    peer_loadings: dict | None
    peer_pcts: dict
    notes: list = field(default_factory=list)
    reference_label: str = ""
    technical: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)


def analyse_panel(returns, profile, factors_by_region, reference, reference_label="",
                  benchmark=None, benchmark_label="", peer_stats=None, peer_returns=None,
                  category_label=None, period=None):
    r = returns.rename(profile.fund_id).dropna()
    if period:
        r = r.loc[period[0]:period[1]]
    fs = factor_set_for(profile)
    factors = factors_by_region.get(fs.key) if fs.usable else None
    bench = benchmark.reindex(r.index).dropna() if benchmark is not None and len(r) else None
    res = analyse(r, profile, factors_by_region, reference, benchmark=bench,
                  category_label=category_label)
    market = (factors["Mkt-RF"] + factors["RF"]).reindex(r.index) if factors is not None and "RF" in factors else None
    ps, _ = rd.peer_stats_for(profile, peer_stats)
    readings = rd.build_readings(res, r, factors, bench, peer_stats, market) if res.engine is not None else []
    first, last = r.index[0], r.index[-1]
    ctx = (f"{profile.region} | {fs.label if factors is not None else 'no market factors for this region in the built-in data (US only)'} | "
           f"{first:%b %Y} to {last:%b %Y} | returns in {profile.currency}")
    band = (float(ps["ac1"].quantile(.05)), float(ps["ac1"].quantile(.95))) if ps is not None and ps["ac1"].notna().sum() >= 5 else None
    pl = None
    if ps is not None and {"beta_mkt", "beta_smb", "beta_hml"} <= set(ps.columns):
        pl = {"Mkt-RF": float(ps.beta_mkt.median()), "SMB": float(ps.beta_smb.median()), "HML": float(ps.beta_hml.median())}
    own = dg.fund_stats(r, factors)
    pcts = {lab: dg.peer_percentile(own[c], ps[c]) for lab, c in
            (("Volatility", "vol"), ("Sharpe ratio", "sharpe"), ("R2 vs factors", "r2"), ("Month-to-month echo", "ac1"))
            if ps is not None and c in ps}
    pmed = None
    if peer_returns is not None and len(peer_returns.columns) >= 5:
        pmed = peer_returns.reindex(r.index).median(axis=1).dropna()
    notes = list(res.notes)
    if bench is None or len(bench) < 24:
        notes.append("No benchmark series: the growth chart uses the regional market factor"
                     + (" (not the fund's own benchmark)." if market is not None else "; not available."))
    tech = {}
    if res.score is not None:
        tech = dict(priority_index=res.score["score"], label=PRIORITY_LABEL,
                    reliability=res.score["reliability"], subscores=res.score["subscores"],
                    weights=res.score["weights_used"], cap_applied=res.score["cap_applied"],
                    report=res.engine.report())
    fpr = rd.false_positive_rates()
    q = dict(months=len(r), first=str(first.date()), last=str(last.date()),
             warnings=list(res.engine.warnings) if res.engine is not None else [],
             reference_size=res.reference_size, reference_adequate=res.reference_adequate,
             factor_currency=fs.currency, fp_rates=fpr)
    return Panel(res, readings, rd.headline(readings) if readings else "No readings: asset class not scored",
                 ctx, r, factors, bench, benchmark_label or ("regional market factor" if market is not None else ""),
                 market, pmed, band, pl, pcts, notes, reference_label, tech, q)


def panel_to_frame(p):
    """One row per reading (CSV export)."""
    return pd.DataFrame([dict(fund=p.res.profile.fund_id, reading=x.title, status=x.level, role=x.tag,
                              measured=x.measured, found=x.found, peers=x.peer_range,
                              uncertainty=x.uncertainty, why_it_may_be_innocent=x.innocent,
                              ask_the_manager=x.ask, applicability=x.reason,
                              false_positive_pct_on_simulated_honest=x.fp_rate,
                              not_validated=x.not_validated) for x in p.readings])


def panel_to_html(p):
    """Self-contained printable page (use the browser's Print to PDF)."""
    e = html.escape
    parts = [f"<h1>FundTruth: {e(p.res.profile.fund_id)}</h1><p>{e(p.ctx)}</p><h2>{e(p.headline)}</h2>",
             f"<p><em>{e(SCREENING)}</em></p>"]
    for x in p.readings:
        parts.append(f"<h3>{e(x.title)}: {e(x.level)}{' (' + e(x.tag) + ')' if x.tag else ''}</h3>")
        for lab, v in (("Measured", x.measured), ("Found", x.found), ("Peers", x.peer_range),
                       ("Uncertainty", x.uncertainty), ("Why it may be innocent", x.innocent),
                       ("What to ask the manager", x.ask), ("Applicability", x.reason),
                       ("Not validated", x.not_validated)):
            if v:
                parts.append(f"<p><b>{lab}.</b> {e(v)}</p>")
    parts.append("<h3>Notes</h3><ul>" + "".join(f"<li>{e(n)}</li>" for n in p.notes) + "</ul>")
    return "<html><meta charset='utf-8'><body style='font-family:sans-serif;max-width:900px'>" + "".join(parts) + "</body></html>"
