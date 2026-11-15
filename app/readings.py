"""Evidence-based READINGS for one fund (presentation layer, nothing here feeds the score).

Each reading has: what we measured, what we found (numbers), the peer range, the
uncertainty, why it may be innocent, what to ask the manager, a 3-level status and the
measured false-positive rate on simulated honest funds, plus what is NOT validated.
Status uses the existing flag thresholds: p < FLAG_P -> 'unusual'; p < P_SCALE (the point
where the subscore starts to fall) -> 'worth a look'; else 'nothing unusual'.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import diagnostics as dg
from fund_profile import select_peers
from fundtruth import ALPHA_MIN_T, CLOSET_R2, FLAG_P, NEAR_ZERO_W, P_SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
UNUSUAL, LOOK, NOTHING = "unusual", "worth a look", "nothing unusual"
DESC, NA = "descriptive", "not applicable"
NOT_VALIDATED = {
    "price_regularity": "Detection is validated only on simulated smoothing; there are no labelled real cases. Honest illiquid funds may show the same pattern.",
    "style": "Closet indexing is validated only on simulated index-like series with a known expense ratio; without the expense ratio it is not detected.",
    "alpha": "Only simulated skill-free funds were used; the sub-period figures have no validation.",
    "risk_shape": "Descriptive only: no threshold has been validated, so no status is given.",
    "distribution": "Detection is validated only on simulated shifted losses; on real funds only false positives were audited (3 of 39 flagged, close to chance).",
    "peers": "The peer test depends on the reference set (frozen, but small); a different universe would give different percentiles.",
}
_FLAGCOL = {"price_regularity": "flag_autocorrelation", "style": "flag_factor_regression",
            "alpha": "flag_factor_regression", "distribution": "flag_near_zero",
            "peers": "flag_peer_anomaly"}


@dataclass
class Reading:
    key: str
    title: str
    level: str
    tag: str = ""                 # 'primary' | 'informational' | ''
    measured: str = ""
    found: str = ""
    peer_range: str = ""
    uncertainty: str = ""
    innocent: str = ""
    ask: str = ""
    reason: str = ""              # why not applicable / why informational
    fp_rate: float | None = None  # % of simulated honest funds flagged by this check
    not_validated: str = ""
    chart: str = ""


def level_from_p(p):
    if p is None or not np.isfinite(p):
        return NA
    return UNUSUAL if p < FLAG_P else LOOK if p < P_SCALE else NOTHING


def false_positive_rates(path=None):
    """Mean flag rate (%) of each check on the simulated HONEST series (results/flag_rates_per_test.csv)."""
    path = path or os.path.join(HERE, "results", "flag_rates_per_test.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, index_col=0)
    hon = [i for i in df.index if i.startswith(("d_", "honest"))]
    return {c: float(df.loc[hon, c].mean()) for c in df.columns}


def _peer_text(stats, col, value, fmt="{:.2f}"):
    if stats is None or col not in stats or stats[col].dropna().shape[0] < 5:
        return "No peer statistics loaded for this group."
    v = stats[col].dropna()
    q = dg.peer_percentile(value, v)
    f = fmt.format
    return (f"Peers (n={len(v)}): median {f(v.median())}, middle 90% {f(v.quantile(.05))} to "
            f"{f(v.quantile(.95))}; this fund {f(value)}"
            + (f", at the {q:.0%} percentile." if q == q else "."))


def peer_stats_for(profile, peer_stats, min_peers=20):
    """Peer statistics rows chosen by (asset class, region) with the same displayed fallback rule."""
    if peer_stats is None or peer_stats.empty:
        return None, "No peer statistics loaded."
    t = peer_stats.rename(columns={"asset_class": "category"}).drop(index=profile.fund_id, errors="ignore")
    sel = select_peers(t, profile, min_peers)
    return (sel.peers if len(sel.peers) else None), sel.note


def _status(res, key):
    r = res.engine.all_tests()[key]
    return r, r["status"] == "ok"


def build_readings(res, returns, factors=None, benchmark=None, peer_stats=None, market=None):
    """List of Reading for one analysed fund.

    res : context.ContextResult (engine already run)   peer_stats : diagnostics.build_peer_stats table
    benchmark : declared benchmark returns or None      market : regional market return series (Mkt-RF+RF)
    """
    p = res.profile
    fp = false_positive_rates()
    ps, peer_note = peer_stats_for(p, peer_stats)
    out = []
    if res.engine is None:
        return out
    tests = res.engine.all_tests()
    det = res.engine.details
    appraised = p.pricing == "appraised_nav"
    unknown_pr = p.pricing == "unknown"
    why_info = {"exchange_traded": "Exchange prices are set by the market, not by the manager.",
                "nav_listed_holdings": "The NAV follows the market prices of listed holdings, not manager estimates."}.get(p.pricing, "")

    # ---- 1. price regularity
    t = tests["autocorrelation"]
    r = Reading("price_regularity", "Price regularity", NA, chart="autocorr",
                fp_rate=fp.get("flag_autocorrelation"), not_validated=NOT_VALIDATED["price_regularity"])
    if t["status"] != "ok":
        r.reason = t["explanation"]
    else:
        d = det["autocorrelation"]
        ac = d["ac"]
        sm = dg.desmoothed_vol(returns, factors)
        r.level = level_from_p(t["p_value"])
        r.tag = "primary" if appraised else "informational"
        r.reason = ("Valuations can be estimates, so serial correlation is the main signal." if appraised
                    else (why_info + " There is little room for smoothing, so read this as information."
                          if why_info else "Pricing basis unknown: relevance cannot be judged."))
        r.measured = (f"Correlation of each month's return with the previous 1, 2 and 3 months, on "
                      f"{d['basis']}; reported against de-smoothed volatility.")
        r.found = (f"{ac[0]:+.2f}, {ac[1]:+.2f}, {ac[2]:+.2f} (Ljung-Box p = {d['p_lb']:.3f}; vs peers p = "
                   f"{d['p_emp']:.3f}). Reported volatility {sm['reported']:.1%} a year; a simple "
                   f"de-smoothing gives {sm['desmoothed']:.1%} (x{sm['ratio']:.2f}).")
        r.peer_range = _peer_text(ps, "ac1", ac[0], "{:+.2f}")
        r.uncertainty = (f"{t['n_obs']} months: a random series stays within about +/-{1.96 / math.sqrt(t['n_obs']):.2f}. "
                         "De-smoothing assumes all the echo is smoothing (limit: not the full Getmansky-Lo-Makarov fit).")
        r.innocent = "Slow-moving holdings, carry in short-term bond funds, or genuinely persistent markets can all create an echo."
        r.ask = "Valuation policy, who prices the assets, how often, and whether stale prices are adjusted."

    # ---- 2. style and benchmark
    t = tests["factor_regression"]
    fl = dg.factor_loadings(returns, factors)
    st = Reading("style", "Style and benchmark", NA, chart="loadings",
                 fp_rate=fp.get("flag_factor_regression"), not_validated=NOT_VALIDATED["style"])
    al = Reading("alpha", "Return beyond the market", NA, chart="alpha",
                 fp_rate=fp.get("flag_factor_regression"), not_validated=NOT_VALIDATED["alpha"])
    if t["status"] != "ok" or fl is None:
        why = t["explanation"] if t["status"] != "ok" else "Regression not defined for this series."
        st.reason = al.reason = why
    else:
        state, basis = res.index_state
        closet_driven = res.engine.style == "active" and fl["r2"] >= CLOSET_R2 and t["flag"]
        st.level = UNUSUAL if closet_driven else (LOOK if (t["p_value"] is not None and t["p_value"] < P_SCALE
                                                           and fl["r2"] >= CLOSET_R2) else NOTHING)
        st.tag = "primary"
        st.measured = (f"Betas on the {res.factor_set.label} (market, size, value) with 95% intervals, R2 and tracking error"
                       + (", and the fit to the declared benchmark." if res.bench_fit else "."))
        bm = (f" Against the declared benchmark index: R2 {res.bench_fit['r2']:.2f}, tracking error {res.bench_fit['te']:.1%}."
              if res.bench_fit and res.bench_fit["r2"] == res.bench_fit["r2"] else "")
        st.found = (f"Market beta {fl['beta_Mkt-RF']:.2f} [{fl['lo_Mkt-RF']:.2f}, {fl['hi_Mkt-RF']:.2f}], size "
                    f"{fl['beta_SMB']:+.2f}, value {fl['beta_HML']:+.2f}; R2 {fl['r2']:.2f}, tracking error against the factor model "
                    f"{fl['te']:.1%} a year.{bm}")
        st.peer_range = _peer_text(ps, "r2", fl["r2"])
        st.uncertainty = f"{fl['n']} months; intervals are classical OLS."
        st.reason = res.closet_reason + (" Equity factors explain little of bond returns." if p.asset_class == "bond" else "")
        st.innocent = "Index funds and low-cost core funds are meant to track a benchmark closely; it is only odd when fees are active."
        st.ask = "The declared benchmark, the fee level and the stated active share or mandate."
        al.level = (UNUSUAL if (t["flag"] and fl["alpha_t"] >= ALPHA_MIN_T and not closet_driven)
                    else LOOK if fl["alpha_t"] >= ALPHA_MIN_T else NOTHING)
        sub = dg.alpha_subperiods(returns, factors)
        subt = "; ".join(f"{k}: " + (f"{v['alpha']:+.1%} (t {v['alpha_t']:.1f}, {v['n']} months)" if v["alpha"] == v["alpha"]
                                    else f"not enough data ({v['n']} months)") for k, v in sub.items())
        al.tag = "primary"
        al.measured = "Annualised alpha from the three-factor model, its t-statistic, and the same model in sub-periods."
        al.found = f"Alpha {fl['alpha']:+.1%} a year (t = {fl['alpha_t']:.1f}). Sub-periods: {subt}."
        al.peer_range = _peer_text(ps, "alpha", fl["alpha"], "{:+.1%}")
        al.uncertainty = f"{fl['n']} months; a t-statistic below 2 means the alpha is not distinguishable from zero. Sub-periods are short."
        al.innocent = "Genuine skill, a factor the model omits (e.g. quality or momentum), or fee and currency effects."
        al.ask = "The source of the excess return, capacity, and whether it persists across market regimes."
        al.reason = "Negative alpha is not treated as an integrity issue."
        if p.currency not in ("USD", "unknown"):
            al.innocent += f" The fund is in {p.currency} and the factors are in {res.factor_set.currency}: currency moves enter alpha."
    out += [r, st, al]

    # ---- 4. risk shape (descriptive)
    rs = dg.risk_shape(returns)
    mk = market if market is not None else (factors["Mkt-RF"] + factors["RF"] if factors is not None and "RF" in factors else None)
    cap = dg.capture_ratios(returns, mk) if mk is not None else dict(up=np.nan, down=np.nan)
    risk = Reading("risk_shape", "Shape of risk", DESC, chart="drawdown", fp_rate=None,
                   not_validated=NOT_VALIDATED["risk_shape"],
                   measured="Worst peak-to-trough loss, skewness, excess kurtosis, and up/down capture against the regional market.",
                   found=(f"Maximum drawdown {rs['max_drawdown']:.1%}; skewness {rs['skew']:+.2f}; excess kurtosis {rs['excess_kurtosis']:+.2f}"
                          + (f"; up capture {cap['up']:.0%}, down capture {cap['down']:.0%}." if cap["up"] == cap["up"] else ".")),
                   peer_range=_peer_text(ps, "max_drawdown", rs["max_drawdown"], "{:.1%}"),
                   uncertainty=f"{len(returns)} months: tail measures are noisy on short samples.",
                   innocent="A calm period or a low-risk mandate; short samples hide tail events.",
                   ask="Risk limits, stress results and how losses were handled.",
                   reason="No validated threshold: shown for context, with no status.")
    out.append(risk)

    # ---- 5. return distribution / near-zero
    t = tests["near_zero"]
    nz = Reading("distribution", "Small gains and losses", NA, chart="hist",
                 fp_rate=fp.get("flag_near_zero"), not_validated=NOT_VALIDATED["distribution"])
    if t["status"] != "ok":
        nz.reason = t["explanation"]
    else:
        d = det["near_zero"]
        n = d["n_loss"] + d["n_gain"]
        nz.level = level_from_p(t["p_value"])
        nz.tag = "primary" if (appraised or unknown_pr) else "informational"
        nz.reason = ("Reported valuations can drift above zero, so this check applies." if appraised else
                     (why_info + " Shifting small losses above zero is not a plausible mechanism, so a flag is more likely chance."
                      if why_info else "Pricing basis unknown: relevance cannot be judged."))
        nz.measured = f"Months with a return within +/-{NEAR_ZERO_W:.1%} of zero: small losses against small gains."
        nz.found = (f"{d['n_loss']} small losses and {d['n_gain']} small gains ({d['n_gain'] / max(n, 1):.0%} gains, "
                    f"{d['p0']:.0%} expected from the fund's own distribution; p = {t['p_value']:.3f}).")
        nz.uncertainty = f"Only {n} months fall in the window: low power."
        nz.innocent = "Chance, or a distribution that is not smooth near zero."
        nz.ask = "How the monthly figures are produced, and whether losses are rounded or smoothed."
    out.append(nz)

    # ---- 6. peers
    t = tests["peer_anomaly"]
    pr = Reading("peers", "Position among peers", NA, chart="peer_bars", fp_rate=fp.get("flag_peer_anomaly"),
                 not_validated=NOT_VALIDATED["peers"])
    if t["status"] != "ok":
        pr.reason = t["explanation"]
    elif not res.reference_adequate:
        pr.reason = f"Reference has only {res.reference_size} funds: peer positions are not meaningful."
    else:
        pr.level = level_from_p(t["p_value"])
        pr.tag = "primary"
        pr.measured = "Share of similar funds that look at least as unusual on volatility, skewness, echo, alpha and R2 together; and percentiles on single measures."
        own = dg.fund_stats(returns, factors)
        pct = ", ".join(f"{lab} {dg.peer_percentile(own[c], ps[c]) :.0%}" for lab, c in
                        (("volatility", "vol"), ("Sharpe", "sharpe"), ("R2", "r2"), ("echo", "ac1"))
                        if ps is not None and c in ps and dg.peer_percentile(own[c], ps[c]) == dg.peer_percentile(own[c], ps[c]))
        pr.found = f"{t['p_value']:.0%} of reference funds look at least this unusual. Percentiles: {pct or 'not available'}."
        pr.peer_range = res.peer_selection.note if res.peer_selection else peer_note
        pr.uncertainty = f"Reference of {res.reference_size} funds; the smallest possible p is about 1/{res.reference_size + 1}."
        pr.innocent = "A genuinely different strategy or a category that the reference covers poorly."
        pr.ask = "How the fund differs from its stated peer group, and why."
    out.append(pr)
    order = ["price_regularity", "style", "alpha", "risk_shape", "distribution", "peers"]
    return sorted(out, key=lambda x: order.index(x.key))


def headline(readings):
    n_u = sum(r.level == UNUSUAL and r.tag == "primary" for r in readings)
    n_l = sum(r.level == LOOK and r.tag == "primary" for r in readings)
    n_i = sum(r.level in (UNUSUAL, LOOK) and r.tag == "informational" for r in readings)
    if n_u:
        h = f"{n_u} unusual reading" + ("s" if n_u > 1 else "") + (f" and {n_l} worth a look" if n_l else "")
    elif n_l:
        h = f"{n_l} reading" + ("s" if n_l > 1 else "") + " worth a look, nothing unusual"
    else:
        h = "Nothing unusual in the checks that apply"
    if n_i:
        h += f" ({n_i} informational reading" + ("s" if n_i > 1 else "") + " not counted)"
    return h
