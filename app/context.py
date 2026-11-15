"""Context layer: chooses factors and peers from the fund profile, then calls the
FROZEN engine. The engine is never modified; the score is whatever the engine
returns for the inputs chosen here, and the context notes are shown NEXT TO it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from features import compute_features
from diagnostics import INDEX_R2, INDEX_TE, benchmark_fit
from fundtruth import CLOSET_R2, MIN_REF
from fund_profile import (ENGINE_CATEGORIES, UNKNOWN, FundProfile, factor_set_for,
                          select_peers)
from fundtruth import FundTruth


@dataclass
class ContextResult:
    profile: FundProfile
    factor_set: object
    peer_selection: object
    engine: FundTruth | None
    score: dict | None
    notes: list = field(default_factory=list)       # warnings to DISPLAY
    relevance: dict = field(default_factory=dict)   # test -> relevance label + reason
    closet_reason: str = ""
    index_state: tuple = ("cannot_establish", "none")
    bench_fit: dict | None = None
    reference_size: int = 0
    reference_adequate: bool = False


def relevance_for(profile):
    """How relevant each test is for this kind of fund, with the reason (text shown
    to the user). Does not change any score."""
    pr, st = profile.pricing, profile.style
    rel = {}
    if pr == "appraised_nav":
        rel["autocorrelation"] = ("primary", "Appraised NAVs can be smoothed or stale; "
                                  "serial correlation is the main signal.")
        rel["near_zero"] = ("relevant", "Reported valuations can drift above zero; the "
                            "near-zero test applies.")
    elif pr in ("exchange_traded", "nav_listed_holdings"):
        basis = ("Exchange prices are set by the market" if pr == "exchange_traded" else
                 "A NAV computed from listed holdings follows market prices")
        rel["autocorrelation"] = ("informational", basis + ", so there is little room for "
                                  "smoothing; read autocorrelation as information.")
        rel["near_zero"] = ("not primary: informational", basis + ", not set by the manager, "
                            "so shifting small losses above zero is not a plausible mechanism; "
                            "a flag here is more likely chance or a mis-specified null.")
    else:
        rel["autocorrelation"] = ("generic", "Pricing basis unknown: smoothing relevance "
                                  "cannot be judged.")
        rel["near_zero"] = ("generic", "Pricing basis unknown: relevance cannot be judged.")
    if st == "passive":
        rel["factor_regression"] = ("tracking quality", "For a declared index fund a very "
                                    "high R2 is expected; the closet-indexing check is "
                                    "replaced by a tracking-quality reading.")
    else:
        rel["factor_regression"] = ("style and alpha", "Closet indexing and implausible "
                                    "alpha are the signals.")
    if profile.asset_class == "bond":
        rel["factor_regression"] = (rel["factor_regression"][0] + " (weak factors)",
                                    "Equity factors explain little of bond returns.")
    return rel


SECTOR_WORDS = ("sector", "thematic", "technology", "health", "energy", "financial",
                "real estate", "utilities", "gold", "biotech", "semiconductor", "clean",
                "infrastructure", "commodity")


def index_likeness(r2_factor, asset_class="equity", category_label=None, bench_fit=None):
    """Is the fund index-like? Returns (state, reason, basis).

    1. With a declared benchmark and its series (>= 24 overlapping months): R2 AND tracking
       error against THAT benchmark (diagnostics.INDEX_R2 / INDEX_TE, provisional).
    2. Otherwise, for BROAD EQUITY funds only: R2 against the regional factors (>= CLOSET_R2).
    3. Otherwise (bond, sector, thematic, multi-asset, or no benchmark): it cannot be
       established, because factor R2 does not say whether a fund is index-like.
    state: 'index_like' | 'not_index_like' | 'cannot_establish'."""
    if bench_fit is not None and bench_fit.get("n", 0) >= 24 and bench_fit["r2"] == bench_fit["r2"]:
        ok = bench_fit["r2"] >= INDEX_R2 and bench_fit["te"] <= INDEX_TE
        return ("index_like" if ok else "not_index_like",
                f"Against the declared benchmark: R2 {bench_fit['r2']:.2f}, tracking error "
                f"{bench_fit['te']:.1%} a year ({bench_fit['n']} months).", "benchmark")
    label = str(category_label or "").lower()
    sector = any(w in label for w in SECTOR_WORDS)
    if asset_class == "equity" and not sector:
        if not (r2_factor == r2_factor):
            return "cannot_establish", "No factor regression available.", "none"
        note = "" if label else " (no category label: assumed a broad equity fund)"
        return (("index_like" if r2_factor >= CLOSET_R2 else "not_index_like"),
                f"No benchmark series: R2 against the regional factors is {r2_factor:.2f}{note}.",
                "factors")
    why = ("bond" if asset_class == "bond" else "sector or thematic" if sector else asset_class)
    return ("cannot_establish",
            f"Index-likeness cannot be established for a {why} fund without its declared "
            "benchmark: factor R2 does not show whether such a fund is index-like.", "none")


def derive_closet_gate(r2, expense_ratio, peer_expense_median, bench_fit=None,
                       asset_class="equity", category_label=None):
    """Closet-indexing rule. The user does not declare a style. The closet flag is allowed
    only if the fund is index-like (see index_likeness) AND its expense ratio is above the
    peer median; otherwise the check is informational. Returns (engine_style, reason):
    engine style 'active' lets the engine's check run, 'passive' skips it."""
    state, why, _ = index_likeness(r2, asset_class, category_label, bench_fit)
    if state == "not_index_like":
        return "passive", why + " Not index-like: no closet signal possible."
    if state == "cannot_establish":
        return "passive", why + " The closet check is informational only."
    if expense_ratio is None or peer_expense_median is None or peer_expense_median != peer_expense_median:
        return "passive", why + (" Index-like, but the expense ratio or the peer median is not "
                                 "available: without it closet indexing is NOT detected.")
    if expense_ratio > peer_expense_median:
        return "active", why + (" Index-like and expense ratio %.2f%% above the peer median %.2f%%: "
                                "closet-indexing flag allowed." % (100 * expense_ratio, 100 * peer_expense_median))
    return "passive", why + (" Index-like with expense ratio %.2f%% at or below the peer median "
                             "%.2f%%: consistent with an index fund, informational only."
                             % (100 * expense_ratio, 100 * peer_expense_median))


def analyse(returns, profile, factors_by_region=None, reference=None, benchmark=None,
            category_label=None):
    """Run the frozen engine in the context of `profile`.

    factors_by_region : {'US': df, 'Europe': df, ...}, decimal, month-end.
    reference         : frozen reference table (columns of the engine + optional 'region').
    """
    factors_by_region = factors_by_region or {}
    notes = list(profile.data_warnings)
    fs = factor_set_for(profile)
    notes += fs.warnings
    factors = None
    if fs.usable:
        factors = factors_by_region.get(fs.key)
        if factors is None:
            notes.append(f"Factor set '{fs.label}' not loaded: factor regression skipped.")
    ps = select_peers(reference, profile, exclude=profile.fund_id) if reference is not None else None
    if ps is not None:
        notes.append(ps.note)
    # style is DERIVED, never chosen: R2 and expense ratio (see derive_closet_gate)
    r2 = compute_features(returns, factors)["r2"] if factors is not None else float("nan")
    peer_exp = None
    if ps is not None and "expense_ratio" in ps.peers.columns:
        peer_exp = float(ps.peers["expense_ratio"].median())
    bfit = benchmark_fit(returns, benchmark) if benchmark is not None else None
    style, closet_reason = derive_closet_gate(r2, profile.expense_ratio, peer_exp, bfit,
                                              profile.asset_class, category_label)
    state, _, basis = index_likeness(r2, profile.asset_class, category_label, bfit)
    index_state = (state, basis)
    res = ContextResult(profile, fs, ps, None, None, notes, relevance_for(profile))
    res.closet_reason = closet_reason
    res.index_state, res.bench_fit = index_state, bfit
    if not profile.scoring_supported:
        notes.append(f"No integrity score: asset class '{profile.asset_class}' is outside "
                     f"the engine scope {ENGINE_CATEGORIES}; diagnostics only.")
        return res
    if ps is not None and ps.engine_pooled:
        peers = reference.drop(index=profile.fund_id, errors="ignore")   # engine pools, as before
    else:
        peers = None if ps is None else ps.peers
    ecat = profile.engine_category
    if profile.asset_class != ecat:
        notes.append(f"Asset class '{profile.asset_class}' is scored with the engine category "
                     f"'{ecat}' as the closest available; peers are the {profile.asset_class} funds "
                     "only. Read the peer-based readings with care.")
        if peers is not None:
            peers = peers.copy()
            peers["category"] = ecat      # the engine sees only the selected peers
    eng = FundTruth(returns.rename(profile.fund_id), factors, peers=peers,
                    category=ecat, style=style)
    res.engine, res.score = eng, eng.score()
    res.reference_size = len(eng._ref)
    res.reference_adequate = res.reference_size >= MIN_REF
    if not res.reference_adequate:
        notes.append(f"Reference has only {res.reference_size} funds (< {MIN_REF}): peer-based "
                     "p-values are not meaningful, so a high score here is NOT reassurance.")
    unknown = [f for f in ("region", "currency", "pricing") if getattr(profile, f) == UNKNOWN]
    if unknown:
        notes.append(f"Reliability note: {', '.join(unknown)} unknown, so the context "
                     "readings are generic.")
    return res


# ------------------------------------------------------------------ references
def build_reference(returns_by_fund, profiles, factors_by_region, min_obs=36):
    """Feature table (one row per eligible fund) for a frozen reference.
    Columns as reference_real_v1 plus 'region'. Funds with too few months or an
    asset class outside the engine scope are left out."""
    rows = []
    for fid, s in returns_by_fund.items():
        p = profiles[fid]
        if len(s) < min_obs or not p.scoring_supported:
            continue
        fs = factor_set_for(p)
        fac = factors_by_region.get(fs.key) if fs.usable else None
        if fac is not None:
            fac = fac.loc[fac.index.intersection(s.index)]
        f = compute_features(s, fac)
        # style of a reference row is derived (index-like -> 'passive'), as in analyse()
        f.update(category=p.asset_class,
                 style="passive" if f["r2"] == f["r2"] and f["r2"] >= CLOSET_R2 else "active",
                 region=p.region)
        if p.expense_ratio is not None:
            f["expense_ratio"] = p.expense_ratio
        rows.append(pd.Series(f, name=fid))
    return pd.DataFrame(rows)


def write_reference(df, version, out_dir="data"):
    """Write reference_real_<version>.csv/.json. Refuses to overwrite an existing
    version: a changed reference needs a new version number."""
    path = os.path.join(out_dir, f"reference_real_{version}.csv")
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: references are frozen; use a new version.")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index_label="fund")
    with open(os.path.join(out_dir, f"reference_real_{version}.json"), "w") as fh:
        json.dump({"version": f"real-{version}-{date.today().isoformat()}",
                   "n_funds": len(df)}, fh)
    return path
