"""Universe layer: profiles, factors, groups, peer statistics, reference, scoring of every fund.

Sits ABOVE the frozen engine (it calls context.analyse / panel.analyse_panel and never changes
a score rule). Works for a Bloomberg-style universe (ingest.py) and for the Yahoo universe in
data/ (degraded mode).
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import data as D
import diagnostics as dg
import panel as pn
from context import analyse, build_reference
from fund_profile import UNKNOWN, build_profile, factor_set_for, profile_from_legacy_meta, propose_pricing
from fundtruth import MIN_REF

TESTS = ("autocorrelation", "factor_regression", "near_zero", "peer_anomaly")   # Benford: weight 0
READING_KEYS = ("price_regularity", "style", "alpha", "risk_shape", "distribution", "peers")
FACTOR_ZIPS = {"US": "F-F_Research_Data_Factors_CSV.zip", "Europe": "Europe_3_Factors_CSV.zip",
               "Global": "Developed_3_Factors_CSV.zip"}


@dataclass
class Universe:
    source: str                                   # 'bloomberg' | 'yahoo' | 'sample'
    returns: dict
    profiles: dict
    meta: pd.DataFrame
    benchmarks: dict
    factors: dict
    reference: pd.DataFrame
    reference_label: str
    peer_stats: pd.DataFrame
    received: str = ""
    excluded: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["fund_id", "reason"]))
    notes: list = field(default_factory=list)
    checksums: dict = field(default_factory=dict)


# ------------------------------------------------------------------ factors
def read_factor_csv(path):
    f = pd.read_csv(path, index_col=0, parse_dates=True)
    f.index = f.index.to_period("M").to_timestamp("M")
    need = {"Mkt-RF", "SMB", "HML", "RF"}
    if not need <= set(f.columns):
        raise ValueError(f"{path}: factor columns missing: {sorted(need - set(f.columns))}. "
                         "Factors must be decimals (0.012 = 1.2%).")
    if f[list(need)].abs().max().max() > 1.0:
        raise ValueError(f"{path}: factor values look like percent; use decimals.")
    return f


def load_factors(regions, factors_dir=None, zips=None):
    """{'US': df, ...}. Sources, in order: --factors-dir/factors_<region>.csv, or French library
    zips given per region. Missing regions are returned in `missing` (factor test skipped)."""
    out, missing = {}, []
    for r in regions:
        p = os.path.join(factors_dir, f"factors_{r}.csv") if factors_dir else None
        if p and os.path.exists(p):
            out[r] = read_factor_csv(p)
        elif zips and zips.get(r):
            out[r] = D.load_ff(zips[r], r)
        else:
            missing.append(r)
    return out, missing


# ------------------------------------------------------------------ build
def make_profiles(meta, returns):
    profs = {}
    for fid in returns:
        row = meta.loc[fid].to_dict()
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        p = build_profile(fid, row)
        if p.pricing == UNKNOWN:
            prop = propose_pricing(None, row.get("category_label"))
            if prop:
                p.pricing, p.provenance["pricing"] = prop, "proposed"
                p.data_warnings.append("Pricing type proposed from the category label; confirm.")
        if row.get("is_dead"):
            p.data_warnings.append("Dead/merged fund (kept in the universe to reduce survivorship bias).")
        p.data_warnings = [w for w in p.data_warnings if not w.startswith("style unknown")]   # style is derived, not supplied
        profs[fid] = p
    return profs


def peer_stats_table(returns, profiles, factors, reference):
    ps = dg.build_peer_stats(returns, profiles, factors)
    if len(ps) and "skew" in reference:
        ps["skew"] = reference["skew"].reindex(ps.index)
    return ps


def build_bloomberg_universe(ing, factors, version, snap_dir, source="bloomberg"):
    profs = make_profiles(ing.meta, ing.returns)
    ref = build_reference(ing.returns, profs, factors)
    path = os.path.join(snap_dir, f"reference_real_{version}.csv")
    os.makedirs(snap_dir, exist_ok=True)
    if os.path.exists(path):                       # frozen: identical content is reused, else refuse
        old = pd.read_csv(path, index_col="fund")
        if not (old.shape == ref.shape and np.allclose(old.select_dtypes("number").to_numpy(),
                                                       ref.select_dtypes("number").to_numpy(), equal_nan=True, atol=1e-9)):
            raise FileExistsError(f"{path} exists with different content: references are frozen; "
                                  "use --version with a new name.")
    else:
        ref.to_csv(path, index_label="fund")
    ref = pd.read_csv(path, index_col="fund")
    ps = peer_stats_table(ing.returns, profs, factors, ref)
    return Universe(source, ing.returns, profs, ing.meta, ing.benchmarks, factors, ref,
                    f"real funds, version {version} (n={len(ref)}, leave-one-out)", ps, ing.received,
                    ing.excluded, list(ing.notes), dict(ing.checksums))


def build_yahoo_universe(data_dir="data"):
    """Degraded mode: the 40 US-listed Yahoo funds with the frozen v1 reference."""
    W = pd.read_csv(os.path.join(data_dir, "fund_returns.csv"), index_col="date", parse_dates=True)
    F = pd.read_csv(os.path.join(data_dir, "factors.csv"), index_col="date", parse_dates=True)
    M = pd.read_csv(os.path.join(data_dir, "funds_meta.csv")).set_index("ticker")
    rets = {t: W[t].dropna().rename(t) for t in W.columns}
    profs = {t: profile_from_legacy_meta(M.loc[t].to_dict(), t) for t in rets}
    ref = pd.read_csv(os.path.join(data_dir, "reference_real_v1.csv"), index_col="fund")
    meta = pd.DataFrame({t: dict(region="US", currency="USD", asset_class=M.loc[t, "category"],
                                 pricing=profs[t].pricing, benchmark_id="", expense_ratio=np.nan,
                                 is_dead=False, category_label="") for t in rets}).T
    ps = peer_stats_table(rets, profs, {"US": F}, ref)
    note = open(os.path.join(data_dir, "data_note.txt")).read().splitlines()[:1] if os.path.exists(
        os.path.join(data_dir, "data_note.txt")) else []
    return Universe("yahoo", rets, profs, meta, {}, {"US": F}, ref, "real funds v1 (Yahoo, n=%d)" % len(ref), ps,
                    "", pd.DataFrame(columns=["fund_id", "reason"]),
                    ["Degraded mode: public Yahoo Finance data, 40 US-listed funds, US factors only, no expense ratios, "
                     "no dead funds (survivorship bias), pricing types proposed from fund type."] + note)


# ------------------------------------------------------------------ groups
def groups_table(U):
    """asset class x region: size, peer rule actually applied, medians of the peer statistics."""
    rows = []
    ps = U.peer_stats
    for (ac, rg), ids in pd.DataFrame({"ac": [p.asset_class for p in U.profiles.values()],
                                       "rg": [p.region for p in U.profiles.values()]},
                                      index=list(U.profiles)).groupby(["ac", "rg"]).groups.items():
        sub = ps.loc[ps.index.intersection(list(ids))] if len(ps) else ps
        n = len(ids)
        ok = n >= MIN_REF
        n_dead = int(sum(bool(U.meta.loc[i].get("is_dead", False)) for i in ids))
        n_ac = sum(1 for p in U.profiles.values() if p.asset_class == ac)
        rule = ("group used as peer set" if ok else
                f"FALLBACK: only {n} funds (< {MIN_REF}); peers are all {ac} funds regardless of region (n={n_ac}); "
                + ("still < %d: the engine pools across asset classes" % MIN_REF if n_ac < MIN_REF else "regions are mixed"))
        med = lambda c: float(sub[c].median()) if c in sub and sub[c].notna().any() else np.nan
        rows.append(dict(asset_class=ac, region=rg, n_funds=n, n_dead=n_dead, enough_peers=ok, peer_rule=rule,
                         n_with_expense=int(sub["expense_ratio"].notna().sum()) if "expense_ratio" in sub else 0,
                         median_expense=med("expense_ratio"), median_vol=med("vol"), median_ac1=med("ac1"),
                         median_r2=med("r2"), median_skew=med("skew"), median_beta_mkt=med("beta_mkt"),
                         median_beta_smb=med("beta_smb"), median_beta_hml=med("beta_hml")))
    return pd.DataFrame(rows).sort_values(["asset_class", "region"]).reset_index(drop=True)


def peer_distributions(U):
    """Per group: n and 5/25/50/75/95th percentiles of each peer statistic (no fund-level values)."""
    ps = U.peer_stats.copy()
    ps["asset_class"] = [U.profiles[i].asset_class for i in ps.index]
    ps["region"] = [U.profiles[i].region for i in ps.index]
    rows = []
    for (ac, rg), d in ps.groupby(["asset_class", "region"]):
        for c in ("vol", "sharpe", "ac1", "r2", "skew", "beta_mkt", "beta_smb", "beta_hml", "max_drawdown", "expense_ratio"):
            if c in d and d[c].notna().sum():
                v = d[c].dropna()
                rows.append(dict(asset_class=ac, region=rg, statistic=c, n=len(v), p05=v.quantile(.05), p25=v.quantile(.25),
                                 p50=v.median(), p75=v.quantile(.75), p95=v.quantile(.95)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scoring
def _bench(U, fid):
    b = U.meta.loc[fid].get("benchmark_id") if "benchmark_id" in U.meta else None
    return (U.benchmarks.get(b), b) if isinstance(b, str) and b in U.benchmarks else (None, "")


def analyse_fund(U, fid, returns=None):
    """Panel for one fund of the universe (leave-one-out inside the context layer)."""
    s = U.returns[fid] if returns is None else returns
    b, bl = _bench(U, fid)
    label = U.meta.loc[fid].get("category_label") if "category_label" in U.meta else None
    return pn.analyse_panel(s, U.profiles[fid], U.factors, U.reference, U.reference_label, b, bl,
                            U.peer_stats, None, category_label=label if isinstance(label, str) else None)


def findings_table(U):
    """One row per fund, aggregated findings only (no series, no dates)."""
    warnings.filterwarnings("ignore")
    rows = []
    for fid in sorted(U.returns):
        p = analyse_fund(U, fid)
        res, prof = p.res, U.profiles[fid]
        row = dict(fund_id=fid, asset_class=prof.asset_class, region=prof.region, currency=prof.currency,
                   pricing=prof.pricing, is_dead=bool(U.meta.loc[fid].get("is_dead", False)),
                   n_obs=len(p.returns), peer_level=res.peer_selection.level if res.peer_selection else "",
                   peer_n=res.peer_selection.n if res.peer_selection else 0, reference_size=res.reference_size,
                   reference_adequate=res.reference_adequate, expense_ratio=prof.expense_ratio,
                   index_state=res.index_state[0], index_basis=res.index_state[1],
                   closet_reason=res.closet_reason, n_notes=len(res.notes),
                   bench_r2=(res.bench_fit or {}).get("r2", np.nan), bench_te=(res.bench_fit or {}).get("te", np.nan))
        if res.score is None:
            row.update(score=np.nan, reliability="none", cap_applied=False)
        else:
            tests = res.engine.all_tests()
            row.update(score=res.score["score"], reliability=res.score["reliability"],
                       cap_applied=bool(res.score["cap_applied"]),
                       **{f"sub_{k}": v for k, v in res.score["subscores"].items()},
                       **{f"p_{k}": tests[k]["p_value"] for k in TESTS},
                       **{f"flag_{k}": bool(tests[k]["flag"] and tests[k]["status"] == "ok") for k in TESTS},
                       **{f"status_{k}": tests[k]["status"] for k in TESTS})
        lv = {x.key: x.level for x in p.readings}
        tg = {x.key: x.tag for x in p.readings}
        for k in READING_KEYS:
            row[f"reading_{k}"] = lv.get(k, "not scored")
            row[f"role_{k}"] = tg.get(k, "")
        row["headline"] = p.headline
        rows.append(row)
    return pd.DataFrame(rows)


def universe_summary(F):
    g = F.groupby(["asset_class", "region", "pricing"], dropna=False)
    return g.agg(n_funds=("fund_id", "size"), n_scored=("score", "count"), n_dead=("is_dead", "sum"),
                 median_months=("n_obs", "median"), median_score=("score", "median"),
                 n_unusual_any=("headline", lambda x: int(x.str.contains("unusual", case=False).sum()))).reset_index()


# ------------------------------------------------------------------ snapshot (local, processed)
def write_snapshot(U, snap_dir, version, meta_extra=None):
    """Processed snapshot for the app: cleaned monthly returns (wide), meta, benchmarks, factors,
    reference and peer statistics. LOCAL ONLY (folder is git-ignored); never part of shared results."""
    os.makedirs(snap_dir, exist_ok=True)
    pd.DataFrame(U.returns).to_csv(os.path.join(snap_dir, "returns_wide.csv"), index_label="date")
    U.meta.to_csv(os.path.join(snap_dir, "meta.csv"), index_label="fund_id")
    if U.benchmarks:
        pd.DataFrame(U.benchmarks).to_csv(os.path.join(snap_dir, "benchmarks_wide.csv"), index_label="date")
    for r, f in U.factors.items():
        f.to_csv(os.path.join(snap_dir, f"factors_{r}.csv"), index_label="date")
    U.peer_stats.to_csv(os.path.join(snap_dir, "peer_stats.csv"), index_label="fund")
    ref_name = f"reference_real_{version}.csv"
    if not os.path.exists(os.path.join(snap_dir, ref_name)):
        U.reference.to_csv(os.path.join(snap_dir, ref_name), index_label="fund")
    U.excluded.to_csv(os.path.join(snap_dir, "excluded.csv"), index=False)
    json.dump(dict(source=U.source, received=U.received, version=version, reference_file=ref_name,
                   reference_label=U.reference_label, notes=U.notes, checksums=U.checksums,
                   **(meta_extra or {})), open(os.path.join(snap_dir, "snapshot.json"), "w"), indent=1)


def load_snapshot(snap_dir):
    j = json.load(open(os.path.join(snap_dir, "snapshot.json")))
    W = pd.read_csv(os.path.join(snap_dir, "returns_wide.csv"), index_col="date", parse_dates=True)
    meta = pd.read_csv(os.path.join(snap_dir, "meta.csv"), index_col="fund_id")
    rets = {c: W[c].dropna().rename(c) for c in W.columns}
    bp = os.path.join(snap_dir, "benchmarks_wide.csv")
    bm = {}
    if os.path.exists(bp):
        B = pd.read_csv(bp, index_col="date", parse_dates=True)
        bm = {c: B[c].dropna() for c in B.columns}
    facs = {}
    for r in ("US", "Europe", "Global"):
        p = os.path.join(snap_dir, f"factors_{r}.csv")
        if os.path.exists(p):
            facs[r] = pd.read_csv(p, index_col="date", parse_dates=True)
    ref = pd.read_csv(os.path.join(snap_dir, j["reference_file"]), index_col="fund")
    ps = pd.read_csv(os.path.join(snap_dir, "peer_stats.csv"), index_col="fund")
    meta["is_dead"] = meta["is_dead"].astype(str).str.lower().isin(["true", "1"])
    profs = make_profiles(meta.where(meta.notna(), None), rets)
    exc = pd.read_csv(os.path.join(snap_dir, "excluded.csv"))
    return Universe(j["source"], rets, profs, meta, bm, facs, ref, j["reference_label"], ps, j["received"], exc,
                    j.get("notes", []), j.get("checksums", {}))
