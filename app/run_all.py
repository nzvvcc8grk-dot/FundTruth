"""One command: ingestion -> universe -> scoring -> real validation -> tables and charts -> audit.

Everything runs LOCALLY. The results folder holds only aggregated results (no raw series, anonymised
IDs, no name mapping); the processed snapshot (cleaned series, used by the app) and the id mapping
stay in snapshot/ and private/ (git-ignored) and are never part of the results.

  Bloomberg extract (bbg/ folder, schema in README):
    python run_all.py --bbg bbg --ff-us F-F_Research_Data_Factors_CSV.zip \
        --ff-europe Europe_3_Factors_CSV.zip --ff-global Developed_3_Factors_CSV.zip
  Degraded mode (public Yahoo data in data/, 40 US funds):
    python run_all.py --yahoo
  Rehearsal on the synthetic sample (factors synthetic: regressions are meaningless):
    python run_all.py --dry-run
Exit codes: 0 ok | 2 ingestion stopped (message says why) | 3 engine changed | 4 audit found issues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ingest as IG
import output_audit as OA
import real_validation as RV
import three_funds as TF
import universe as UN
from fundtruth import FLAG_P, MIN_REF, P_SCALE

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
CITE = "Bloomberg Terminal, retrieved {date}"


def engine_ok():
    bad = []
    for line in open(os.path.join(HERE, "tests", "engine_hashes.txt")):
        h, n = line.split()
        if n in ("fundtruth.py", "features.py"):
            if hashlib.sha256(open(os.path.join(HERE, n), "rb").read()).hexdigest() != h:
                bad.append(n)
    return bad


def versions():
    import matplotlib as m, scipy, sklearn, statsmodels
    return dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__,
                statsmodels=statsmodels.__version__, scikit_learn=sklearn.__version__, matplotlib=m.__version__)


class Numbers:
    """Every figure of the report goes through here: numeri_report.csv/json."""
    def __init__(self):
        self.rows = []

    def add(self, key, value, unit="", source="", note=""):
        if isinstance(value, (np.floating, float)):
            value = None if value != value else round(float(value), 4)
        elif isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.bool_,)):
            value = bool(value)
        self.rows.append(dict(key=key, figure=value, unit=unit, source_file=source, note=note))

    def frame(self):
        return pd.DataFrame(self.rows)


def synthetic_numbers(N):
    """Figures from the earlier synthetic validation files (not recomputed here)."""
    r = os.path.join(HERE, "results")
    p = os.path.join(r, "validation_summary.csv")
    if os.path.exists(p):
        for x in pd.read_csv(p).itertuples():
            N.add(f"synthetic.{x.label}.pct_high_concern", x.pct_high_concern, "%", "validation_summary.csv",
                  f"n={x.n} simulated series, score < 50")
            N.add(f"synthetic.{x.label}.mean_score", x.mean_score, "0-100", "validation_summary.csv")
    p = os.path.join(r, "flag_rates_per_test.csv")
    if os.path.exists(p):
        d = pd.read_csv(p, index_col=0)
        hon = [i for i in d.index if i.startswith(("d_", "honest"))]
        for c in d.columns:
            N.add(f"synthetic.false_positive_pct.{c.replace('flag_', '')}", d.loc[hon, c].mean(), "%",
                  "flag_rates_per_test.csv", "mean over simulated honest series")
    p = os.path.join(r, "power_curve.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        c = c[c.reference.str.startswith("synthetic")]
        for x in c.itertuples():
            N.add(f"synthetic.power.smoothing_{x.strength}.flag_pct", x.flag_gated, "%", "power_curve.csv")
    p = os.path.join(r, "closet_rule_validation.csv")
    if os.path.exists(p):
        for i, x in enumerate(pd.read_csv(p).itertuples()):
            N.add(f"synthetic.closet.{i}.factor_flag_pct", x.factor_test_flag_pct, "%", "closet_rule_validation.csv", x.scenario)


def plot_groups(G, path):
    g = G.copy(); g["lab"] = g.asset_class + " x " + g.region
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(g) + 1.3))
    ax.barh(g.lab, g.n_funds, color=np.where(g.enough_peers, "#2e7d5b", "#c58a00"))
    ax.axvline(MIN_REF, color="#333", ls="--", lw=1)
    ax.text(MIN_REF, -0.9, f" {MIN_REF}-fund minimum", fontsize=8)
    for i, v in enumerate(g.n_funds):
        ax.text(v + .3, i, str(v), va="center", fontsize=8)
    ax.set_xlabel("Funds in group (amber: below minimum, peer fallback applies)"); ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_rates(tables, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    tests = list(UN.TESTS) + ["any of the four"]
    w = 0.8 / max(1, len(tables))
    cols = ["#1f4e79", "#c58a00", "#2e7d5b"]
    for j, (lab, fr) in enumerate(tables.items()):
        a = fr[(fr.level == "all funds")].set_index("test").reindex(tests)
        y = a.flag_pct.to_numpy()
        err = np.clip(np.vstack([y - a.ci95_low.to_numpy(), a.ci95_high.to_numpy() - y]), 0, None)
        ax.bar(np.arange(len(tests)) + j * w, y, w, yerr=err, color=cols[j % 3], label=lab, capsize=3)
    ax.set_xticks(np.arange(len(tests)) + w * (len(tables) - 1) / 2); ax.set_xticklabels([t.replace("_", " ") for t in tests], fontsize=8)
    ax.set_ylabel("% of funds flagged (95% Wilson interval)")
    ax.axhline(5, color="#999", ls=":", lw=1); ax.text(len(tests) - .5, 5.3, "5%", fontsize=7, color="#666", ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Flag rate per check on real funds (upper bound on false positives: no labelled cases)", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_injection(inj, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = inj.detection_pct.to_numpy()
    ax.barh(inj.scenario, y, color=["#9aa5b1"] + ["#1f4e79"] * (len(inj) - 1),
            xerr=np.clip(np.vstack([y - inj.ci95_low, inj.ci95_high - y]), 0, None), capsize=3)
    for i, v in enumerate(y):
        ax.text(min(v + 2, 98), i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, 105); ax.invert_yaxis(); ax.set_xlabel("% of funds where the relevant check flags (baseline: any of four)")
    ax.set_title("Anomalies injected into real series (simulated on real funds, not real cases)", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_scores(F, path):
    F = F.dropna(subset=["score"]).copy(); F["lab"] = F.asset_class + "\n" + F.region
    labs = sorted(F.lab.unique())
    fig, ax = plt.subplots(figsize=(max(6, .9 * len(labs)), 3.6))
    ax.boxplot([F[F.lab == l].score for l in labs], tick_labels=labs, showfliers=True)
    ax.set_ylabel("Priority index (technical, not calibrated)"); ax.tick_params(axis="x", labelsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def process(U, out, N, prefix, with_injection=True):
    """Score a universe and write its aggregated tables. Returns (F, G, validation dict)."""
    os.makedirs(out, exist_ok=True)
    F = UN.findings_table(U)
    G = UN.groups_table(U)
    F.to_csv(os.path.join(out, "findings_by_fund.csv"), index=False)
    G.to_csv(os.path.join(out, "groups.csv"), index=False)
    UN.universe_summary(F).to_csv(os.path.join(out, "universe_summary.csv"), index=False)
    UN.peer_distributions(U).to_csv(os.path.join(out, "peer_distributions.csv"), index=False)
    V = RV.run(U, F, U.source, with_injection)
    V["flag_rates"].to_csv(os.path.join(out, "flag_rates_real.csv"), index=False)
    V["reading_counts"].to_csv(os.path.join(out, "reading_status_real.csv"), index=False)
    if with_injection:
        V["injection"].to_csv(os.path.join(out, "injection_real.csv"), index=False)
    N.add(f"{prefix}.n_funds", len(F), "funds", "universe_summary.csv")
    N.add(f"{prefix}.n_scored", int(F.score.count()), "funds", "findings_by_fund.csv")
    N.add(f"{prefix}.n_dead", int(F.is_dead.sum()), "funds", "findings_by_fund.csv", V["survivorship"])
    N.add(f"{prefix}.n_groups", len(G), "groups", "groups.csv")
    N.add(f"{prefix}.n_groups_below_minimum", int((~G.enough_peers).sum()), "groups", "groups.csv", f"minimum {MIN_REF} funds")
    for g in G.itertuples():
        N.add(f"{prefix}.group.{g.asset_class}.{g.region}.n_funds", g.n_funds, "funds", "groups.csv", g.peer_rule)
    fr = V["flag_rates"]
    for r in fr[(fr.level == "all funds")].itertuples():
        t = r.test.replace(" ", "_")
        N.add(f"{prefix}.flag_pct.{t}", r.flag_pct, "%", "flag_rates_real.csv", f"{r.n_flagged}/{r.n_tested}; 95% CI {r.ci95_low:.1f}-{r.ci95_high:.1f}")
    for r in fr[(fr.level == "dead vs live")].itertuples():
        if r.test == "any of the four":
            N.add(f"{prefix}.flag_pct.any.{r.key.replace('/', '_').replace(' ', '_')}", r.flag_pct, "%", "flag_rates_real.csv", f"n={r.n_funds}")
    if with_injection:
        for r in V["injection"].itertuples():
            k = r.scenario.split(" (")[0].replace(" ", "_").replace("=", "")
            N.add(f"{prefix}.injection.{k}.detection_pct", r.detection_pct, "%", "injection_real.csv",
                  f"{r.n_flagged}/{r.n_funds}; target {r.target_test}")
    N.add(f"{prefix}.pct_score_below_50", 100 * float((F.score < 50).sum()) / max(1, F.score.count()), "%", "findings_by_fund.csv")
    N.add(f"{prefix}.pct_reference_inadequate", 100 * float((~F.reference_adequate).mean()), "%", "findings_by_fund.csv", "funds scored with a reference < 20")
    return F, G, V


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbg", default="bbg")
    ap.add_argument("--out", default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--private-dir", default="private")
    ap.add_argument("--factors-dir"); ap.add_argument("--ff-us"); ap.add_argument("--ff-europe"); ap.add_argument("--ff-global")
    ap.add_argument("--yahoo", action="store_true", help="degraded mode: public Yahoo universe in data/")
    ap.add_argument("--dry-run", action="store_true", help="rehearsal on sample_data/bbg with synthetic factors")
    ap.add_argument("--no-yahoo-compare", action="store_true")
    ap.add_argument("--anonymise", action="store_true"); ap.add_argument("--max-gaps", type=int, default=0)
    ap.add_argument("--version", default=None); ap.add_argument("--three", default=None, help="three fund ids, comma separated")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    a = ap.parse_args(argv)
    np.random.seed(SEED)

    bad = engine_ok()
    if bad:
        print(f"ENGINE CHANGED: {bad} differ from tests/engine_hashes.txt. Stopping.", file=sys.stderr)
        return 3
    mode = "yahoo" if a.yahoo else "dry-run" if a.dry_run else "bloomberg"
    if a.dry_run:
        a.bbg, a.factors_dir = os.path.join(HERE, "sample_data", "bbg"), os.path.join(HERE, "sample_data", "bbg")
    out = a.out or {"yahoo": "results/yahoo_run", "dry-run": "results/dryrun", "bloomberg": "results/bbg_run"}[mode]
    snap = a.snapshot or {"yahoo": "snapshot_yahoo", "dry-run": "snapshot_dryrun", "bloomberg": "snapshot"}[mode]
    if os.path.exists(out):
        shutil.rmtree(out)                             # outputs are regenerated, never mixed with an older run
    os.makedirs(out)
    N = Numbers()
    log = dict(mode=mode, seed=SEED, versions=versions(), engine_hashes="match tests/engine_hashes.txt",
               parameters=dict(P_SCALE=P_SCALE, FLAG_P=FLAG_P, MIN_REF=MIN_REF, retuned=RV.RETUNED))

    if mode == "yahoo":
        U = UN.build_yahoo_universe(a.data_dir)
        dq = ["FundTruth data-quality report (DEGRADED MODE: public Yahoo Finance data)"] + U.notes + [
            f"Funds: {len(U.returns)}; factors: US only (Kenneth French library); IDs are public tickers (not Bloomberg).",
            "A flag is a signal to investigate, never proof."]
        open(os.path.join(out, "data_quality_report.txt"), "w").write("\n".join(dq) + "\n")
        log["data_received"] = "Yahoo Finance (see data/data_note.txt)"
    else:
        try:
            ing = IG.ingest(a.bbg, a.anonymise, a.max_gaps)
        except IG.IngestError as e:
            print(f"INGEST STOPPED: {e}", file=sys.stderr)
            return 2
        if ing.id_map is not None:
            os.makedirs(a.private_dir, exist_ok=True)
            ing.id_map.to_csv(os.path.join(a.private_dir, "id_map.csv"), index=False)
        regions = sorted({p.region for p in UN.make_profiles(ing.meta, ing.returns).values()} & {"US", "Europe", "Global"})
        zips = dict(US=a.ff_us, Europe=a.ff_europe, Global=a.ff_global)
        fac, missing = UN.load_factors(regions, a.factors_dir, zips)
        extra = [f"Factor file not provided for region {r}: factor test skipped for those funds." for r in missing]
        extra += [f"Factors loaded: {', '.join(sorted(fac)) or 'none'}."]
        if mode == "dry-run":
            extra.append("DRY RUN: factors and data are synthetic; regressions and all results are meaningless.")
        ver = a.version or ("dryrun-" if mode == "dry-run" else "bbg-") + ing.checksums["returns.csv"][:8]
        U = UN.build_bloomberg_universe(ing, fac, ver, snap, source="sample" if mode == "dry-run" else "bloomberg")
        log.update(data_received=ing.received, input_checksums=ing.checksums, reference_version=ver)
        N.add("data.received", ing.received, "date", "field_log.csv", CITE.format(date=ing.received))
        N.add("data.n_excluded", len(ing.excluded), "funds", "data_quality_report.txt")
        N.add("data.n_loaded_before_exclusion", len(ing.returns) + len(ing.excluded), "funds", "data_quality_report.txt")
        open(os.path.join(out, "data_quality_report.txt"), "w").write(IG.quality_report(ing, [""] + extra))

    F, G, V = process(U, out, N, "universe")
    cmp = {"this universe": V["flag_rates"]}
    yd = a.data_dir
    if mode != "yahoo" and not a.no_yahoo_compare and os.path.exists(os.path.join(yd, "fund_returns.csv")):
        Uy = UN.build_yahoo_universe(yd)
        Fy, Gy, Vy = process(Uy, os.path.join(out, "yahoo_compare"), N, "yahoo", with_injection=False)
        cmp["Yahoo universe (40 US funds)"] = Vy["flag_rates"]
        open(os.path.join(out, "yahoo_compare", "NOTE.txt"), "w").write("Degraded-mode universe for comparison. Public tickers, not Bloomberg data.\n")
    UN.write_snapshot(U, snap, log.get("reference_version", "v1"), dict(mode=mode))

    cards, missing = TF.build(U, a.three.split(",") if a.three else None)
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    if cards:
        T = TF.table(cards)
        T.to_csv(os.path.join(out, "three_funds.csv"), index=False)
        TF.chart(cards, os.path.join(out, "charts", "three_funds.png"))
        for slot, fid, p, own in cards:
            N.add(f"three_funds.{fid}.priority_index", p.technical.get("priority_index", np.nan) if p.technical else np.nan, "0-100", "three_funds.csv", slot)
            N.add(f"three_funds.{fid}.headline", p.headline, "", "three_funds.csv", slot)
    for m in missing:
        N.add(f"three_funds.missing.{m}", "no fund matches", "", "three_funds.csv", "pick IDs with --three")
    plot_groups(G, os.path.join(out, "charts", "group_sizes.png"))
    plot_rates(cmp, os.path.join(out, "charts", "flag_rates.png"))
    if "injection" in V:
        plot_injection(V["injection"], os.path.join(out, "charts", "injection.png"))
    plot_scores(F, os.path.join(out, "charts", "priority_index_by_group.png"))
    synthetic_numbers(N)
    N.add("parameters.P_SCALE", P_SCALE, "", "fundtruth.py", "provisional")
    N.add("parameters.FLAG_P", FLAG_P, "", "fundtruth.py", "provisional")
    N.add("parameters.cap", 30, "points", "fundtruth.py", "provisional")
    N.add("parameters.retuned_on_these_data", RV.RETUNED, "", "real_validation.py", "no retuning; split not needed")
    nf = N.frame()
    nf.to_csv(os.path.join(out, "numeri_report.csv"), index=False)
    json.dump({r["key"]: dict(figure=r["figure"], unit=r["unit"], source=r["source_file"], note=r["note"]) for r in N.rows},
              open(os.path.join(out, "numeri_report.json"), "w"), indent=1)
    log["run_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["n_numbers"] = len(nf)
    open(os.path.join(out, "run_log.json"), "w").write(json.dumps(log, indent=1))
    open(os.path.join(out, "README_results.txt"), "w").write(
        "Aggregated results only (no raw series, anonymised IDs, no name mapping). Safe to share.\n"
        "universe_summary.csv, groups.csv, peer_distributions.csv, findings_by_fund.csv, flag_rates_real.csv,\n"
        "reading_status_real.csv, injection_real.csv, three_funds.csv, numeri_report.csv/json, run_log.json, charts/.\n"
        + (CITE.format(date=log.get("data_received", "")) + "\n" if mode != "yahoo" else "")
        + "A flag is a signal to investigate, never proof. Priority index: not calibrated against known cases.\n")
    ids = pd.read_csv(os.path.join(a.private_dir, "id_map.csv"))["original_id"].astype(str).tolist() \
        if os.path.exists(os.path.join(a.private_dir, "id_map.csv")) else []
    yc = os.path.join(out, "yahoo_compare")
    iss = OA.audit(out, allow_public_ids=(mode == "yahoo"), private_ids=ids)
    if os.path.isdir(yc):          # public tickers are expected there; everything else must be anonymous
        iss = [i for i in iss if not i.startswith("yahoo_compare")] + OA.audit(yc, True, ids)
    open(os.path.join(out, "audit_report.txt"), "w").write("clean\n" if not iss else "\n".join(iss) + "\n")
    print(f"run_all OK ({mode}): {len(F)} funds, {len(G)} groups, {len(nf)} numbers -> {out}/")
    if iss:
        print("AUDIT ISSUES:\n  " + "\n  ".join(iss), file=sys.stderr)
        return 4
    print("audit: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
