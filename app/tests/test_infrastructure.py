"""Ingestion hard stops, universe/groups, frozen reference, one-command run, audit, converter."""
import filecmp
import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import ingest as IG
import output_audit as OA
import real_validation as RV
import universe as UN

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(__file__))
SAMPLE = os.path.join(ROOT, "sample_data", "bbg")


def clone(tmp_path):
    d = tmp_path / "bbg"
    shutil.copytree(SAMPLE, d)
    return d


def edit(d, name, fn):
    df = pd.read_csv(d / name, dtype=str)
    fn(df).to_csv(d / name, index=False)


# ------------------------------------------------------------------ ingestion: stops
def test_sample_ingests_with_exact_exclusions():
    ing = IG.ingest(SAMPLE)
    assert len(ing.returns) == 80 and len(ing.excluded) == 4
    reasons = " ".join(ing.excluded.reason)
    for k in ("short history", "missing month", "repeated zero", "constant"):
        assert k in reasons
    assert ing.meta.is_dead.sum() == 2 and "1900-01-01" == ing.received


@pytest.mark.parametrize("name,fn,msg", [
    ("returns.csv", lambda d: d.drop(columns=["kind"]), "kind"),
    ("returns.csv", lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    ("returns.csv", lambda d: d.assign(date=d.date.str.replace(r"-31$", "-15", regex=True).where(d.index == 0, d.date)), "month"),
    ("meta.csv", lambda d: d.drop(columns=["currency"]), "currency"),
    ("meta.csv", lambda d: d.assign(currency=d.currency.where(d.index != 5, "")), "currency"),
    ("meta.csv", lambda d: d.assign(expense_ratio=d.expense_ratio.where(d.index != 2, "1.1")), "percent"),
    ("meta.csv", lambda d: d.iloc[1:], "meta"),
])
def test_ingest_stops_with_clear_error(tmp_path, name, fn, msg):
    d = clone(tmp_path)
    edit(d, name, fn)
    with pytest.raises(IG.IngestError) as e:
        IG.ingest(d)
    assert msg in str(e.value).lower()


def test_wrong_unit_stops_everything(tmp_path):
    r = subprocess.run([sys.executable, "run_all.py", "--bbg", "sample_data/bbg_bad", "--out", str(tmp_path / "o"),
                        "--snapshot", str(tmp_path / "s"), "--factors-dir", SAMPLE],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "wrong units" in r.stderr and "nothing was converted" in r.stderr
    assert not (tmp_path / "o" / "numeri_report.csv").exists()


def test_missing_file_stops(tmp_path):
    d = clone(tmp_path)
    os.remove(d / "field_log.csv")
    with pytest.raises(IG.IngestError):
        IG.ingest(d)


def test_non_anonymous_ids_need_flag_and_map_stays_private(tmp_path):
    d = clone(tmp_path)
    for n, c in (("returns.csv", "fund_id"), ("meta.csv", "fund_id")):
        edit(d, n, lambda x, c=c: x.assign(**{c: x[c].str.replace("F", "REAL_NAME_", regex=False)}))
    with pytest.raises(IG.IngestError) as e:
        IG.ingest(d)
    assert "anonymised" in str(e.value)
    ing = IG.ingest(d, anonymise=True)
    assert all(i.startswith("F") for i in ing.returns) and ing.id_map is not None
    assert "REAL_NAME_" not in IG.quality_report(ing)


# ------------------------------------------------------------------ universe
@pytest.fixture(scope="module")
def uni(tmp_path_factory):
    ing = IG.ingest(SAMPLE)
    fac, miss = UN.load_factors(["US", "Europe", "Global"], SAMPLE)
    assert not miss
    snap = tmp_path_factory.mktemp("snap")
    return UN.build_bloomberg_universe(ing, fac, "t", str(snap)), ing, fac, snap


def test_groups_and_fallback_rule(uni):
    U = uni[0]
    G = UN.groups_table(U).set_index(["asset_class", "region"])
    assert G.loc[("equity", "US"), "n_funds"] == 24 and bool(G.loc[("equity", "US"), "enough_peers"])
    assert G.loc[("equity", "Europe"), "n_funds"] == 22
    r = G.loc[("real_assets", "Europe")]
    assert not r.enough_peers and "FALLBACK" in r.peer_rule
    assert G.loc[("equity", "US"), "median_expense"] > 0
    d = UN.peer_distributions(U)
    assert {"p05", "p50", "p95"} <= set(d.columns) and d.n.min() >= 1


def test_reference_is_frozen_but_idempotent(uni):
    U, ing, fac, snap = uni
    U2 = UN.build_bloomberg_universe(ing, fac, "t", str(snap))            # same content: reused
    assert len(U2.reference) == len(U.reference)
    ing.returns[next(iter(ing.returns))] = ing.returns[next(iter(ing.returns))] * 1.01
    with pytest.raises(FileExistsError):
        UN.build_bloomberg_universe(ing, fac, "t", str(snap))


def test_benchmark_and_expense_feed_closet_rule(uni):
    U = uni[0]
    F = UN.findings_table(U)
    f = F.set_index("fund_id")
    assert f.loc["F001", "index_state"] == "index_like" or f.loc["F001", "bench_r2"] > 0.9   # index-like with declared benchmark
    cl = f.loc["F009"]                       # index-like, expense 1.10% (above group median)
    ix = f.loc["F001"]                       # index-like, expense 0.04%
    assert cl.bench_r2 > 0.9 and "above the peer median" in cl.closet_reason
    assert "at or below the peer median" in ix.closet_reason
    assert not any(c in F.columns for c in ("date", "value", "nav"))


def test_snapshot_roundtrip_and_no_profile_leak(uni, tmp_path):
    U = uni[0]
    UN.write_snapshot(U, str(tmp_path), "t")
    V = UN.load_snapshot(str(tmp_path))
    assert set(V.returns) == set(U.returns) and set(V.factors) == set(U.factors)
    assert V.reference.shape == U.reference.shape and len(V.excluded) == 4
    a, b = UN.analyse_fund(U, "F025"), UN.analyse_fund(V, "F025")
    assert a.technical["priority_index"] == pytest.approx(b.technical["priority_index"], abs=1e-6)


def test_real_assets_and_appraised_are_scored_with_notes(uni):
    U = uni[0]
    p = UN.analyse_fund(U, "F069")
    assert p.res.profile.pricing == "appraised_nav" and p.res.score is not None
    assert any("closest available" in n for n in p.notes)


# ------------------------------------------------------------------ validation
def test_wilson_interval():
    lo, hi = RV.wilson(0, 40)
    assert lo == 0 and 8 < hi < 10
    assert RV.wilson(0, 0)[0] != RV.wilson(0, 0)[0]


def test_injection_is_graded_and_no_retuning(uni):
    inj = RV.injection(uni[0], max_funds=24).set_index("scenario")
    assert not RV.RETUNED
    assert inj.loc["smoothing s=0.6", "detection_pct"] >= inj.loc["smoothing s=0.3", "detection_pct"] \
        > inj.loc["unmodified (baseline)", "detection_pct"]
    assert inj.loc["losses shifted above zero", "detection_pct"] > 50


# ------------------------------------------------------------------ audit
def test_audit_catches_raw_and_nonanonymous(tmp_path):
    pd.DataFrame(dict(date=["2020-01-31"] * 3, fund_id=["SPY"] * 3, value=[1, 2, 3])).to_csv(tmp_path / "x.csv", index=False)
    pd.DataFrame(dict(id=["a"])).to_csv(tmp_path / "id_map.csv", index=False)
    (tmp_path / "n.txt").write_text("fund REAL_NAME_1 ok")
    iss = OA.audit(str(tmp_path), private_ids=["REAL_NAME_1"])
    txt = " ".join(iss)
    for k in ("raw series fields", "holds dates", "non-anonymous", "file name", "original fund identifier", "repeated fund ids"):
        assert k in txt


# ------------------------------------------------------------------ the one command (end to end)
def run(tmp, name, *extra):
    out, snap = tmp / name, tmp / (name + "_snap")
    r = subprocess.run([sys.executable, "run_all.py", "--dry-run", "--out", str(out), "--snapshot", str(snap),
                        "--private-dir", str(tmp / "priv"), *extra], cwd=ROOT, capture_output=True, text=True)
    return r, out


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    t = tmp_path_factory.mktemp("e2e")
    r1, o1 = run(t, "a")
    r2, o2 = run(t, "b")
    return r1, o1, r2, o2


def test_run_all_end_to_end(e2e):
    r1, o1, *_ = e2e
    assert r1.returncode == 0, r1.stderr + r1.stdout
    for n in ("universe_summary.csv", "groups.csv", "findings_by_fund.csv", "data_quality_report.txt", "numeri_report.csv",
              "numeri_report.json", "peer_distributions.csv", "flag_rates_real.csv", "injection_real.csv",
              "three_funds.csv", "run_log.json", "audit_report.txt", "charts/three_funds.png", "charts/flag_rates.png",
              "charts/group_sizes.png", "charts/injection.png"):
        assert (o1 / n).exists(), n
    assert (o1 / "audit_report.txt").read_text().strip() == "clean"
    assert "1900-01-01" in (o1 / "run_log.json").read_text() and "scipy" in (o1 / "run_log.json").read_text()
    T = pd.read_csv(o1 / "three_funds.csv")
    assert len(T) == 3 and set(T["Pricing type"]) == {"exchange_traded", "nav_listed_holdings", "appraised_nav"} \
        and "Europe" in set(T.Region)
    N = pd.read_csv(o1 / "numeri_report.csv")
    assert {"key", "figure", "unit", "source_file", "note"} == set(N.columns)
    for k in ("universe.n_funds", "universe.flag_pct.autocorrelation", "universe.injection.smoothing_s0.6.detection_pct",
              "parameters.retuned_on_these_data", "synthetic.d_control.pct_high_concern", "data.received"):
        assert k in set(N.key), k
    assert not N.key.duplicated().any()


def test_run_all_is_deterministic(e2e):
    _, o1, _, o2 = e2e
    for root, _, files in os.walk(o1):
        for n in files:
            if n in ("run_log.json",) or n.endswith(".png"):
                continue
            p1 = os.path.join(root, n)
            p2 = p1.replace(str(o1), str(o2))
            assert open(p1, "rb").read() == open(p2, "rb").read(), n


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "fund_returns.csv")), reason="data/ not shipped")
def test_run_all_yahoo_degraded_mode(tmp_path):
    out = tmp_path / "y"
    r = subprocess.run([sys.executable, "run_all.py", "--yahoo", "--out", str(out), "--snapshot", str(tmp_path / "s")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    N = pd.read_csv(out / "numeri_report.csv").set_index("key").figure
    assert int(N["universe.n_funds"]) == 40
    assert "DEGRADED" in (out / "data_quality_report.txt").read_text()
    F = pd.read_csv(out / "findings_by_fund.csv")
    assert F.flag_near_zero.sum() == 3 and F.flag_autocorrelation.sum() == 0   # same as the audited 40-fund run


# ------------------------------------------------------------------ Bloomberg workbook converter
def test_excel_converter_roundtrip(tmp_path):
    from openpyxl import Workbook, load_workbook
    import make_bloomberg_template as MT
    import bbg_from_excel as BX
    p = tmp_path / "t.xlsx"
    MT.build(str(p), 3)
    wb = load_workbook(p)
    assert "BDH(" in wb["NAV"]["A3"].value and "BDP(" in wb["Universe"]["G2"].value
    wb = Workbook()
    rd = wb.active; rd.title = "README"; rd["B3"] = "2026-11-14"; rd["B6"] = "NAV"
    for r, v in zip(range(8, 18), "NAV TRI CRNCY EXP INC CAT GEO DOM TYP ST".split()):
        rd[f"B{r}"] = v
    u = wb.create_sheet("Universe")
    u.append(["ticker", "region", "asset_class", "pricing", "benchmark_ticker", "is_dead", "currency",
              "expense_ratio_percent", "inception", "category_label", "geo_focus", "domicile", "fund_type", "status"])
    nav = wb.create_sheet("NAV")
    nav.append([None]); 
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    for k, tk in enumerate(["AAA US Equity", "BBB LX Equity"]):
        u.append([tk, "US" if k == 0 else "Europe", "equity", "nav_listed_holdings", None, 0, "USD" if k == 0 else "EUR", 0.75,
                  "2010-01-31", "Large", "", "", "", ""])
        nav.cell(2, 2 * k + 1, tk)
        v = 100 * np.cumprod(1 + rng.normal(0.006, 0.04, 48))
        for i, (d, x) in enumerate(zip(dates, v)):
            nav.cell(3 + i, 2 * k + 1, d.to_pydatetime()); nav.cell(3 + i, 2 * k + 2, float(x))
    wb.create_sheet("Benchmarks")
    f = tmp_path / "w.xlsx"; wb.save(f)
    out, priv = tmp_path / "bbg", tmp_path / "private"
    assert BX.main([str(f), "--out", str(out), "--private-dir", str(priv)]) == 0
    assert pd.read_csv(priv / "id_map.csv").original_id.tolist() == ["AAA US Equity", "BBB LX Equity"]
    assert "AAA" not in (out / "returns.csv").read_text() and "AAA" not in (out / "meta.csv").read_text()
    m = pd.read_csv(out / "meta.csv")
    assert m.expense_ratio.tolist() == [0.0075, 0.0075]
    ing = IG.ingest(out)                                    # the converter output passes the real checks
    assert len(ing.returns) == 2 and ing.received == "2026-11-14"
