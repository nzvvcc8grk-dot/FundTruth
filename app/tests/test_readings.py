"""Checkpoint B: diagnostics (known answers), readings, panel, context-aware app."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import diagnostics as dg
import readings as rd
from context import analyse, derive_closet_gate, index_likeness
from csv_connector import load_csv_universe
from fund_profile import build_profile
from panel import analyse_panel, panel_to_frame, panel_to_html
from synthetic import honest_fund, make_factors, smoothed_fund, synthetic_reference

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(__file__))
IDX = pd.date_range("2018-01-31", periods=60, freq="ME")


# ------------------------------------------------------------ diagnostics, known answers
def test_drawdown_known_value():
    r = pd.Series([0.10, -0.20, 0.05])
    assert dg.drawdown(r).min() == pytest.approx(0.88 / 1.1 - 1)
    assert dg.risk_shape(r)["max_drawdown"] == pytest.approx(-0.2)


def test_capture_ratios_known():
    m = pd.Series(np.tile([0.02, -0.01, 0.03, -0.02, 0.01, -0.03], 4))
    c = dg.capture_ratios(1.0 * m, m)
    assert c["up"] == pytest.approx(1.0) and c["down"] == pytest.approx(1.0)
    c2 = dg.capture_ratios(0.5 * m, m)
    assert c2["up"] == pytest.approx(0.5)


def test_rolling_autocorr_alternating_series():
    x = pd.Series(np.tile([0.02, -0.02], 40))
    assert dg.rolling_autocorr(x, 36).iloc[-1] == pytest.approx(-1.0, abs=1e-9)


def test_desmoothing_iid_is_neutral_and_smoothed_rises():
    r, f = honest_fund("equity", 5, 96)
    assert dg.desmoothed_vol(r, f)["ratio"] < 1.25
    s, fs = smoothed_fund(5)
    assert dg.desmoothed_vol(s, fs)["ratio"] > 1.4


def test_factor_loadings_recover_known_betas():
    f = make_factors(96, 3)
    rng = np.random.default_rng(0)
    r = f["RF"] + 0.001 + 1.2 * f["Mkt-RF"] + 0.3 * f["SMB"] - 0.2 * f["HML"] + rng.normal(0, 0.001, 96)
    fl = dg.factor_loadings(r, f)
    assert fl["beta_Mkt-RF"] == pytest.approx(1.2, abs=0.05)
    assert fl["beta_SMB"] == pytest.approx(0.3, abs=0.08) and fl["beta_HML"] == pytest.approx(-0.2, abs=0.08)
    assert fl["lo_Mkt-RF"] < 1.2 < fl["hi_Mkt-RF"]
    assert fl["alpha"] == pytest.approx(0.012, abs=0.004) and fl["r2"] > 0.99


def test_alpha_subperiods_and_insufficient():
    f = make_factors(96, 3)                 # ends 2025-12
    r = f["RF"] + 0.002 + 0.9 * f["Mkt-RF"]
    sub = dg.alpha_subperiods(r, f, {"2024": ("2024-01-01", "2024-12-31"), "late": ("2025-10-01", "2025-12-31")})
    assert sub["2024"]["alpha"] == pytest.approx(0.024, abs=1e-6) and sub["2024"]["n"] == 12
    assert np.isnan(sub["late"]["alpha"]) and sub["late"]["n"] == 3


def test_benchmark_fit_identical_and_short():
    r, _ = honest_fund("equity", 2, 60)
    b = dg.benchmark_fit(r, r)
    assert b["r2"] == pytest.approx(1.0) and b["te"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(dg.benchmark_fit(r.iloc[:10], r.iloc[:10])["r2"])


def test_edge_cases_short_and_constant():
    r, f = honest_fund("equity", 2, 60)
    assert np.isnan(dg.desmoothed_vol(r.iloc[:10], f)["ratio"])
    assert np.isnan(dg.desmoothed_vol(pd.Series(0.01, index=r.index), None)["ratio"])
    assert dg.factor_loadings(r.iloc[:10], f) is None
    assert dg.factor_loadings(f["RF"] + 0.01, f) is None
    assert len(dg.rolling_autocorr(r.iloc[:20], 36)) == 0


def test_peer_percentile():
    assert dg.peer_percentile(5, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(0.5)
    assert np.isnan(dg.peer_percentile(5, []))


def test_diagnostics_independent_of_other_funds():
    r, f = honest_fund("equity", 7, 96)
    a = dg.fund_stats(r, f)
    _ = [dg.fund_stats(honest_fund("bond", s, 96)[0], f) for s in range(3)]
    assert dg.fund_stats(r, f) == a


# ------------------------------------------------------------ index-likeness (amendment 1)
def test_index_likeness_rules():
    assert index_likeness(0.99, "bond")[0] == "cannot_establish"
    assert "bond" in index_likeness(0.99, "bond")[1]
    assert index_likeness(0.99, "equity", "Technology sector")[0] == "cannot_establish"
    assert index_likeness(0.99, "equity")[0] == "index_like" and index_likeness(0.99, "equity")[2] == "factors"
    assert index_likeness(0.80, "equity")[0] == "not_index_like"
    close = dict(n=60, r2=0.99, te=0.01)
    far = dict(n=60, r2=0.99, te=0.06)
    assert index_likeness(0.50, "bond", None, close) == ("index_like", index_likeness(0.50, "bond", None, close)[1], "benchmark")
    assert index_likeness(0.99, "equity", None, far)[0] == "not_index_like"      # benchmark wins over factors
    assert derive_closet_gate(0.99, 0.01, 0.006, close, "bond")[0] == "active"
    assert derive_closet_gate(0.99, None, 0.006, close, "bond")[0] == "passive"
    assert "NOT detected" in derive_closet_gate(0.99, None, 0.006)[1]


# ------------------------------------------------------------ readings and panel
def _panel(fid, pricing="exchange_traded", **meta):
    u = load_csv_universe(os.path.join(ROOT, "sample_data/bbg_small/returns.csv"), os.path.join(ROOT, "sample_data/bbg_small/meta.csv"))
    F = make_factors(96, 0)
    prof = u.profiles()[fid]
    if pricing:
        prof.pricing = pricing
    for k, v in meta.items():
        setattr(prof, k, v)
    return analyse_panel(u.returns[fid], prof, {"US": F, "Europe": F, "Global": F},
                         synthetic_reference(n_per_category=80), "synthetic", None, "", None)


def test_level_thresholds_use_existing_flag_levels():
    assert rd.level_from_p(0.049) == rd.UNUSUAL and rd.level_from_p(0.05) == rd.LOOK
    assert rd.level_from_p(0.099) == rd.LOOK and rd.level_from_p(0.10) == rd.NOTHING
    assert rd.level_from_p(float("nan")) == rd.NA


def test_every_reading_has_the_required_parts():
    p = _panel("SYN_APPRAISED", "appraised_nav")
    keys = [x.key for x in p.readings]
    assert keys == ["price_regularity", "style", "alpha", "risk_shape", "distribution", "peers"]
    for x in p.readings:
        assert x.level in (rd.NOTHING, rd.LOOK, rd.UNUSUAL, rd.DESC, rd.NA)
        if x.level != rd.NA:
            assert x.measured and x.found and x.uncertainty and x.ask and x.not_validated
        assert "nan" not in x.found.lower()


def test_pricing_changes_role_not_numbers():
    a = _panel("SYN_APPRAISED", "appraised_nav")
    b = _panel("SYN_APPRAISED", "exchange_traded")
    ra = {x.key: x for x in a.readings}; rb = {x.key: x for x in b.readings}
    assert ra["price_regularity"].tag == "primary" and rb["price_regularity"].tag == "informational"
    assert ra["price_regularity"].level == rb["price_regularity"].level          # same evidence
    assert ra["price_regularity"].found == rb["price_regularity"].found
    assert "market" in rb["distribution"].reason.lower() and rb["distribution"].tag == "informational"
    assert a.res.score["score"] == b.res.score["score"]                          # score untouched


def test_smoothed_appraised_fund_is_read_as_unusual_primary():
    p = _panel("SYN_APPRAISED", "appraised_nav")
    r = {x.key: x for x in p.readings}["price_regularity"]
    assert r.level == rd.UNUSUAL and r.tag == "primary"
    assert "unusual" in p.headline


def test_headline_does_not_count_informational():
    p = _panel("SYN_APPRAISED", "exchange_traded")
    h = rd.headline([x for x in p.readings if x.key == "price_regularity"])
    assert "informational" in h and "unusual reading" not in h.split("(")[0]


def test_us_etf_and_european_fund_read_differently():
    us = _panel("SYN_US_ETF", "exchange_traded")
    eu = _panel("SYN_EU_EQ", "nav_listed_holdings")
    assert "US Fama/French" in us.ctx and "Europe" in eu.ctx and "EUR" in eu.ctx
    assert us.res.factor_set.file != eu.res.factor_set.file
    assert any("currency EUR differs" in n for n in eu.notes) and not any("differs" in n for n in us.notes)
    assert us.res.profile.region != eu.res.profile.region


def test_bond_fund_gets_weak_factor_warning_and_closet_not_established():
    p = _panel("SYN_BOND")
    assert any("weak" in n or "little of bond" in n for n in p.notes + [p.res.closet_reason])
    assert "cannot be established" in p.res.closet_reason


def test_small_reference_makes_peer_reading_not_applicable():
    u = load_csv_universe(os.path.join(ROOT, "sample_data/bbg_small/returns.csv"), os.path.join(ROOT, "sample_data/bbg_small/meta.csv"))
    F = make_factors(96, 0)
    p = analyse_panel(u.returns["SYN_US_ETF"], u.profiles()["SYN_US_ETF"], {"US": F},
                      synthetic_reference(n_per_category=1000).iloc[:6], "tiny")
    peers = {x.key: x for x in p.readings}["peers"]
    assert peers.level == rd.NA and ("not meaningful" in peers.reason or "not applicable" in peers.reason)


def test_unscored_asset_class_has_no_readings_but_no_crash():
    r, f = honest_fund("equity", 3, 96)
    prof = build_profile("R", dict(region="US", currency="USD"))
    p = analyse_panel(r, prof, {"US": f}, synthetic_reference(n_per_category=30))
    assert p.readings == [] and "not scored" in p.headline


def test_exports():
    p = _panel("SYN_APPRAISED", "appraised_nav")
    df = panel_to_frame(p)
    assert len(df) == 6 and "ask_the_manager" in df.columns
    page = panel_to_html(p)
    assert "never who is at fault" in page and "Price regularity" in page


def test_false_positive_rates_come_from_results():
    fp = rd.false_positive_rates()
    assert 0 < fp["flag_autocorrelation"] < 10 and fp["flag_benford"] > 20


def test_engine_outputs_unchanged_by_the_new_layers():
    """Same inputs through the plain engine and through panel/readings give the same score."""
    from fundtruth import FundTruth
    r, f = honest_fund("equity", 11, 96)
    ref = synthetic_reference(n_per_category=60)
    prof = build_profile("X", dict(region="US", currency="USD", asset_class="equity", pricing="exchange_traded"))
    p = analyse_panel(r, prof, {"US": f}, ref, "syn")
    plain = FundTruth(r.rename("X"), f, peers=ref, category="equity", style="passive").score()
    assert p.technical["priority_index"] == pytest.approx(plain["score"], abs=1e-12)
    assert "not calibrated" in p.technical["label"]


# ------------------------------------------------------------ the app itself
st_testing = pytest.importorskip("streamlit.testing.v1")


def _app(page=None, timeout=300):
    at = st_testing.AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=timeout).run()
    if page:
        at.switch_page(page).run()
    return at


def test_app_starts_on_home_and_every_page_loads():
    for page in (None, "views/check.py", "views/method.py"):
        at = _app(page)
        assert not at.exception, page
        labels = [s.label for s in at.selectbox] + [c.label for c in at.checkbox] + [r.label for r in at.radio]
        assert not any("style" in l.lower() and "active" in l.lower() for l in labels)
        assert not any("reference" in l.lower() for l in labels)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "fund_returns.csv")), reason="data/ not shipped")
def test_app_fund_card_shows_a_verdict_not_a_score():
    at = _app()
    at.session_state["fund"] = "SPY"
    at.switch_page("views/fund.py").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "/ 100" not in text and "score" not in text.lower()
    assert any(w in text.lower() for w in ("nothing unusual", "question"))


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "esg_returns.csv")), reason="data/ not shipped")
def test_app_esg_fund_card_shows_the_promise():
    at = _app()
    at.session_state["fund"] = "SPYX"
    at.switch_page("views/fund.py").run()
    assert not at.exception
    assert any("SPYX" in m.value for m in at.markdown)
    assert any("Promise vs" in e.label for e in at.expander) or any("Promise vs" in m.value for m in at.markdown)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "esg_returns.csv")), reason="data/ not shipped")
def test_app_watchlist_lists_every_fund_flagged_first():
    at = _app("views/watchlist.py")
    assert not at.exception
    df = at.dataframe[0].value
    df = getattr(df, "data", df)                    # Styler -> DataFrame
    assert len(df) >= 46
    q = df["Questions"].tolist()
    flagged = [x > 0 for x in q]
    assert flagged == sorted(flagged, reverse=True)


def test_every_fund_report_is_one_page():
    import fund_report as fr
    ctx = fr.load_context()
    if ctx["UV"] is None or ctx["R"] is None:
        pytest.skip("data/ not shipped")
    from pypdf import PdfReader
    import io
    for f in ["SPYX", "EFIV", "FBNDX"]:
        assert len(PdfReader(io.BytesIO(fr.build_pdf(f, ctx))).pages) == 1


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [t.value for t in at.title] + [h.value for h in at.header])


def test_opening_screen_comes_first_and_each_button_opens_its_page():
    at = _app()
    assert not at.exception
    keys = {b.key for b in at.button}
    assert {"go_watchlist", "go_check", "go_method", "go_home"} <= keys
    assert "Funds make promises." in _texts(at)
    at.button(key="go_method").click().run()
    assert not at.exception and "How FundTruth works" in _texts(at)
    at = _app()
    at.button(key="go_home").click().run()
    assert not at.exception and "Is this fund what it claims to be?" in _texts(at)
    at = _app("views/check.py")                      # a direct link skips the opening screen
    assert "Funds make promises." not in _texts(at)
