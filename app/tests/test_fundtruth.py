"""Unit and invariant checks for FundTruth (run: pytest -q)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import autocorrelations, ljung_box_q, regress_factors  # noqa: E402
from fundtruth import DEFAULT_WEIGHTS, FundTruth  # noqa: E402
from synthetic import (closet_fund, control_fund, make_factors,  # noqa: E402
                       smoothed_fund, synthetic_reference)
from validation import percent_to_decimal, to_month_end, valida_dati  # noqa: E402

KEYS = {"test", "score", "statistic", "p_value", "flag", "explanation",
        "status", "n_obs"}
TESTS = ["test_autocorrelation", "test_near_zero", "test_benford",
         "test_factor_regression", "test_peer_anomaly"]


def _series(n=60, seed=0, name="X"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2025-12-31", periods=n, freq="ME")
    return pd.Series(rng.normal(0.005, 0.03, n), index=idx, name=name)


# ---------------------------------------------------------- validation / units
def test_percent_returns_raise():
    with pytest.raises(ValueError):
        FundTruth(_series() * 100)


def test_percent_factors_raise():
    r, f = control_fund(1)
    with pytest.raises(ValueError):
        FundTruth(r, f * 100)
    FundTruth(r, percent_to_decimal(f * 100))      # explicit conversion works


def test_bad_index_raises():
    s = _series()
    with pytest.raises(ValueError):                # not month-end
        valida_dati(pd.Series(s.to_numpy(), index=s.index - pd.Timedelta(days=3)))
    dup = pd.concat([s, s.iloc[:1]])
    with pytest.raises(ValueError):                # duplicated + unsorted
        valida_dati(dup)
    fixed = to_month_end(pd.Series(s.to_numpy(), index=s.index - pd.Timedelta(days=3)))
    valida_dati(fixed)


def test_nan_dropped_with_warning():
    s = _series()
    s.iloc[5] = np.nan
    r, _, w = valida_dati(s)
    assert len(r) == 59 and any("missing" in x for x in w)


# -------------------------------------------------------------------- contract
@pytest.mark.parametrize("name", TESTS)
def test_contract_keys(name):
    r, f = control_fund(3)
    out = getattr(FundTruth(r, f), name)()
    assert set(out) == KEYS
    assert out["status"] in {"ok", "insufficient_data", "not_applicable"}
    assert isinstance(out["flag"], bool) and isinstance(out["explanation"], str)
    if out["status"] == "ok":
        assert 0 <= out["score"] <= 100


def test_insufficient_data():
    ft = FundTruth(_series(30))
    for name in TESTS:
        out = getattr(ft, name)()
        assert out["status"] == "insufficient_data" and math.isnan(out["score"])
    sc = ft.score()
    assert sc["issued"] is False and math.isnan(sc["score"])
    assert sc["reliability"] == "low"


def test_constant_series_no_crash():
    s = _series()
    s[:] = 0.004
    ft = FundTruth(s)
    for name in TESTS:
        assert getattr(ft, name)()["status"] == "not_applicable"
    assert ft.score()["issued"] is False


def test_extreme_cases_no_crash():
    s = _series()
    s[:] = 0.0
    s.iloc[::7] = 0.01                  # zeros mixed with a few values
    FundTruth(s).score()
    r, f = closet_fund(1)               # exact-replica-like, tiny residual
    exact = (f["RF"] + f["Mkt-RF"]).rename("EXACT")
    out = FundTruth(exact, f).test_factor_regression()
    assert out["status"] == "ok" and out["flag"] and out["score"] < 5


# ----------------------------------------------------------------- weights
def test_weights_sum_and_validation():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 100) < 1e-9
    assert DEFAULT_WEIGHTS["benford"] == 0
    ft = FundTruth(*control_fund(1))
    with pytest.raises(ValueError):
        ft.score({"autocorrelation": 50, "factor_regression": 30,
                  "near_zero": 20, "peer_anomaly": 20})       # sums to 120
    with pytest.raises(ValueError):
        ft.score({"autocorrelation": 150, "factor_regression": -50})
    with pytest.raises(ValueError):
        ft.score({"nonsense": 100})


def test_renormalisation_and_not_issued():
    r, f = control_fund(1)
    sc = FundTruth(r, None).score()           # no factors: 30% excluded
    assert sc["issued"] and "factor_regression" not in sc["tests_included"]
    assert abs(sum(sc["weights_used"].values()) - 100) < 1e-9
    assert sc["weights_used"]["autocorrelation"] == pytest.approx(100 * 30 / 70)
    small = synthetic_reference(n_per_category=5).assign(category="equity")
    sc2 = FundTruth(r, None, peers=small).score()   # + peer excluded: 50% > 1/3
    assert sc2["issued"] is False and math.isnan(sc2["score"])
    assert sc2["reliability"] == "low"


def test_reliability_separate_from_score():
    r, f = control_fund(1, n=96)
    assert FundTruth(r, f).score()["reliability"] == "high"
    r2, f2 = control_fund(1, n=60)
    assert FundTruth(r2, f2).score()["reliability"] == "medium"
    r3, f3 = control_fund(1, n=40)
    assert FundTruth(r3, f3).score()["reliability"] == "low"


def test_cap_rule():
    r, f = smoothed_fund(1)
    sc = FundTruth(r, f, category="multi-asset").score()
    assert sc["cap_applied"]
    assert sc["score"] <= sc["worst_subscore"] + 30 + 1e-9


# ------------------------------------------------------------ known answers
def test_autocorr_matches_statsmodels():
    x = _series(80, 4).to_numpy()
    rho = autocorrelations(x, 3)
    assert np.allclose(rho, sm.tsa.acf(x, nlags=3, fft=False)[1:])
    q, p = ljung_box_q(x, 3)
    lb = acorr_ljungbox(x, lags=[3])
    assert q == pytest.approx(float(lb["lb_stat"].iloc[0]))
    assert p == pytest.approx(float(lb["lb_pvalue"].iloc[0]))


def test_known_ar1_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=20000)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = 0.6 * x[t - 1] + e[t]
    assert autocorrelations(x, 1)[0] == pytest.approx(0.6, abs=0.02)


def test_regression_matches_numpy():
    r, f = control_fund(5)
    reg = regress_factors(r, f)
    X = np.column_stack([np.ones(len(r)), f[["Mkt-RF", "SMB", "HML"]].to_numpy()])
    y = (r - f["RF"]).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    assert reg["alpha"] == pytest.approx(beta[0] * 12)
    assert reg["r2"] == pytest.approx(1 - resid.var() / y.var())
    assert reg["te"] == pytest.approx(
        np.sqrt(resid @ resid / (len(y) - 4)) * np.sqrt(12))


def test_benford_perfect_sample():
    counts = np.round(1000 * np.log10(1 + 1 / np.arange(1, 10))).astype(int)
    vals = np.concatenate([np.full(c, d * 0.001) for d, c in zip(range(1, 10), counts)])
    idx = pd.date_range(end="2025-12-31", periods=len(vals), freq="ME")
    out = FundTruth(pd.Series(vals, index=idx, name="B"))._benford()
    assert out["p_value"] > 0.99 and not out["flag"]
    skew = np.full(600, 0.0051)                   # all leading digit 5
    idx = pd.date_range(end="2025-12-31", periods=600, freq="ME")
    sk = pd.Series(skew * np.where(np.arange(600) % 2, 1, -1), index=idx)
    sk.iloc[:50] += np.linspace(0, 1e-6, 50)
    assert FundTruth(sk)._benford()["flag"]


def test_near_zero_handmade():
    rng = np.random.default_rng(1)
    big = np.concatenate([rng.uniform(0.01, 0.05, 25), -rng.uniform(0.01, 0.05, 25)])
    small = rng.uniform(0.0, 0.005, 14)            # 14 gains, 0 small losses
    x = rng.permutation(np.concatenate([big, small]))
    idx = pd.date_range(end="2025-12-31", periods=len(x), freq="ME")
    out = FundTruth(pd.Series(x, index=idx, name="NZ"))._near_zero()
    assert out["status"] == "ok" and out["flag"] and out["statistic"] == 1.0


def test_near_zero_too_few_observations_is_not_applicable():
    idx = pd.date_range(end="2025-12-31", periods=60, freq="ME")
    wide = pd.Series(np.tile([0.05, -0.05, 0.04, -0.06], 15), index=idx, name="W")
    assert FundTruth(wide)._near_zero()["status"] == "not_applicable"


# --------------------------------------------------------------- invariants
def test_determinism_and_group_independence():
    r, f = control_fund(7)
    a = FundTruth(r, f).score()
    for s in range(3):                              # analyse other funds in between
        FundTruth(*control_fund(100 + s)).score()
        FundTruth(*smoothed_fund(100 + s), category="multi-asset").score()
    b = FundTruth(r, f).score()
    assert a["score"] == b["score"] and a["subscores"] == b["subscores"]


def test_bounds_never_nan_when_issued():
    for seed in range(15):
        for cat in ("equity", "bond", "multi-asset"):
            from synthetic import honest_fund
            r, f = honest_fund(cat, 500 + seed)
            sc = FundTruth(r, f, category=cat).score()
            assert sc["issued"] and 0 <= sc["score"] <= 100


def test_negative_control_shuffle():
    """Shuffling month order kills autocorrelation but not distribution tests."""
    r, f = smoothed_fund(11)
    before = FundTruth(r, f, category="multi-asset")
    sh = pd.Series(np.random.default_rng(0).permutation(r.to_numpy()),
                   index=r.index, name=r.name)
    after = FundTruth(sh, f, category="multi-asset")
    assert before.test_autocorrelation()["flag"]
    assert after.test_autocorrelation()["score"] > before.test_autocorrelation()["score"]
    nb, na = before.test_near_zero(), after.test_near_zero()
    assert nb["statistic"] == na["statistic"] and nb["p_value"] == na["p_value"]
    assert before.test_benford()["statistic"] == pytest.approx(
        after.test_benford()["statistic"])


def test_monotonic_smoothing_autocorr_subscore():
    r, f = _series(96, seed=0, name="IID"), None    # iid base: autocorr ~ 0
    scores = []
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        mixed = ((1 - lam) * r + lam * r.shift(1)).iloc[1:].rename("M")
        scores.append(FundTruth(mixed, f).test_autocorrelation()["score"])
    assert all(b <= a + 1e-9 for a, b in zip(scores, scores[1:])), scores
    assert scores[-1] < scores[0]


def test_positive_control_index_vs_itself():
    r, f = closet_fund(3)
    sc = FundTruth(r, f).score()
    assert sc["subscores"]["factor_regression"] < 25
    passive = FundTruth(r, f, style="passive").test_factor_regression()
    assert passive["score"] == 100.0                # exempt when declared passive


def test_share_class_similar_scores():
    r, f = control_fund(9)
    twin = (r - 0.0001).rename("TWIN")              # 1.2% p.a. extra fee class
    a, b = FundTruth(r, f).score(), FundTruth(twin, f).score()
    assert abs(a["score"] - b["score"]) <= 10


# ------------------------------------------------------------- analysis.py
def test_analysis_module_on_synthetic_panel():
    from analysis import carry_check, near_zero_audit, weight_scenarios
    from synthetic import honest_fund
    f = make_factors(96, 3)
    cols, meta = {}, []
    for i in range(8):
        r, _ = honest_fund("equity", 300 + i, 96)
        r.index = f.index
        cols[f"H{i}"] = r
        meta.append(dict(ticker=f"H{i}", category="equity", style="active"))
    w = pd.DataFrame(cols)
    a = near_zero_audit(w, pd.DataFrame(meta).set_index("ticker"), f)
    assert 0 <= a["flagged"] <= a["applicable"] <= 8
    assert 0 <= a["p_at_least"] <= 1 and np.isfinite(a["t_clustered"])
    c = carry_check(w, f, "H0")
    assert set(c) == {"raw", "excess"} and len(c["raw"]["rho"]) == 3
    synth = pd.DataFrame({"label": ["b_nearzero"] * 2 + ["d_control"] * 2,
                          "sub_autocorrelation": 100.0, "sub_factor_regression": 100.0,
                          "sub_near_zero": [0.0, 0.0, 100.0, 100.0],
                          "sub_peer_anomaly": 100.0})
    real = pd.DataFrame({"ticker": ["SPY", "FBNDX", "AOM"], "sub_autocorrelation": 100.0,
                         "sub_factor_regression": 100.0,
                         "sub_near_zero": [43.0, 31.0, 48.0], "sub_peer_anomaly": 100.0})
    s = weight_scenarios(synth, real).set_index("Scenario")
    assert s.loc["Current: 30/30/20/20", "Detect (b)"] == 100.0       # cap binds
    assert s.loc["Near-zero weight 0 (40/35/0/25)", "Detect (b)"] == 0.0
    # lowering the weight leaves cap-bound scores unchanged
    assert s.loc["Current: 30/30/20/20", "SPY"] == s.loc["Near-zero 10 (35/35/10/20)", "SPY"]


# ------------------------------------- excess returns + significance gate
def test_persistent_riskfree_not_flagged_on_excess_returns():
    rng = np.random.default_rng(5)
    f = make_factors(96, 9)
    rf = np.zeros(96)
    rf[0] = 0.003
    for t in range(1, 96):                       # slowly varying carry
        rf[t] = 0.0005 + 0.95 * rf[t - 1] + rng.normal(0, 0.0004)
    f["RF"] = np.clip(rf, 0, None)
    r = (f["RF"] + rng.normal(0, 0.0015, 96)).rename("CARRY")
    with_rf = FundTruth(r, f)
    assert with_rf.test_autocorrelation()["flag"] is False
    assert with_rf.details["autocorrelation"]["basis"] == "excess over RF"
    raw = FundTruth(r, f.drop(columns="RF"))      # no RF -> raw returns
    assert raw.test_autocorrelation()["flag"] is True
    assert raw.details["autocorrelation"]["basis"] == "raw returns"


def test_significance_gate_blocks_top_of_small_reference():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.03, 96)
    x = np.zeros(96)
    for t in range(1, 96):
        x[t] = 0.3 * x[t - 1] + e[t]            # weak autocorrelation
    idx = pd.date_range(end="2025-12-31", periods=96, freq="ME")
    r = pd.Series(x, index=idx, name="WEAK")
    ref = synthetic_reference(n_per_category=25)
    ft = FundTruth(r, peers=ref, category="equity")
    out = ft.test_autocorrelation()
    d = ft.details["autocorrelation"]
    assert d["p_emp"] < 0.05                      # most extreme of a small reference
    assert d["p_lb"] > 0.05 and out["flag"] is False and out["p_value"] == pytest.approx(d["p_lb"])
