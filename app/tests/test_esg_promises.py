"""ESG promises module: the three readings behave as designed on series with a known answer."""
import numpy as np
import pandas as pd
import pytest

import esg_promises as ml
from readings import LOOK, NA, NOTHING, DESC


def _series(seed=0, n=120, w=0.04, leak=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end="2026-09-30", periods=n, freq="ME")
    energy = pd.Series(rng.normal(0.006, 0.07, n), idx)
    rest = pd.Series(rng.normal(0.008, 0.045, n), idx)
    parent = w * energy + (1 - w) * rest
    fossil_free = rest + rng.normal(0, 0.0005, n)            # parent minus its energy stocks
    leaky = (1 - leak) * fossil_free + leak * parent           # keeps part of the energy exposure
    return pd.DataFrame({"PAR": parent, "NRG": energy, "FREE": fossil_free, "LEAK": leaky})


def _claim(promise="full", er=0.002, per=0.0003):
    return dict(name="Test fund", index="Test index", parent="PAR", parent_name="parent", energy_proxy="NRG",
                expense_ratio=er, parent_expense_ratio=per, fossil_promise=promise,
                exclusions=["oil", "gas", "coal"], quotes=["excludes oil, gas and coal"], source="test")


def test_implied_underweight_recovers_the_true_energy_weight():
    R = _series(w=0.05)
    s = ml.fossil_slope(R.FREE, R.PAR, R.NRG)
    assert abs(s["implied_underweight"] - 0.05) < 0.01 and s["t"] < -ml.T_CRIT


def test_full_promise_kept_is_nothing_unusual_and_broken_is_worth_a_look():
    R = _series(w=0.05)
    ok = ml.reading_fossil_behaviour(_claim(), "FREE", R.FREE, R.PAR, R.NRG)
    bad = ml.reading_fossil_behaviour(_claim(), "PAR2", R.PAR + 1e-4, R.PAR, R.NRG)
    assert ok.level == NOTHING and bad.level == LOOK and bad.ask


def test_partial_promise_is_descriptive_only():
    R = _series()
    r = ml.reading_fossil_behaviour(_claim("partial"), "FREE", R.FREE, R.PAR, R.NRG)
    assert r.level == DESC and r.tag == "informational"


def test_near_index_fund_with_higher_fee_is_worth_a_look_and_cost_is_in_pounds():
    R = _series(w=0.0)
    r = ml.reading_label_vs_index(_claim(er=0.0020, per=0.0003), "FREE", R.FREE, R.PAR)
    assert r.level == LOOK and "£17 a year more on £10,000" in r.found


def test_cheaper_fund_is_never_flagged_for_cost():
    R = _series(w=0.0)
    r = ml.reading_label_vs_index(_claim(er=0.0002, per=0.0030), "FREE", R.FREE, R.PAR)
    assert r.level == NOTHING


def test_holdings_reading_finds_energy_and_excluded_names():
    h = pd.DataFrame({"name": ["Apple Inc", "Exxon Mobil Corp", "Altria Group Inc"], "ticker": ["AAPL", "XOM", "MO"],
                      "weight": [0.90, 0.06, 0.04], "sector": ["Information Technology", "Energy", "Consumer Staples"]})
    excl = {ml.norm_name("Altria Group, Inc."): ("Altria Group, Inc.", "tobacco")}
    r = ml.reading_promise_vs_holdings(_claim(), "T", h, excl)
    assert r.level == LOOK and "Altria" in r.found and "6.0%" in r.found


def test_missing_holdings_is_not_applicable():
    assert ml.reading_promise_vs_holdings(_claim(), "T", None, {}).level == NA


def test_active_share_bounds():
    h = pd.DataFrame({"name": ["A", "B"], "ticker": ["A", "B"], "weight": [0.5, 0.5], "sector": ["", ""]})
    g = pd.DataFrame({"name": ["C"], "ticker": ["C"], "weight": [1.0], "sector": [""]})
    assert ml.active_share(h, h) == pytest.approx(0.0) and ml.active_share(h, g) == pytest.approx(1.0)


def test_claim_register_is_complete():
    c = ml.load_claims()
    for t, v in c["funds"].items():
        for k in ("parent", "energy_proxy", "expense_ratio", "parent_expense_ratio", "fossil_promise", "quotes", "source"):
            assert k in v, (t, k)
        assert v["fossil_promise"] in ml.PROMISE_LABEL and v["quotes"]


def test_norway_match_counts_only_when_the_fund_promise_covers_the_reason():
    h = pd.DataFrame({"name": ["Apple Inc", "Altria Group Inc"], "ticker": ["AAPL", "MO"],
                      "weight": [0.98, 0.02], "sector": ["-", "-"]})
    excl = {ml.norm_name("Altria Group Inc"): ("Altria Group Inc", "Product-based - Production of tobacco")}
    tob = dict(_claim("partial"), exclusions=["tobacco", "thermal coal"])
    assert ml.reading_promise_vs_holdings(tob, "T", h, excl).level == LOOK
    assert ml.reading_promise_vs_holdings(_claim("partial"), "T", h, excl).level == NOTHING


def test_energy_from_sector_etf_when_issuer_gives_no_sector():
    h = pd.DataFrame({"name": ["Apple", "Exxon"], "ticker": ["AAPL", "XOM"], "weight": [0.95, 0.05], "sector": ["-", "-"]})
    assert ml.energy_weight(h) is None and ml.energy_weight(h, {"XOM"}) == pytest.approx(0.05)


def test_immaterial_fee_gap_is_not_flagged():
    R = _series(w=0.0)
    r = ml.reading_label_vs_index(_claim(er=0.0010, per=0.000945), "FREE", R.FREE, R.PAR)
    assert r.level == NOTHING and "immaterial" in r.found


def test_keyword_reading_of_a_pasted_promise():
    c = ml.keyword_claims("The index excludes companies that own fossil fuel reserves. It also excludes tobacco.")
    assert c["fossil_promise"] == "full" and "tobacco" in c["exclusions"] and c["quotes"]
    assert ml.keyword_claims("Only thermal coal and oil sands are excluded.")["fossil_promise"] == "partial"
    assert ml.keyword_claims("A global equity fund.")["fossil_promise"] == "none"
    # a pasted "reserves free" promise must not count coal-power utilities as breaches (as for built-in SPYX)
    c = ml.keyword_claims("The fund tracks the S&P 500 Fossil Fuel Reserves Free Index, eliminating companies "
                          "that own fossil fuel reserves (thermal coal, oil and natural gas reserves).")
    assert "fossil fuel reserves" in c["exclusions"]
    assert ml.keyword_claims("Excludes thermal coal mining.")["exclusions"] == ["thermal coal"]


def test_holdings_without_tickers_are_matched_by_name():
    a = pd.DataFrame({"name": ["Apple Inc", "Exxon Mobil Corp"], "ticker": ["", ""], "weight": [0.9, 0.1], "sector": ["", ""]})
    b = pd.DataFrame({"name": ["APPLE INC", "EXXON MOBIL CORP"], "ticker": ["AAPL", "XOM"], "weight": [0.9, 0.1], "sector": ["-", "-"]})
    assert ml.active_share(a, b) == pytest.approx(0.0)


def test_reserves_promise_does_not_cover_coal_power_utilities():
    """Norway's 'coal or coal-based energy' reason mixes miners and power utilities; a promise about
    fossil-fuel RESERVES covers only the former (energy sector), so a utility is information, not a breach."""
    h = pd.DataFrame({"name": ["Apple Inc", "American Electric Power"], "ticker": ["AAPL", "AEP"],
                      "weight": [0.99, 0.01], "sector": ["-", "-"]})
    excl = {ml.norm_name("American Electric Power"): ("American Electric Power",
                                                      "Product-based - Production of coal or coal-based energy")}
    reserves = dict(_claim(), exclusions=["fossil fuel reserves: thermal coal", "oil reserves", "natural gas reserves"])
    r = ml.reading_promise_vs_holdings(reserves, "T", h, excl, energy_tickers=({"XOM"}, set()))
    assert r.level == NOTHING and "coal-based power" in r.found and not r.ask
    coal = dict(_claim(), exclusions=["coal, oil or gas"])          # a promise that excludes coal itself still counts
    assert ml.reading_promise_vs_holdings(coal, "T", h, excl, energy_tickers=({"XOM"}, set())).level == LOOK


def test_question_names_the_holdings_that_raised_it():
    h = pd.DataFrame({"name": ["Apple Inc", "Altria Group Inc"], "ticker": ["AAPL", "MO"],
                      "weight": [0.98, 0.02], "sector": ["-", "-"]})
    excl = {ml.norm_name("Altria Group Inc"): ("Altria Group Inc", "Product-based - Production of tobacco")}
    r = ml.reading_promise_vs_holdings(dict(_claim("partial"), exclusions=["tobacco"]), "T", h, excl)
    assert r.level == LOOK and "Altria Group Inc" in r.ask and r.found.index("Altria") < 200


