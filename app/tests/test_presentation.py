"""Offline tests for narrative.py and charts.py (presentation layer).

The score must be untouched by the presentation: these tests read results, never
change them, and check that every chart specification is valid.
"""
import os
import re
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import narrative as nv  # noqa: E402
from fundtruth import FundTruth, NEAR_ZERO_W  # noqa: E402
from synthetic import honest_fund, smoothed_fund  # noqa: E402


def _run(r, f, category="multi-asset", style="active"):
    ft = FundTruth(r, f, category=category, style=style)
    sc, tests = ft.score(), ft.all_tests()
    return ft, sc, tests


def test_band_thresholds():
    assert nv.band(49.9) == "high concern"
    assert nv.band(50.0) == "review"
    assert nv.band(69.9) == "review"
    assert nv.band(70.0) == "no major concern"


def test_smoothed_fund_verdict_names_the_signal():
    r, f = smoothed_fund(4)
    ft, sc, tests = _run(r, f)
    v = nv.verdict(sc, tests)
    assert v["band"] == "high concern"
    assert "autocorrelation" in v["body"]
    assert "never proof" in v["body"]


def test_verdict_when_no_score_issued():
    v = nv.verdict({"issued": False}, {})
    assert v["band"] == "n/a"


def test_every_test_has_three_readable_paragraphs():
    r, f = honest_fund("equity", 3)
    ft, sc, tests = _run(r, f, "equity", "active")
    for k in nv.ORDER:
        q, found, meaning = nv.story(k, tests[k], ft.details, ft.style, "synthetic placeholder")
        for text in (q, found, meaning):
            assert isinstance(text, str) and len(text) > 20
        assert not re.search(r"\bnan\b", found.lower())


def test_passive_story_mentions_skipped_closet_check():
    r, f = honest_fund("equity", 3)
    ft, sc, tests = _run(r, f, "equity", "passive")
    _, found, _ = nv.story("factor_regression", tests["factor_regression"],
                           ft.details, ft.style, "x")
    assert "closet-indexing check is skipped" in found


def test_presentation_does_not_change_the_score():
    r, f = honest_fund("equity", 5)
    ft = FundTruth(r, f, category="equity", style="active")
    before = ft.score()["score"]
    ft2, sc2, tests2 = _run(r, f, "equity", "active")
    nv.verdict(sc2, tests2)
    for k in nv.ORDER:
        nv.story(k, tests2[k], ft2.details, ft2.style, "x")
    assert ft2.score()["score"] == pytest.approx(before)


def test_chart_specs_are_valid():
    pytest.importorskip("altair")
    import charts
    r, f = honest_fund("equity", 3)
    ft, sc, tests = _run(r, f, "equity", "active")
    rows = pd.DataFrame([{"test": nv.SHORT[k], "score": tests[k]["score"],
                          "flag": bool(tests[k]["flag"])}
                         for k in nv.ORDER if tests[k]["status"] == "ok"])
    specs = [charts.gauge(sc["score"]), charts.subscores(rows),
             charts.returns_hist(ft.returns.to_numpy(), NEAR_ZERO_W),
             charts.autocorr(ft.details["autocorrelation"]["ac"], ft.n_obs),
             charts.growth(ft.returns, ft.factors)]
    for chart in specs:
        assert chart is not None
        assert isinstance(chart.to_dict(), dict)      # raises if the spec is invalid


def test_growth_is_none_without_factors():
    pytest.importorskip("altair")
    import charts
    r, f = honest_fund("equity", 3)
    assert charts.growth(r, None) is None
