"""Offline tests for data.py (no network): parser, units, share classes."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import build_dataset, monthly_returns, parse_ff_factors  # noqa: E402

# Layout of the Kenneth French monthly CSV (header text, monthly block, annual
# block). Values here are made up for the parser test only.
SAMPLE = """This file was created by a test fixture.

,Mkt-RF,SMB,HML,RF
202001,   1.00,  -2.00,   3.00,   0.10
202002,  -8.00,   0.50,  -1.00,   0.12
202003, -99.99,   0.50,  -1.00,   0.12
202004,  13.50,   2.00,  -4.00,   0.00

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
2020,   5.00,   1.00,   1.00,   0.5
"""


def test_parse_ff_units_missing_and_annual_block():
    f = parse_ff_factors(SAMPLE)
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(f) == 3                                   # missing row dropped
    assert f.loc["2020-01-31", "Mkt-RF"] == pytest.approx(0.01)   # percent -> decimal
    assert f.loc["2020-04-30", "Mkt-RF"] == pytest.approx(0.135)
    assert f.index.is_month_end.all()
    assert not (f.index.year == 2021).any()              # annual rows excluded


def test_parse_ff_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ff_factors("nothing here")


def _prices(n=70, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n * 22)
    out = {}
    for t in ("AAA", "AAB", "SHORT"):
        out[t] = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(idx))))
    df = pd.DataFrame(out, index=idx)
    df.loc[df.index[:-500], "SHORT"] = np.nan             # short history
    return df


def test_monthly_returns_month_end_and_values():
    p = pd.DataFrame({"X": [100.0, 110.0, 121.0]},
                     index=pd.to_datetime(["2020-01-15", "2020-02-14", "2020-03-16"]))
    r = monthly_returns(p)
    assert r.index.is_month_end.all()
    assert r["X"].tolist() == pytest.approx([0.10, 0.10])


def test_build_dataset_rejections():
    prices = _prices()
    rets = monthly_returns(prices)
    f = pd.DataFrame({"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0, "RF": 0.001},
                     index=rets.index)
    f["Mkt-RF"] = np.random.default_rng(1).normal(0.007, 0.04, len(f))
    meta = pd.DataFrame({
        "ticker": ["AAA", "AAB", "SHORT", "MISSING"],
        "category": "equity", "style": "active",
        "fund_id": ["F1", "F1", "F2", "F3"]})           # AAB = same fund as AAA
    wide, ok, rej = build_dataset(meta, prices, f)
    assert list(wide.columns) == ["AAA"]
    reasons = dict(zip(rej["ticker"], rej["reason"]))
    assert "duplicate share class" in reasons["AAB"]
    assert "months" in reasons["SHORT"]
    assert "no price data" in reasons["MISSING"]
