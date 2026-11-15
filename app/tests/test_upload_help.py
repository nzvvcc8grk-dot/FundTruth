"""The easier input: pasted tables, factsheet grids, units and dates are read as a person would."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import upload_help as uh
from csv_connector import load_csv_universe


def _read(text):
    return uh.parse_history(uh._rows(text))


def _series(text, fid="F"):
    df, units, _ = _read(text)
    uni = load_csv_universe(uh.to_long(df, units, fid), None)
    return uni.returns[fid], units


def test_excel_paste_with_percent_signs():
    text = "\n".join(f"{d:%Y-%m-%d}\t{v:.2f}%" for d, v in zip(pd.date_range("2020-01-31", periods=40, freq="ME"),
                                                              [1.25, -0.8, 2.1, 0.4] * 10))
    r, units = _series(text)
    assert units == "pct" and len(r) == 40 and abs(r.iloc[0] - 0.0125) < 1e-12


def test_factsheet_grid_from_a_pdf_with_spaces():
    months = "Year Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec YTD"
    rows = [f"{y} " + " ".join(f"{(m % 5 - 2) * 0.5:.1f}" for m in range(12)) + " 3.0" for y in (2021, 2022, 2023)]
    rows.append("2024 0.5 -0.3 1.0")                          # current year, three months only
    df, units, what = _read("\n".join([months] + rows))
    assert "by year" in what and units == "pct" and len(df) == 39
    assert df["date"].iloc[-1] == pd.Timestamp("2024-03-31")


def test_grid_without_a_year_header_cell():
    text = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec\n2022 " + " ".join(["1.0"] * 12)
    df, _, _ = _read(text)
    assert len(df) == 12 and df["value"].eq(1.0).all()


def test_nav_prices_european_decimals_and_day_first_dates():
    navs = [100 + i * 0.7 for i in range(40)]
    text = "Data;Prezzo\n" + "\n".join(f"{d:%d/%m/%Y};{v:.2f}".replace(".", ",")
                                       for d, v in zip(pd.date_range("2020-01-31", periods=40, freq="ME"), navs))
    r, units = _series(text)
    assert units == "nav" and len(r) == 39 and abs(r.iloc[0] - (100.7 / 100 - 1)) < 1e-9


def test_decimal_returns_with_comma_separator_and_header():
    text = "date,return\n" + "\n".join(f"{d:%Y-%m-%d},{v}" for d, v in
                                        zip(pd.date_range("2019-01-31", periods=36, freq="ME"), [0.01, -0.02, 0.005] * 12))
    r, units = _series(text)
    assert units == "dec" and len(r) == 36


def test_space_separated_with_decimal_comma():
    text = "\n".join(f"{d:%Y-%m-%d} {v}" for d, v in zip(pd.date_range("2019-01-31", periods=36, freq="ME"),
                                                       ["1,2", "-0,5", "0,8"] * 12))
    r, units = _series(text)
    assert units == "pct" and abs(r.iloc[0] - 0.012) < 1e-12


def test_nothing_readable_gives_a_plain_message():
    with pytest.raises(ValueError, match="dates and numbers"):
        _read("hello world\nfoo bar")
