"""Offline tests for connector.py (no network: downloads are replaced by fakes)."""
import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import connector  # noqa: E402
from synthetic import honest_fund  # noqa: E402


def _fake_prices(r, ticker):
    """Month-end price path whose monthly returns equal r."""
    idx = [r.index[0] - pd.offsets.MonthEnd(1)] + list(r.index)
    vals = [100.0]
    for x in r:
        vals.append(vals[-1] * (1.0 + x))
    return pd.DataFrame({ticker: vals}, index=pd.DatetimeIndex(idx))


@pytest.fixture
def fund():
    r, f = honest_fund("equity", 3)
    return r, f


def test_fetch_returns_matches_source_and_uses_cache(tmp_path, monkeypatch, fund):
    r, f = fund
    calls = []

    def fake_dl(tickers, start):
        calls.append(tickers)
        return _fake_prices(r, tickers[0])

    monkeypatch.setattr(connector, "download_prices", fake_dl)
    out = connector.fetch_returns("abc", "equity", "active", cache_dir=str(tmp_path), factors=f)
    assert len(out) == len(r)
    assert out.to_numpy() == pytest.approx(r.to_numpy(), abs=1e-9)
    # second call comes from the cache: works even if the download now fails
    monkeypatch.setattr(connector, "download_prices",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    again = connector.fetch_returns("ABC", "equity", "active", cache_dir=str(tmp_path), factors=f)
    assert len(again) == len(r)
    assert len(calls) == 1


def test_empty_download_raises_readable_error(tmp_path, monkeypatch, fund):
    _, f = fund
    monkeypatch.setattr(connector, "download_prices", lambda t, s: pd.DataFrame())
    with pytest.raises(connector.ConnectorError, match="No price data"):
        connector.fetch_returns("ZZZZ", cache_dir=str(tmp_path), factors=f)


def test_invalid_ticker_is_rejected():
    for bad in ("", "../etc/passwd", "A B", "X" * 30):
        with pytest.raises(connector.ConnectorError):
            connector.clean_ticker(bad)


def test_too_short_history_is_rejected(tmp_path, monkeypatch, fund):
    r, f = fund
    monkeypatch.setattr(connector, "download_prices",
                        lambda t, s: _fake_prices(r.iloc[:20], t[0]))
    with pytest.raises(connector.ConnectorError, match="cannot be scored"):
        connector.fetch_returns("SHORT", cache_dir=str(tmp_path), factors=f)


def test_factors_fall_back_to_stale_cache(tmp_path, monkeypatch, fund):
    _, f = fund
    path = tmp_path / "factors.csv"
    f.to_csv(path, index_label="date")
    old = time.time() - 90 * 86400
    os.utime(path, (old, old))
    monkeypatch.setattr(connector, "load_ff",
                        lambda src=None: (_ for _ in ()).throw(OSError("offline")))
    got = connector.get_factors(str(tmp_path))
    assert list(got.columns) == list(f.columns) and len(got) == len(f)


def test_factors_without_any_source_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(connector, "load_ff",
                        lambda src=None: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(connector, "HERE", str(tmp_path / "nowhere"))
    with pytest.raises(connector.ConnectorError):
        connector.get_factors(str(tmp_path / "empty"))


def test_search_falls_back_to_builtin_list(monkeypatch):
    monkeypatch.setattr(connector, "_yahoo_search",
                        lambda q, n: (_ for _ in ()).throw(OSError("offline")))
    hits = connector.search("SPY")
    assert hits and hits[0]["ticker"] == "SPY" and hits[0]["source"] == "built-in list"
    assert connector.search("   ") == []


def test_lookup_known_and_unknown():
    assert connector.lookup("spy") == ("equity", "passive")
    assert connector.lookup("NOPE") is None


def test_analyse_end_to_end_returns_valid_score(tmp_path, monkeypatch, fund):
    r, f = fund
    monkeypatch.setattr(connector, "download_prices",
                        lambda t, s: _fake_prices(r, t[0]))
    monkeypatch.setattr(connector, "get_factors", lambda *a, **k: f)
    ft, info = connector.analyse("ABC", "equity", "active", cache_dir=str(tmp_path))
    sc = ft.score()
    assert sc["issued"] and 0.0 <= sc["score"] <= 100.0
    assert info["months"] == len(r) and "synthetic" in info["reference"]
