"""Continuous improvement: a checked fund joins the comparison set once, anonymously, as summary statistics."""
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import ui
from fund_profile import build_profile
from synthetic import honest_fund


def test_contribution_grows_the_comparison_set_once(tmp_path, monkeypatch):
    if not os.path.exists(os.path.join(ui.HERE, "data", "reference_real_v1.csv")):
        return
    monkeypatch.setattr(ui, "CONTRIB", str(tmp_path / "contributed.csv"))
    base, _ = ui.reference_for_checks()
    r, _ = honest_fund("multi-asset", 7, 60)
    prof = build_profile("Private fund name", dict(region="US", currency="USD", asset_class="multi-asset",
                                                   pricing="nav_listed_holdings"))
    ok, _ = ui.contribute(r, prof)
    assert ok
    ref, label = ui.reference_for_checks()
    assert len(ref) == len(base) + 1 and "added by users" in label
    assert not ui.contribute(r, prof)[0]                          # same fund twice: refused
    saved = pd.read_csv(tmp_path / "contributed.csv", index_col=0)
    assert saved.index[0].startswith("U_") and "Private fund name" not in saved.to_csv()
    assert not any(c.startswith("20") for c in saved.columns)       # no returns stored, only statistics
