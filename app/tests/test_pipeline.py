import os
import subprocess
import sys
import warnings

import pandas as pd
import pytest

from context import analyse
from fund_profile import build_profile, propose_pricing
from synthetic import honest_fund, synthetic_reference

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(__file__))


def test_small_reference_is_flagged_not_reassuring():
    r, f = honest_fund("equity", 5, 96)
    p = build_profile("A", dict(region="US", currency="USD", asset_class="equity"))
    res = analyse(r, p, {"US": f}, synthetic_reference(n_per_category=1000).iloc[:5])
    assert not res.reference_adequate and any("NOT reassurance" in n for n in res.notes)


def test_pricing_proposal_and_levels():
    assert propose_pricing("ETF") == "exchange_traded"
    assert propose_pricing(None, "UK Commercial Property") == "appraised_nav"
    assert propose_pricing(None, "something") is None
    p = build_profile("A", dict(pricing="market_priced"))
    assert p.pricing == "exchange_traded"           # legacy alias


def test_closet_rule_needs_expense_ratio():
    from context import derive_closet_gate
    assert derive_closet_gate(0.99, None, 0.006)[0] == "passive"
    assert derive_closet_gate(0.99, 0.003, 0.006)[0] == "passive"
    assert derive_closet_gate(0.99, 0.010, 0.006)[0] == "active"
    assert derive_closet_gate(0.60, 0.010, 0.006)[0] == "passive"
