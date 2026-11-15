"""Learning curve: with 10 comparison funds no empirical p-value can fall below 1/11, so nothing is flagged."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import learning_curve as lc


def test_small_reference_cannot_flag_and_larger_one_can(monkeypatch):
    monkeypatch.setattr(lc, "N_TEST", 3)
    monkeypatch.setattr(lc, "SIZES", (10, 40))
    s = lc.summarise(lc.run(1)).set_index(["n_ref", "kind"])
    assert s.loc[(10, "strong smoothing"), "smoothing_unusual_pct"] == 0
    assert s.loc[(40, "strong smoothing"), "smoothing_unusual_pct"] == 100
    assert s.loc[(10, "honest"), "peer_check_available_pct"] == 0
