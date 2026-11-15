"""FundTruth: integrity checks on reported monthly fund returns (contract v2).

A flag is a statistical signal to investigate, never proof of irregularity.
All thresholds marked PROVISIONAL are assumptions to be calibrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features import autocorr_basis, compute_features, ljung_box_q
from synthetic import synthetic_reference
from validation import MIN_OBS, valida_dati

VERSION = "synthetic-v0"          # reference/calibration version
DISCLAIMER = ("Statistical signal to investigate, not proof of irregularity "
              "or fraud.")
DEFAULT_WEIGHTS = {"autocorrelation": 30.0, "factor_regression": 30.0,
                   "near_zero": 20.0, "peer_anomaly": 20.0, "benford": 0.0}
CATEGORIES = ("equity", "bond", "multi-asset")

# --- PROVISIONAL parameters (to calibrate) ---------------------------------
P_SCALE = 0.10            # subscore = 100 * min(1, p / P_SCALE); 0.10 chosen from
                          # results/parameter_grid.csv (FP target <= 10%)
FLAG_P = 0.05             # a test flags when p < FLAG_P
PEER_CUTOFF = 0.05        # peer subscore < 100 only below this percentile
CAP_OFFSET = 30.0         # final <= worst subscore + CAP_OFFSET
MAX_EXCLUDED = 1.0 / 3.0  # more excluded weight than this -> no score
REL_HIGH_N, REL_LOW_N = 84, 48
REL_LOW_COVERAGE = 0.70
NEAR_ZERO_W = 0.005       # +/-0.5% window
NEAR_ZERO_MIN = 5         # min returns inside the window
CLOSET_R2 = 0.95          # closet check applies only above this R-squared
ALPHA_MIN_T = 2.0         # alpha check applies only above this t-stat
MIN_REF = 20              # min reference funds for a category
_IF_CACHE: dict = {}


def _sub(p):
    return float(np.clip(100.0 * min(1.0, p / P_SCALE), 0.0, 100.0))


def _emp(ref, x, upper=True):
    """Empirical p-value with +1 correction: share of ref >= x (or <= x)."""
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    k = int(np.sum(ref >= x)) if upper else int(np.sum(ref <= x))
    return (1.0 + k) / (len(ref) + 1.0)


def _res(test, status="ok", score=np.nan, statistic=np.nan, p=None, flag=False,
         expl="", n_obs=0):
    return {"test": test, "score": float(score), "statistic": float(statistic),
            "p_value": None if p is None else float(p), "flag": bool(flag),
            "explanation": expl, "status": status, "n_obs": int(n_obs)}


class FundTruth:
    """Integrity checks for one fund.

    returns : monthly returns (decimal), month-end DatetimeIndex, name = id.
    factors : decimal Fama-French factors (Mkt-RF, SMB, HML[, RF]) or None.
    peers   : frozen reference table (features per fund + 'category', 'style');
              default = synthetic placeholder reference.
    category: 'equity' | 'bond' | 'multi-asset' (selects the peer set).
    style   : 'active' | 'passive'. A declared passive fund (index ETF) is
              exempt from the closet-indexing check.
    """

    def __init__(self, returns, factors=None, peers=None, category="equity",
                 style="active"):
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}.")
        if style not in ("active", "passive"):
            raise ValueError("style must be 'active' or 'passive'.")
        self.name = str(returns.name) if getattr(returns, "name", None) else "fund"
        self.returns, self.factors, self.warnings = valida_dati(returns, factors)
        self.category, self.style = category, style
        if peers is None:
            self.reference, self._ref_key = synthetic_reference(), "default"
        else:
            need = {"category", "vol", "skew", "ac1", "lb_q"}
            if not need.issubset(peers.columns):
                raise ValueError(f"peers must contain columns {sorted(need)}.")
            self.reference = peers
            self._ref_key = int(pd.util.hash_pandas_object(peers).sum())
        ref = self.reference
        sub = ref[ref["category"] == category]
        self.pooled = len(sub) < MIN_REF
        self._ref = ref if self.pooled else sub
        self._results: dict = {}
        self._feats = None
        self.details: dict = {}

    # ------------------------------------------------------------------ utils
    @property
    def n_obs(self):
        return len(self.returns)

    def _features(self):
        if self._feats is None:
            self._feats = compute_features(self.returns, self.factors)
        return self._feats

    def _gate(self, test):
        """Return a not-ok result when the series cannot be tested."""
        if self.n_obs < MIN_OBS:
            return _res(test, "insufficient_data", n_obs=self.n_obs, expl=(
                f"Only {self.n_obs} monthly observations (minimum {MIN_OBS}); "
                "test not computed."))
        if float(self.returns.std(ddof=0)) == 0.0:
            return _res(test, "not_applicable", n_obs=self.n_obs,
                        expl="Constant series: test not applicable.")
        return None

    def _cached(self, key, fn):
        if key not in self._results:
            self._results[key] = fn()
        return self._results[key]

    # ------------------------------------------------------------------ tests
    def test_autocorrelation(self):
        return self._cached("autocorrelation", self._autocorr)

    def _autocorr(self):
        name = "Autocorrelation (smoothing)"
        bad = self._gate(name)
        if bad:
            return bad
        f = self._features()
        x, basis = autocorr_basis(self.returns, self.factors)
        q, p_lb = ljung_box_q(x.to_numpy(), 3)
        positive = f["lb_q"] > 0
        p_emp = _emp(self._ref["lb_q"], f["lb_q"])
        # flag only if the fund is an outlier vs peers AND its own Ljung-Box
        # test is significant (avoids rank artefacts with small reference sets)
        p = max(p_emp, p_lb) if positive else 1.0
        med = float(np.nanmedian(self._ref["ac1"]))
        direction = "positive" if positive else "non-positive"
        expl = (f"On {basis}, lag-1/2/3 autocorrelation is {f['ac1']:.2f}/"
                f"{f['ac2']:.2f}/{f['ac3']:.2f} ({direction} overall; Ljung-Box "
                f"Q(3)={q:.1f}, own p={p_lb:.3f}); peer-set median lag-1 is "
                f"{med:.2f}, empirical p vs peers {p_emp:.3f}; the test uses the "
                f"larger of the two (p={p:.3f}). Strong positive autocorrelation "
                "can indicate smoothed or stale pricing.")
        self.details["autocorrelation"] = dict(
            ac=[f["ac1"], f["ac2"], f["ac3"]], basis=basis, p_lb=p_lb, p_emp=p_emp)
        return _res(name, score=_sub(p), statistic=f["lb_q"], p=p,
                    flag=p < FLAG_P, expl=expl, n_obs=self.n_obs)

    def test_near_zero(self):
        return self._cached("near_zero", self._near_zero)

    def _near_zero(self):
        name = "Near-zero (small losses vs small gains)"
        bad = self._gate(name)
        if bad:
            return bad
        x = self.returns.to_numpy()
        w = NEAR_ZERO_W
        n_loss = int(np.sum((x >= -w) & (x < 0.0)))
        n_gain = int(np.sum((x >= 0.0) & (x < w)))
        m = float(np.median(x))
        s = 1.4826 * float(np.median(np.abs(x - m)))
        if n_loss + n_gain < NEAR_ZERO_MIN or s < 1e-9:
            return _res(name, "not_applicable", n_obs=self.n_obs, expl=(
                f"Only {n_loss + n_gain} returns fall within +/-{w:.1%}; "
                "too few for a meaningful test (low power)."))
        cdf = stats.norm.cdf
        mass_gain = cdf((w - m) / s) - cdf((0 - m) / s)
        mass_all = cdf((w - m) / s) - cdf((-w - m) / s)
        p0 = float(np.clip(mass_gain / mass_all, 1e-6, 1 - 1e-6))
        n = n_loss + n_gain
        p = float(stats.binom.sf(n_gain - 1, n, p0))   # P(G >= observed)
        expl = (f"{n_loss} returns in [-{w:.1%}, 0) and {n_gain} in [0, {w:.1%}): "
                f"{n_gain / n:.0%} are gains versus {p0:.0%} expected from the "
                f"fund's own central distribution (one-sided binomial p={p:.3f}). "
                "Too few small losses and too many small gains can indicate "
                "earnings-management-style reporting. Low power with few returns "
                "near zero.")
        self.details["near_zero"] = dict(n_loss=n_loss, n_gain=n_gain, p0=p0)
        return _res(name, score=_sub(p), statistic=n_gain / n, p=p,
                    flag=p < FLAG_P, expl=expl, n_obs=self.n_obs)

    def test_benford(self):
        return self._cached("benford", self._benford)

    def _benford(self):
        name = "Benford first digit (experimental)"
        bad = self._gate(name)
        if bad:
            return bad
        a = np.abs(self.returns.to_numpy())
        a = a[a > 0]
        if len(a) < MIN_OBS:
            return _res(name, "insufficient_data", n_obs=len(a),
                        expl="Too few non-zero returns for a digit test.")
        d = np.floor(a / 10.0 ** np.floor(np.log10(a))).astype(int)
        obs = np.bincount(d, minlength=10)[1:10].astype(float)
        exp = len(d) * np.log10(1.0 + 1.0 / np.arange(1, 10))
        chi2 = float(np.sum((obs - exp) ** 2 / exp))
        p = float(stats.chi2.sf(chi2, 8))
        expl = (f"First-digit chi-square {chi2:.1f} (8 df, p={p:.3f}) on "
                f"{len(d)} returns. EXPERIMENTAL and weight 0: returns are small "
                "percentages and do not span orders of magnitude, so honest "
                "funds also deviate from Benford (high false-positive risk).")
        return _res(name, score=_sub(p), statistic=chi2, p=p, flag=p < FLAG_P,
                    expl=expl, n_obs=len(d))

    def test_factor_regression(self):
        return self._cached("factor_regression", self._factor)

    def _factor(self):
        name = "Factor regression (closet indexing / implausible alpha)"
        bad = self._gate(name)
        if bad:
            return bad
        if self.factors is None:
            return _res(name, "not_applicable", n_obs=self.n_obs,
                        expl="No factors supplied: regression not computed.")
        f = self._features()
        n_ov = len(self.returns.index.intersection(self.factors.index))
        if n_ov < MIN_OBS:
            return _res(name, "insufficient_data", n_obs=n_ov, expl=(
                f"Only {n_ov} months overlap with the factors "
                f"(minimum {MIN_OBS})."))
        if not np.isfinite(f["r2"]):
            return _res(name, "not_applicable", n_obs=n_ov,
                        expl="Regression not defined for this series.")
        ref = self._ref
        if "style" in ref:
            ref = ref[ref["style"] == "active"]
        # (a) closet indexing: very high R2 AND tracking error in the lower tail
        if self.style == "passive":
            p_closet, note = 1.0, "closet check skipped (declared passive)"
        elif f["r2"] < CLOSET_R2:
            p_closet, note = 1.0, f"R2 below {CLOSET_R2:.2f}: no closet signal"
        else:
            p_closet = _emp(ref["te"], f["te"], upper=False)
            note = f"tracking error at the {p_closet:.1%} lower-tail percentile of peers"
        # (b) implausibly high AND significant alpha (negative alpha ignored)
        if f["alpha_t"] < ALPHA_MIN_T:
            p_alpha = 1.0
        else:
            p_alpha = _emp(ref["alpha"], f["alpha"], upper=True)
        p = float(min(1.0, 2.0 * min(p_closet, p_alpha)))   # 2 sub-checks
        closet_binds = p_closet <= p_alpha
        stat = f["te"] * 100 if closet_binds else f["alpha"] * 100
        expl = (f"R2={f['r2']:.2f}, tracking error {f['te']:.1%} a year, "
                f"annualised alpha {f['alpha']:.1%} (t={f['alpha_t']:.1f}); "
                f"{note}; alpha percentile vs peers p={p_alpha:.3f}. Very high "
                "R2 with tiny tracking error suggests closet indexing; an "
                "implausibly high, significant alpha is a plausibility concern. "
                "Negative alpha is not treated as an integrity issue.")
        self.details["factor_regression"] = dict(
            r2=f["r2"], te=f["te"], alpha=f["alpha"], alpha_t=f["alpha_t"])
        return _res(name, score=_sub(p), statistic=stat, p=p, flag=p < FLAG_P,
                    expl=expl, n_obs=n_ov)

    def test_peer_anomaly(self):
        return self._cached("peer_anomaly", self._peer)

    def _peer_model(self, cols):
        key = (self._ref_key, self.category, self.pooled, cols)
        if key not in _IF_CACHE:
            X = self._ref[list(cols)].replace([np.inf, -np.inf], np.nan).dropna()
            X = X.to_numpy(dtype=float)
            sc = StandardScaler().fit(X)
            Z = sc.transform(X)
            iso = IsolationForest(n_estimators=300, random_state=0).fit(Z)
            km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(Z)
            dist = km.transform(Z).min(axis=1)
            _IF_CACHE[key] = (sc, iso, km, iso.decision_function(Z), dist)
        return _IF_CACHE[key]

    def _peer(self):
        name = "Peer anomaly (Isolation Forest + k-means)"
        bad = self._gate(name)
        if bad:
            return bad
        f = self._features()
        cols = ["vol", "skew", "ac1"]
        if self.factors is not None and np.isfinite(f["r2"]):
            cols += ["alpha", "r2"]
        cols = tuple(cols)
        x = np.array([[f[c] for c in cols]])
        if not np.isfinite(x).all():
            return _res(name, "not_applicable", n_obs=self.n_obs,
                        expl="Some peer features are undefined for this fund.")
        if len(self._ref) < MIN_REF or not all(c in self._ref for c in cols):
            return _res(name, "not_applicable", n_obs=self.n_obs, expl=(
                f"Fewer than {MIN_REF} reference funds: peer test not applicable."))
        sc, iso, km, ref_scores, ref_dist = self._peer_model(cols)
        z = sc.transform(x)
        s = float(iso.decision_function(z)[0])
        q = _emp(ref_scores, s, upper=False)       # low score = more anomalous
        d = float(km.transform(z).min(axis=1)[0])
        d_pct = float(np.mean(ref_dist >= d))
        sub = 100.0 if q >= PEER_CUTOFF else 100.0 * q / PEER_CUTOFF
        pool = " (pooled across categories: small reference)" if self.pooled else ""
        expl = (f"Isolation Forest percentile vs the frozen peer set is {q:.1%}"
                f"{pool}; subscore drops only below the {PEER_CUTOFF:.0%} "
                f"percentile. Distance to nearest k-means cluster is larger "
                f"than {1 - d_pct:.0%} of peers (information only). Features: "
                f"{', '.join(cols)}.")
        return _res(name, score=sub, statistic=s, p=q, flag=q < PEER_CUTOFF,
                    expl=expl, n_obs=self.n_obs)

    # ---------------------------------------------------------------- scoring
    def all_tests(self):
        return {"autocorrelation": self.test_autocorrelation(),
                "factor_regression": self.test_factor_regression(),
                "near_zero": self.test_near_zero(),
                "peer_anomaly": self.test_peer_anomaly(),
                "benford": self.test_benford()}

    def score(self, weights=None, cap_offset=None):
        """Combine tests. See module docs: renormalise over tests that ran,
        no score if more than 1/3 of the weight is missing, a severe single
        alarm cannot be diluted (final <= worst subscore + cap_offset)."""
        w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        unknown = set(w) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown test(s) in weights: {sorted(unknown)}")
        w = {k: float(w.get(k, 0.0)) for k in DEFAULT_WEIGHTS}
        if any(v < 0 for v in w.values()):
            raise ValueError("weights must be non-negative.")
        if abs(sum(w.values()) - 100.0) > 1e-9:
            raise ValueError(f"weights must sum to 100 (got {sum(w.values())}).")
        cap = CAP_OFFSET if cap_offset is None else float(cap_offset)
        res = self.all_tests()
        included, excluded = [], []
        for k, v in w.items():
            if v == 0.0:
                excluded.append(f"{k}: weight 0 (shown, not scored)")
            elif res[k]["status"] != "ok":
                excluded.append(f"{k}: {res[k]['status']}")
            else:
                included.append(k)
        excl_w = sum(w[k] for k in w if w[k] > 0 and k not in included)
        issued = excl_w <= 100.0 * MAX_EXCLUDED + 1e-9 and len(included) > 0
        coverage = (100.0 - excl_w) / 100.0
        out = {"score": float("nan"), "issued": issued, "reliability": "low",
               "tests_included": included, "tests_excluded": excluded,
               "weights_used": {}, "cap_applied": False, "version": VERSION,
               "disclaimer": DISCLAIMER, "subscores": {},
               "weighted_average": float("nan"),
               "worst_subscore": float("nan")}
        if not issued:
            return out
        tot = sum(w[k] for k in included)
        wu = {k: 100.0 * w[k] / tot for k in included}
        subs = {k: res[k]["score"] for k in included}
        avg = sum(wu[k] * subs[k] for k in included) / 100.0
        worst = min(subs.values())
        final = float(np.clip(min(avg, worst + cap), 0.0, 100.0))
        if self.n_obs >= REL_HIGH_N and coverage >= 0.999 and not self.warnings:
            rel = "high"
        elif self.n_obs < REL_LOW_N or coverage < REL_LOW_COVERAGE:
            rel = "low"
        else:
            rel = "medium"
        out.update(score=final, reliability=rel, weights_used=wu,
                   cap_applied=bool(worst + cap < avg - 1e-12), subscores=subs,
                   weighted_average=float(avg), worst_subscore=float(worst))
        return out

    # ----------------------------------------------------------------- output
    def report(self):
        sc = self.score()
        res = self.all_tests()
        L = [f"FundTruth integrity report: {self.name}",
             f"Category: {self.category} ({self.style}); {self.n_obs} monthly "
             f"observations; reference version {VERSION}"]
        if sc["issued"]:
            L.append(f"Integrity score: {sc['score']:.0f} / 100 "
                     "(100 = no concern, 0 = high concern)")
        else:
            L.append("Integrity score: not issued (too many tests could not run).")
        L.append(f"Reliability: {sc['reliability']}")
        if sc["cap_applied"]:
            L.append("Note: the score was capped by the worst single test "
                     f"(weighted average {sc['weighted_average']:.0f}).")
        L.append("")
        for k, r in res.items():
            tag = "FLAG" if r["flag"] else r["status"].upper()
            ws = ("" if k in sc["weights_used"] else " [not in score]")
            val = "n/a" if np.isnan(r["score"]) else f"{r['score']:.0f}"
            L.append(f"- {r['test']}: subscore {val} [{tag}]{ws}")
            L.append(f"    {r['explanation']}")
        if sc["tests_excluded"]:
            L.append("")
            L.append("Excluded: " + "; ".join(sc["tests_excluded"]))
        for wmsg in self.warnings:
            L.append(f"Data warning: {wmsg}")
        L += ["", "Limits: few observations, survivorship bias in public "
              "data, private funds not covered, Benford experimental, "
              "false positives are expected.", DISCLAIMER]
        return "\n".join(L)

    def plot(self, figsize=(14, 3.8)):
        import matplotlib.pyplot as plt
        res, sc = self.all_tests(), self.score()
        fig, ax = plt.subplots(1, 3, figsize=figsize)
        x = self.returns.to_numpy()
        ax[0].hist(x, bins=25, color="#4c78a8")
        ax[0].axvspan(-NEAR_ZERO_W, NEAR_ZERO_W, color="#f58518", alpha=0.25)
        ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set_title("Returns (orange: +/-0.5% window)")
        ax[0].set_xlabel("monthly return")
        ac = self._features()
        ax[1].bar([1, 2, 3], [ac["ac1"], ac["ac2"], ac["ac3"]], color="#4c78a8")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_xticks([1, 2, 3])
        ax[1].set_title("Autocorrelation, lags 1-3")
        ax[1].set_xlabel("lag")
        names = [k for k in res if res[k]["status"] == "ok"]
        vals = [res[k]["score"] for k in names]
        cols = ["#e45756" if res[k]["flag"] else "#54a24b" for k in names]
        ax[2].barh(names, vals, color=cols)
        ax[2].set_xlim(0, 100)
        ttl = f"{sc['score']:.0f}" if sc["issued"] else "n/a"
        ax[2].set_title(f"Subscores | score {ttl}, reliability {sc['reliability']}")
        fig.suptitle(f"{self.name}: statistical signals, not proof of irregularity",
                     fontsize=10)
        fig.tight_layout()
        return fig
