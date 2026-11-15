"""Plain-English reading of a FundTruth result.

Presentation layer only: every number comes from FundTruth (score(),
all_tests(), details); nothing is recomputed or changed here.
"""
from __future__ import annotations

import math

from fundtruth import CLOSET_R2, NEAR_ZERO_W, PEER_CUTOFF

BANDS = (50.0, 70.0)          # PROVISIONAL, same bands as validate_synthetic.py
ORDER = ["autocorrelation", "factor_regression", "near_zero", "peer_anomaly", "benford"]
SHORT = {"autocorrelation": "Autocorrelation (smoothing)",
         "factor_regression": "Factor regression (closet indexing, alpha)",
         "near_zero": "Near-zero (small losses vs small gains)",
         "peer_anomaly": "Peer anomaly (unusual vs similar funds)",
         "benford": "Benford first digit (experimental)"}
COLORS = {"high concern": "#b33a3a", "review": "#c58a00", "no major concern": "#2e7d5b",
          "n/a": "#6b7785"}


def band(score):
    if score < BANDS[0]:
        return "high concern"
    return "review" if score < BANDS[1] else "no major concern"


def _names(keys):
    return ", ".join(SHORT[k].split(" (")[0].lower() for k in keys)


def verdict(sc, tests):
    """Headline and paragraph for the top of the page."""
    if not sc["issued"]:
        return dict(band="n/a", headline="No score could be issued",
                    body=("Too many tests could not run on this series (usually too few "
                          "months of data), so FundTruth does not give a score."))
    b = band(sc["score"])
    flagged = [k for k in ORDER if k in sc["weights_used"] and tests[k]["flag"]]
    if b == "no major concern" and not flagged:
        head = "No statistical signal of concern"
        body = ("None of the scored tests found anything unusual in this fund's reported "
                "returns. This does not prove the numbers are right; it means these "
                "particular checks found no irregular pattern.")
    elif b == "no major concern":
        head = "No major concern overall, with one signal to note"
        body = (f"The overall score is in the comfortable range, but the test on "
                f"{_names(flagged)} raised a signal. A single signal, especially on a "
                "small sample, is often a false alarm: read the explanation of that test "
                "before drawing a conclusion.")
    elif b == "review":
        head = "Worth a review"
        body = (f"The score is in the middle range. Signal on: {_names(flagged) or 'none individually strong'}. "
                "This is a reason to ask questions, not a conclusion.")
    else:
        head = "High concern: worth a closer look"
        body = (f"The score is low, driven by: {_names(flagged) or 'several tests'}. "
                "It means the reported returns show a pattern that honest funds rarely "
                "show. It is a signal to investigate, never proof of irregularity or fraud.")
    if sc["cap_applied"]:
        body += (f" Note: the weighted average of the tests is {sc['weighted_average']:.0f}, "
                 "but the score is held down to the weakest test plus 30 points, so one "
                 "serious signal cannot be hidden by several good results.")
    return dict(band=b, headline=head, body=body)


def reliability_text(rel, n_obs):
    return {"high": f"High: {n_obs} monthly observations and every test could run.",
            "medium": f"Medium: {n_obs} monthly observations, or a data warning applies; "
                      "treat the result with some care.",
            "low": f"Low: only {n_obs} monthly observations or too few tests ran; "
                   "the result is fragile."}[rel]


def story(key, res, details, style, reference):
    """(question, found, meaning) for one test, in plain English."""
    if res["status"] != "ok":
        return (_QUESTION[key], res["explanation"],
                "This test could not be computed, so it is left out of the score "
                "and the weights of the other tests are rescaled.")
    d = details.get(key, {})
    flag = bool(res["flag"])
    if key == "autocorrelation":
        ac = d.get("ac", [float("nan")] * 3)
        n = max(res["n_obs"], 1)
        found = (f"Last month's return predicts this month's with a correlation of "
                 f"{ac[0]:+.2f} (two and three months back: {ac[1]:+.2f}, {ac[2]:+.2f}), "
                 f"measured on {d.get('basis', 'returns')}. A purely random series would "
                 f"typically stay within about +/-{1.96 / math.sqrt(n):.2f}.")
        if flag:
            meaning = ("A positive echo this strong is significant and unusual against peers. "
                       "It is the typical signature of smoothed or stale valuations.")
        elif res["statistic"] <= 0:
            meaning = ("The echo is negative or absent, so there is no sign of smoothing. "
                       "The test only looks for a positive echo.")
        else:
            meaning = "A mild positive echo, in line with peers: no signal."
    elif key == "factor_regression":
        found = (f"The market, size and value factors explain {d.get('r2', float('nan')):.0%} "
                 f"of the fund's monthly moves (R-squared). Tracking error is "
                 f"{d.get('te', float('nan')):.1%} a year; alpha (return not explained by the "
                 f"factors) is {d.get('alpha', float('nan')):+.1%} a year "
                 f"(t-statistic {d.get('alpha_t', float('nan')):.1f}).")
        if style == "passive":
            found += " The fund is declared passive, so the closet-indexing check is skipped."
        if flag:
            meaning = ("Either the fund tracks the market far more closely than comparable "
                       "active funds (closet indexing: active fees for index-like returns), "
                       "or its alpha is implausibly high and significant. Check the declared "
                       "style and the fees.")
        elif style == "passive":
            meaning = "A high R-squared is expected for an index fund: nothing unusual."
        elif d.get("r2", 0.0) < CLOSET_R2:
            meaning = (f"R-squared is below {CLOSET_R2:.2f}, so there is no closet-indexing "
                       "signal, and alpha is not high enough to be implausible.")
        else:
            meaning = ("R-squared is high, but tracking error is not unusually low against "
                       "peers, and alpha is not implausible: no signal.")
    elif key == "near_zero":
        nl, ng = d.get("n_loss", 0), d.get("n_gain", 0)
        n = nl + ng
        found = (f"Of the {n} months with a return within +/-{NEAR_ZERO_W:.1%} of zero, "
                 f"{nl} were small losses and {ng} small gains "
                 f"({ng / max(n, 1):.0%} gains, against {d.get('p0', float('nan')):.0%} expected "
                 f"from the fund's own return distribution; p = {res['p_value']:.3f}).")
        if flag:
            meaning = ("There are more small gains and fewer small losses than expected, "
                       "which is how losses rounded up to zero would look. "
                       f"With only {n} months near zero the test has low power: a flag here is "
                       "a reason to look and often a false alarm.")
        else:
            meaning = "Small gains and small losses are balanced as expected: no signal."
    elif key == "peer_anomaly":
        q = res["p_value"]
        found = (f"About {q:.0%} of the reference funds look at least as unusual as this one "
                 f"on volatility, skewness, autocorrelation, alpha and R-squared together. "
                 f"Reference used: {reference}.")
        if flag:
            meaning = (f"Fewer than {PEER_CUTOFF:.0%} of reference funds look this unusual: "
                       "the overall profile is out of the ordinary for the category.")
        else:
            meaning = (f"Nothing unusual (the test reacts only below {PEER_CUTOFF:.0%}). "
                       "While the reference is synthetic, read this as indicative.")
    else:  # benford
        found = f"Chi-square {res['statistic']:.1f}, p = {res['p_value']:.3f}."
        meaning = ("Shown for information only and not in the score: returns are small "
                   "numbers, and honest funds often deviate from Benford's law too.")
    return _QUESTION[key], found, meaning


_QUESTION = {
    "autocorrelation": (
        "Do each month's returns echo the previous months? In liquid markets they rarely "
        "do. Smoothed or stale valuations, typical of illiquid assets and of Madoff-style "
        "reporting, make returns follow each other."),
    "factor_regression": (
        "How much of the fund's ups and downs can be explained by the market and by the "
        "usual size and value factors? Two things look odd: a fund that calls itself active "
        "but moves like the index (closet indexing), and a high, statistically solid alpha "
        "beyond what is plausible."),
    "near_zero": (
        "Are small losses quietly reported as small gains? An honest fund has about as many "
        "tiny losses as tiny gains; reporting that rounds losses up to zero leaves too few "
        "small losses."),
    "peer_anomaly": (
        "Compared with funds of the same category, does this one look unusual when its "
        "volatility, skewness, autocorrelation, alpha and R-squared are taken together?"),
    "benford": (
        "Do the first digits of the returns follow Benford's law, as many natural data do? "
        "Experimental: it is not used in the score."),
}
