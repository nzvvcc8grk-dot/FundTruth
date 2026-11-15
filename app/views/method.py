"""How FundTruth works: the checks in plain language, the evidence that they work, data, limits, glossary."""
import os

import numpy as np
import pandas as pd
import streamlit as st

import ui

HERE = ui.HERE
ui.page_header("How FundTruth works",
               "What each check looks for, how it is measured, how we know it works, and where it stops. "
               "Written for the people who will rely on it.")


def num(key, default=None):
    p = os.path.join(HERE, "results", "yahoo_run", "numeri_report.csv")
    if not os.path.exists(p):
        return default
    s = pd.read_csv(p).set_index("key")["figure"]
    return float(s[key]) if key in s else default


st.subheader("Module 1: are the reported returns credible?")
st.markdown("""
| Check | The question it answers | How it is measured | Grounded in |
|---|---|---|---|
| Price regularity | Are returns too smooth for market prices? | Autocorrelation of monthly excess returns, Ljung-Box test, against similar funds | Getmansky, Lo & Makarov (2004) |
| Style and benchmark | Does an "active" fund just track its index? | R² and tracking error against market factors and the declared benchmark, with the fee | Cremers & Petajisto (2009) |
| Return beyond the market | Is the alpha too good to be true? | Fama-French regression, alpha and its t-statistic, by sub-period | Fama & French (1993) |
| Small gains and losses | Are small losses missing? | Share of small gains in the ±0.5% window, binomial test | Bollen & Pool (2009) |
| Position among peers | Does the fund behave unlike similar funds? | Isolation Forest on volatility, skew, autocorrelation, alpha and R² | Machine learning, unsupervised |
| Shape of risk | Context | Drawdowns, skewness, fat tails | Descriptive only |
""")

st.subheader("Module 2: does an ESG fund keep its promises?")
st.markdown("""
| Check | The question it answers | How it is measured | Data |
|---|---|---|---|
| Promise vs portfolio | Does the fund hold what it says it excludes? | Holdings against the energy sector and Norway's exclusion list, matched to the fund's own promise | Issuer holdings files, Norges Bank list |
| Promise vs behaviour | Does a "fossil-free" fund still move with oil and gas? | Regression of the fund's return over its ordinary index fund on energy's return over the same fund | Monthly prices |
| Label vs index | Is the ESG version really different, and what does it cost? | R², tracking error, active share and the fee gap in pounds per £10,000 | Prices, holdings, fees |

The fund's promises are read from its own documents with a language model. No ESG rating is used:
ratings are paid for and disagree across providers, while the fund's own words are public and binding.
""")

st.subheader("How we know the checks work")
s06, s03, shift = num("universe.injection.smoothing_s0.6.detection_pct"), num("universe.injection.smoothing_s0.3.detection_pct"), \
    num("universe.injection.losses_shifted_above_zero.detection_pct")
fps = [num(f"synthetic.false_positive_pct.{k}") for k in ("autocorrelation", "factor_regression", "near_zero", "peer_anomaly")]
fps = [f for f in fps if f is not None]
items = []
if s06 is not None:
    items.append(("Anomalies planted in real funds", f"{s06:.0f}%", f"strong smoothing found; {s03:.1f}% of mild smoothing; {shift:.1f}% of hidden losses"))
if fps:
    items.append(("False alarms per check, simulated honest funds", f"{min(fps):.1f}-{max(fps):.1f}%",
                  "share of honest funds each check flags by mistake"))
ov_path = os.path.join(HERE, "results", "overview.csv")
if os.path.exists(ov_path):
    ov = pd.read_csv(ov_path)
    real = ov[ov["ESG fund"] != True]
    flagged = real[~real["Integrity of returns"].astype(str).str.startswith("Nothing unusual")]
    # Like-for-like benchmark: honest simulated funds through the same pipeline (same reference set, readings and
    # question rule; see negative_control.py). Only the 'unusual' rate is compared: 'worth a look' rates depend
    # strongly on how each fund type is simulated.
    nc_path = os.path.join(HERE, "results", "negative_control.csv")
    bench = ""
    if os.path.exists(nc_path):
        nc = pd.read_csv(nc_path).set_index(["category", "pricing"])
        if ("all", "all") in nc.index:
            row = nc.loc[("all", "all")]
            bench = (f"; honest simulated funds under the same rule reach 'unusual' "
                     f"{row['unusual_pct']:.0f}% of the time ({int(row['funds'])} funds)")
    n_unusual = int(real["Integrity of returns"].astype(str).str.contains("unusual reading").sum())
    items.append(("Real liquid funds with a question", f"{len(flagged)} of {len(real)}",
                  f"{100 * len(flagged) / len(real):.1f}% 'worth a look' ({', '.join(flagged['Fund'])}); "
                  f"{n_unusual} 'unusual'" + bench))
items.append(("Two methods consistent", "SPYX, EFIV", "energy exposure from returns is in line with the holdings"))
ui.kpis(items)
st.caption("There are no labelled fraud cases to test on, so detection is measured on anomalies planted in real and "
           "simulated funds, and false alarms on funds presumed honest. Real funds are counted with the same rule as the "
           "fund cards (a question when a check reads 'worth a look' or 'unusual'); false alarms per check use the 5% "
           "significance level of the statistical engine.")

st.subheader("How FundTruth gets better with use")
counts, added = ui.comparison_counts()
st.markdown(
    "FundTruth is built to improve with every fund it sees. Three kinds of data feed it:\n\n"
    "- **Funds checked by users** join the comparison set. The checks judge a fund against similar ones, so more "
    "comparable funds mean sharper, more reliable readings.\n"
    "- **Corrections to the pre-filled ESG promises** are labelled examples: they measure and improve how promises "
    "are read from fund documents.\n"
    "- **Managers' answers** to the questions tell which signals were real: the labels that no public dataset has.\n\n"
    "The simulation below shows what the first kind is worth. A check compares each fund with funds of the same type. "
    "With fewer than 20 of them it could not reach 'Unusual' (the smallest possible p-value is 1/(n+1)), so today, "
    "while every type has 10 to 16 funds, FundTruth compares each fund with all 40 funds of every type: the checks "
    "work, but a bond fund is also measured against equity funds. With 20 funds of its own type the price-regularity "
    "check finds all strong smoothing and 57% of mild smoothing; from about 40 it levels off at 61-62%, with at "
    "most 1.5% false alarms. Every fund users add "
    "brings each type closer to 40 comparable funds; beyond that, longer histories matter more than more funds.")
img = os.path.join(HERE, "results", "learning_curve.png")
if os.path.exists(img):
    st.image(img, width=780, caption="Simulated funds with known truth: 20 repetitions, 1,800 test funds, 96 months each. "
                          "Script: learning_curve.py.")
groups = ", ".join(f"{dict(bond='bond', equity='equity').get(k, 'mixed')} {v}" for k, v in sorted(counts.items()))
ui.kpis([("Comparison set today", f"{sum(counts.values())} funds", f"by group: {groups}"),
         ("Added by users", f"{added}", "summary statistics only, on this computer"),
         ("Where the gain stops", "about 40", "comparable funds per group")])

st.subheader("Data")
st.markdown("""
- **Prices**: Yahoo Finance, adjusted for dividends and splits (total return), monthly.
- **Market factors**: Kenneth R. French Data Library, Fama-French factors.
- **Holdings**: daily holdings files published by the issuers (State Street SPDR).
- **Exclusions**: Norges Bank Investment Management exclusion list, via OpenSanctions.
- **Promises and fees**: the funds' own web pages and factsheets, with the date they were checked.
""")

st.subheader("Limits")
st.markdown("""
- A status is a reason to ask a question, never proof of wrongdoing.
- Thresholds are provisional and the reference set of funds is small.
- Public data have survivorship bias: funds that closed are missing.
- Holdings are a one-day snapshot; Norway's criteria differ from each fund's own.
- Private and appraised-value funds, where smoothing matters most, need data that is not public.
""")

with st.expander("Glossary"):
    st.markdown("""
- **Alpha**: the part of a fund's return not explained by market factors.
- **Active share**: the share of a portfolio that differs from its index; 0% is a copy.
- **Autocorrelation**: how much one month's return predicts the next; high values suggest smoothed prices.
- **Closet indexing**: charging active fees for a portfolio that tracks its index.
- **Fama-French factors**: market, size and value returns used to explain fund returns.
- **R²**: how much of a fund's movement is explained by another series (1 = all of it).
- **Tracking error**: the yearly volatility of the difference between a fund and its index.
""")
ui.footer()
