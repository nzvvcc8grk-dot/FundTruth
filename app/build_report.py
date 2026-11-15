"""Builds FundTruth_report.ipynb (anonymised 5-page report).

All numbers shown in the report are read from results/*.csv by the notebook
cells; the Markdown text contains no hard-coded results. Placeholders in
[SQUARE BRACKETS] must be completed from official sources before submission.
Run: python build_report.py && see README_report.md for the export commands.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["title"] = "FundTruth: statistical integrity screening of reported fund returns"
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                             "language": "python"}
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s, tags=()):
    c = nbf.v4.new_code_cell(s.strip("\n"))
    c.metadata["tags"] = list(tags)
    C.append(c)


# ------------------------------------------------------------------ setup
code('''
import glob, os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False})
R = "results"
LAB = {"a_smoothed": "(a) Smoothed series", "b_nearzero": "(b) Losses shifted above zero",
       "c_closet": "(c) Closet indexer", "d_control": "(d) Honest control (equity)",
       "honest_bond": "Honest bond", "honest_equity": "Honest equity",
       "honest_multi-asset": "Honest multi-asset"}
def table(df, widths=None):
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("-" * (w if widths else 12) for w in (widths or [0]*len(cols))) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    display(Markdown("\\n".join([head, sep] + rows)))
''', tags=["setup"])

# ------------------------------------------------------------------ 1
md(r"""
*Business proposal, product introduction and implementation plan*

# Problem and proposal

**Problem.** Allocators choose and monitor funds largely from monthly returns that the manager, or a valuation chain controlled by the manager, reports. Three statistical patterns in such series are classic reasons for a closer look: (i) *smoothing or stale pricing*, which makes returns look calmer and serially correlated; (ii) *small losses reported as small gains*, which leaves an unusual gap in the return distribution around zero; (iii) *closet indexing*, where a fund that charges active fees delivers almost pure benchmark exposure. None of these patterns proves misconduct, because illiquid assets can produce smoothing legitimately. Screening for them is still cheap, systematic and repeatable, whereas a full operational due diligence is neither.

**Proposal.** FundTruth is a Python library, later offered as a subscription service, that takes a fund's monthly returns (and, optionally, public risk factors) and returns:

* an **integrity score from 0 (high concern) to 100 (no concern)**, calibrated against a fixed, versioned reference set so that the number does not depend on which other funds are analysed;
* a **separate reliability level** (high, medium, low) driven by sample length and test coverage, so a short history is never mistaken for a clean one;
* a **plain-English explanation per test**, with the numbers behind it.

Every output carries the same statement: *a flag is a statistical signal to investigate, never proof of irregularity or fraud.*

**Target users.** Pension funds, insurers, consultants and wealth managers selecting funds; investment platforms that must evidence ongoing monitoring of fund value; asset managers checking their own series before reporting; auditors.

**Approach.** The tool builds on standard, published statistical methods (serial-correlation analysis, factor regression, clustering and outlier detection) applied to public market data, and adds an original integrity-scoring layer on top of them.
""")

# ------------------------------------------------------------------ 2
md(r"""
# Method

Four tests enter the score; a fifth is shown but excluded from it. Inputs are monthly returns in decimal form with month-end dates (at least 36 observations); wrong units, such as percentages, raise an explicit error instead of producing a score.
""")
code('''
tests = pd.DataFrame([
 ["Autocorrelation", "Smoothing / stale pricing: Ljung-Box Q on lags 1-3 of returns in excess of the risk-free rate, positive direction only; a flag needs both a tail position in the peer reference and the test's own p < 0.05", "30"],
 ["Factor regression", "Closet indexing (R2 >= 0.95 and tracking error in the lower tail of peers; skipped if declared passive) and implausibly high, significant alpha (t >= 2, upper tail). Negative alpha is not an integrity issue", "30"],
 ["Near-zero", "Share of gains in [0, 0.5%) among returns in [-0.5%, 0.5%): one-sided binomial test against the fund's own central distribution", "20"],
 ["Peer anomaly", "Isolation Forest on volatility, skewness, autocorrelation, alpha, R2, scored against the frozen peer set; k-means distance shown for information", "20"],
 ["Benford (experimental)", "First-digit chi-square. Returns are small percentages, so honest funds also deviate: shown, not scored", "0"],
], columns=["Test", "What it looks for", "Weight"])
table(tests, widths=[16, 60, 7])
''', tags=["table"])
md(r"""
**Score design.** Each test yields a p-value against the hypothesis "honest fund of the same category", obtained empirically from a frozen reference set. The subscore is a fixed monotone function of it, $s_i = 100\cdot\min(1, p_i/0.10)$; for the peer test the subscore stays at 100 above the 5th percentile and falls linearly below it, so that 5% of honest funds are not penalised by construction. The final score is
$$\text{score} = \min\Big(\sum_i w_i s_i,\; \min_i s_i + 30\Big),$$
so that a single severe alarm cannot be diluted by three normal tests. A test that cannot run (too little data, not applicable) is excluded and the remaining weights are renormalised; if more than one third of the weight is missing, no score is issued. For autocorrelation the p-value used is the larger of the empirical one and the Ljung-Box one, so that the most extreme fund of a small reference is not flagged unless the serial correlation is also significant on its own. The constants 0.10 and 30 are provisional parameters, not calibrated facts. Peer thresholds are category-specific (equity, bond, multi-asset) because illiquid bond funds legitimately show more serial correlation than equity funds.
""")


code('''
import sys; sys.path.insert(0, ".")
from fundtruth import FundTruth
from synthetic import smoothed_fund
r_, f_ = smoothed_fund(4)
ft = FundTruth(r_, f_, category="multi-asset")
sc_ = ft.score()
fig = ft.plot(figsize=(7.2, 2.1))
for a_ in fig.axes: a_.tick_params(labelsize=6); a_.title.set_fontsize(6.5); a_.xaxis.label.set_fontsize(6)
fig.suptitle("")
plt.show()
names = {"autocorrelation": "Autocorrelation", "factor_regression": "Factor regression",
         "near_zero": "Near-zero", "peer_anomaly": "Peer anomaly"}
lines = [f"* Integrity score {sc_['score']:.0f} / 100; reliability: {sc_['reliability']}."]
lines += [f"* {names[k]}: subscore {v:.0f}" + (" (flag)" if ft.all_tests()[k]["flag"] else "") for k, v in sc_["subscores"].items()]
display(Markdown("**Worked example.** Output for synthetic series (a); each subscore comes with a plain-English explanation (omitted here).\\n\\n" + "\\n".join(lines)))
''', tags=["figure", "example"])

# ------------------------------------------------------------------ 3
md(r"""
# Validation on synthetic data

Series with *known* anomalies are generated: (a) a smoothed Madoff-style series (weighted average of current and two lagged true returns), (b) losses between -0.5% and 0 reported as gains between 0 and 0.5%, (c) market return plus tiny noise, (d) an honest factor-model control. Each is repeated over 300 seeds, together with honest bond, equity and multi-asset series. "High concern" means a score below 50 (provisional band).
""")
code('''
s = pd.read_csv(f"{R}/validation_summary.csv", index_col=0)
out = pd.DataFrame({
    "Series": [LAB[i] for i in s.index],
    "Runs": s["n"].astype(int),
    "Mean score": s["mean_score"].map("{:.0f}".format),
    "Share below 50": s["pct_high_concern"].map("{:.1f}%".format),
    "Role": ["detection rate" if i[0] in "abc" and i[1] == "_" else "false-positive rate" for i in s.index]})
table(out, widths=[34, 6, 10, 14, 20])
''', tags=["table"])
code('''
sc = pd.read_csv(f"{R}/scores.csv")
order = list(LAB)
fig, ax = plt.subplots(figsize=(6.4, 1.9))
data = [sc.loc[sc.label == k, "score"].to_numpy() for k in order]
ax.boxplot(data, vert=False, widths=0.6, showfliers=True,
           flierprops=dict(markersize=2), medianprops=dict(color="black"))
ax.set_yticklabels([LAB[k] for k in order])
ax.axvline(50, color="#c0392b", lw=1, ls="--")
ax.set_xlabel("integrity score (100 = no concern); dashed line: 'high concern' below 50")
ax.invert_yaxis(); fig.tight_layout(); plt.show()
''', tags=["figure"])
code('''
fl = pd.read_csv(f"{R}/flag_rates_per_test.csv", index_col=0)
sn = pd.read_csv(f"{R}/sensitivity.csv")
mx = sn.groupby("label")["pct_band_changed"].max()
hon = [i for i in fl.index if i.startswith(("d_", "honest"))]
bf = fl.loc[hon, "flag_benford"]
display(Markdown(
 f"**Reading the results.** Under the provisional bands the three anomalies are recognised in "
 f"{s.loc['a_smoothed','pct_high_concern']:.0f}%, {s.loc['b_nearzero','pct_high_concern']:.0f}% and "
 f"{s.loc['c_closet','pct_high_concern']:.0f}% of runs, while honest series fall below 50 in "
 f"{s.loc[hon,'pct_high_concern'].min():.1f}-{s.loc[hon,'pct_high_concern'].max():.1f}% of runs. "
 f"Varying each weight by +/-20% (and the cap by +/-20%) changes the band of at most "
 f"{mx[mx.index.isin(hon)].max():.0f}% of honest series and of at most {mx[~mx.index.isin(hon)].max():.1f}% of anomalous ones. "
 f"The Benford test flags {bf.min():.0f}-{bf.max():.0f}% of honest series, which is why it carries weight 0. "
 f"The smoothed series also triggers the factor test in {fl.loc['a_smoothed','flag_factor_regression']:.0f}% of runs: "
 f"smoothing biases the contemporaneous regression and creates a spurious positive alpha."))
''', tags=["text"])

code('''
g = pd.read_csv(f"{R}/parameter_grid.csv")
g = g[(g.cap == 30) & (g.cut == 50)].set_index("p_scale")
display(Markdown(f"**Parameter choice** (cap offset 30, cut at 50; subscore = 100 min(1, p / p0)). Moving p0 from {0.20:.2f} to {0.10:.2f} lowers the honest false-positive rate from {g.loc[0.20,'fp_honest']:.1f}% to {g.loc[0.10,'fp_honest']:.1f}% with detection of (a), (b), (c) at {g.loc[0.10,'power_a']:.0f}%, {g.loc[0.10,'power_b']:.0f}% and {g.loc[0.10,'power_c']:.0f}% (full grid in results/parameter_grid.csv); the target was at most 10%."))
''', tags=["text"])

code('''
pc = pd.read_csv(f"{R}/power_curve.csv")
pc = pc[pc.strength.isin([0.0, 0.2, 0.3, 0.4, 0.6])]
sy = pc[pc.reference.str.startswith("synthetic")].set_index("strength")
rl = pc[pc.reference.str.startswith("real")].set_index("strength")
t = pd.DataFrame({"Smoothing s": [f"{x:.1f}" for x in sy.index], "Mean lag-1": sy.mean_ac1.map("{:.2f}".format).to_numpy(),
                  "Flag, gated": sy.flag_gated.map("{:.0f}%".format).to_numpy(), "Flag, no gate": sy.flag_ungated.map("{:.0f}%".format).to_numpy(),
                  "Score <50 (synth. ref.)": sy.score_below_50.map("{:.0f}%".format).to_numpy(),
                  "Score <50 (real ref.)": rl.score_below_50.map("{:.0f}%".format).to_numpy(),
                  "Score <70 (real ref.)": rl.score_below_70.map("{:.0f}%".format).to_numpy()})
display(Markdown(
 "**Sensitivity to graded smoothing.** Reported return = (1-s) r(t) + s/2 r(t-1) + s/2 r(t-2); s = 0 is honest and s = 0.6 is series (a). 200 seeds per row. "
 f"The significance gate costs sensitivity at moderate smoothing (flag rate at s = 0.2: {sy.loc[0.2,'flag_ungated']:.0f}% without the gate, {sy.loc[0.2,'flag_gated']:.0f}% with it; at s = 0.3: {sy.loc[0.3,'flag_ungated']:.0f}% and {sy.loc[0.3,'flag_gated']:.0f}%) and is irrelevant from s = 0.5. "
 f"Its benefit is on small references: with the 40-fund real reference, honest series are flagged {rl.loc[0.0,'flag_ungated']:.0f}% of the time without the gate and {rl.loc[0.0,'flag_gated']:.0f}% with it. "
 f"**Limit:** with that reference the lowest possible empirical p is about 1/41, so the autocorrelation subscore cannot fall below about 24 and the cap holds the score near 54: even strong smoothing lands in the 50-70 review band ({rl.loc[0.6,'score_below_70']:.0f}% below 70), not below 50 ({rl.loc[0.6,'score_below_50']:.1f}%). The 100% detection of (a) above refers to the synthetic reference."))
table(t, widths=[10, 9, 10, 11, 17, 16, 16])
''', tags=["table", "text"])
md(r"""
**What this validation does and does not show.** An independent re-run on a separate machine reproduced every figure above and passed all automated tests. This confirms **reproducibility, not the method**. The honest series come from the same generator that built the synthetic reference, so false-positive rates are optimistic, and the parameter 0.10 was chosen on the same synthetic data (it lowered the honest false-positive rate below the 10% target at unchanged detection). **Detection power is validated only on synthetic data; the real-fund exercise below audits false positives on presumed honest funds and cannot establish detection.**
""")
md(r"""
# Real-fund results
""")
code('''
import sys; sys.path.insert(0, ".")
from analysis import near_zero_audit, carry_check, weight_scenarios
W = pd.read_csv("data/fund_returns.csv", index_col="date", parse_dates=True)
FAC = pd.read_csv("data/factors.csv", index_col="date", parse_dates=True)
META = pd.read_csv("data/funds_meta.csv").set_index("ticker")
RR = pd.read_csv(f"{R}/real_funds_scores_real.csv")
RS = pd.read_csv(f"{R}/real_funds_scores_synthetic.csv")
cat = META.category.value_counts(); typ = META.type.value_counts()
bond_syn = RS[(RS.category == "bond")]
low = RR.sort_values("score").head(5).copy()
display(Markdown(
 f"**Set-up.** {len(META)} US-listed funds ({typ['ETF']} ETFs, {typ['MF']} mutual funds; equity {cat['equity']}, bond {cat['bond']}, multi-asset {cat['multi-asset']}), "
 f"{len(W)} monthly returns from {W.index[0]:%b %Y} to {W.index[-1]:%b %Y} (Yahoo Finance adjusted prices; Fama-French factors); no fund was rejected. "
 f"The reference is the {len(META)} funds themselves, each scored leave-one-out; with fewer than 20 funds per category the peer set is pooled across categories. "
 f"These are well-known, widely held funds, presumed honest (an assumption, not verified), so the exercise audits **false positives**, not detection power.\\n\\n"
 f"**Result.** With the real reference {int((RR.score == 100).sum())} of {len(RR)} funds score 100, none scores below 50 and {int(((RR.score >= 50) & (RR.score < 70)).sum())} fall in the 50-70 review band "
 f"(lowest score {RR.score.min():.0f}). With the synthetic reference {int((bond_syn.score.round(0) <= 50).sum())} of the {len(bond_syn)} bond funds score 50 or less, almost all through the peer test: "
 f"a reference that does not resemble the funds scored produces false alarms, so the real reference is the one to use."))
low["Score"] = low.score.round(1)
t = pd.DataFrame({"Fund": low.ticker, "Type": META.loc[low.ticker, "type"].to_numpy(), "Category": low.category,
                  "Score": low.Score.map("{:.1f}".format),
                  "Autocorr.": low.sub_autocorrelation.round(0).astype(int), "Near-zero": low.sub_near_zero.round(0).map(lambda v: "n/a" if pd.isna(v) else str(int(v))),
                  "Flagged": low.flagged.fillna("").str.replace("autocorrelation", "autocorr.")})
table(t, widths=[8, 6, 12, 7, 10, 10, 18])
''', tags=["table", "real"])
code('''
BEF = pd.read_csv(f"{R}/before_refinement/real_funds_scores_real.csv").set_index("ticker")
nz = near_zero_audit(W, META, FAC)
sh = carry_check(W, FAC, "SHY")
spy = nz["table"].set_index("ticker").loc["SPY"]
bk, hy = RR.set_index("ticker").loc["BKLN"], RR.set_index("ticker").loc["HYG"]
ac = pd.read_csv("data/reference_real_v1.csv", index_col="fund")
ben = int(RR.flagged.fillna("").str.contains("benford").sum())
display(Markdown(
 "**Hand review of the lowest and highest scores.**\\n\\n"
 f"* **SHY (now {RR.set_index('ticker').loc['SHY','score']:.0f}; {int(BEF.loc['SHY','score'])} before the refinement).** Lag 1-3 autocorrelation is {sh['raw']['rho'][0]:.2f}/{sh['raw']['rho'][1]:.2f}/{sh['raw']['rho'][2]:.2f} (Ljung-Box p = {sh['raw']['p']:.3f}) "
 f"on raw returns but {sh['excess']['rho'][0]:.2f}/{sh['excess']['rho'][1]:.2f}/{sh['excess']['rho'][2]:.2f} (p = {sh['excess']['p']:.2f}) in excess of the risk-free rate. Its volatility is very low ({ac.loc['SHY','vol']:.1%} a year) and, if it holds short-maturity Treasuries as its name suggests [fund description to verify], slowly varying carry rather than smoothing was the likely cause of the earlier flag, a false positive removed by a change **chosen after seeing this fund** (see the declaration in the validation section). "
 f"Across the {len(RR)} funds, {int((BEF.score < 100).sum())} scored below 100 before and {int((RR.set_index('ticker').score < 100).sum())} after.\\n"
 f"* **Near-zero (FBNDX {RR.set_index('ticker').loc['FBNDX','score']:.0f}, SPY {RR.set_index('ticker').loc['SPY','score']:.0f}, AOM {RR.set_index('ticker').loc['AOM','score']:.0f}).** "
 f"{nz['flagged']} of {nz['applicable']} applicable funds are flagged at 5% against {nz['expected_flagged']:.1f} expected, and P(at least {nz['flagged']}) is {nz['p_at_least']:.2f}: consistent with chance. "
 f"SPY, an index ETF priced by the market, shows {int(spy.n_gain)} gains and {int(spy.n_loss)} losses in the +/-0.5% window (median {nz['median_in_window']:.0f} returns per fund in the window): a **false positive by construction**. "
 f"Pooled over funds, gains make up {nz['share_gains']:.1%} of window returns against {nz['expected_share']:.1%} expected; clustering by month (funds share months) the evidence is weak (t = {nz['t_clustered']:.1f}, one-sided p = {nz['p_clustered']:.2f}), a hint that the null may be mildly anti-conservative.\\n"
 f"* **Illiquid bond ETFs.** BKLN and HYG, the expected false-positive risks, score {bk.score:.0f} and {hy.score:.0f}: their lag-1 autocorrelation is {ac.loc['BKLN','ac1']:.2f} and {ac.loc['HYG','ac1']:.2f}. As exchange-traded ETFs they carry market prices, not appraised NAVs: the smoothing risk is largely absent, and untested for NAV-based reporting.\\n"
 f"* **Highest scores.** {int((RR.score == 100).sum())} funds tie at 100, so the top of the ranking carries no information; the experimental Benford test flags {ben} of {len(RR)} funds ({ben/len(RR):.0%}) and stays outside the score."))
''', tags=["text", "real"])
code('''
sc = pd.read_csv(f"{R}/scores.csv")
ws = weight_scenarios(sc, RR)
ws = ws[~ws.Scenario.str.startswith("Near-zero 1")]
display(Markdown("**Would a lower near-zero weight help?** No. The three flagged funds are bound by the cap (worst subscore + 30), not by the weighted average, so changing the weight leaves their scores unchanged; removing the test from the cap lifts them but detection of series (b), which only this test sees, falls to about 5%. The weight is therefore kept and the limitation stated."))
table(pd.DataFrame({"Scenario": ws.Scenario, "Detect (b)": ws["Detect (b)"].map("{:.0f}%".format), "Honest <50": ws["Honest below 50"].map("{:.1f}%".format),
                    "SPY": ws.SPY.map("{:.1f}".format), "FBNDX": ws.FBNDX.map("{:.1f}".format), "AOM": ws.AOM.map("{:.1f}".format)}), widths=[34, 11, 11, 7, 8, 7])
''', tags=["table", "real"])

# ------------------------------------------------------------------ 4
md(r"""
# Business section

Facts below come from the sources linked beside them (collected 12 Nov 2026). Items in [BRACKETS] could not be verified and are deliberately left open.

**Customers and need.**

| Segment | Need | Evidence |
|:-------------|:--------------------------------|:-----------------------|
| Allocators (pension funds, insurers, consultants, wealth managers) | Repeatable first screen before and during manager due diligence | [interviews and willingness to pay: to collect] |
| Distribution platforms | Evidence of ongoing monitoring of the funds offered | Distributors, including platforms, must assess price and value under the FCA Consumer Duty ([FCA platforms portfolio letter, 28 Sep 2023](https://www.fca.org.uk/publication/correspondence/platforms-supervision-strategy-portfolio-letter-2023.pdf)); the four outcomes are on the [FCA Consumer Duty page](https://www.fca.org.uk/firms/consumer-duty/about) |
| Asset managers | Self-check of reported series before publication | [to verify] |
| Auditors | Analytical-procedure support on valuation and returns | [to verify] |

**Market context.** The [Investment Association](https://www.theia.org/news/press-releases/uk-investment-management-industry-hits-record-ps111-trillion-aum) reports GBP 11.1 trillion managed in the UK at end 2025 (GBP 10.0 trillion in 2024) and GBP 5.2 trillion managed for UK clients. This is assets under management, **not** the size of the market for integrity screening, and willingness to pay is [unverified]. For private markets the FCA's [valuation practices review (5 Mar 2025)](https://www.fca.org.uk/publications/multi-firm-reviews/private-market-valuation-practices) found valuation-related conflicts only partly identified, uneven independence of the valuation function and ad hoc valuations rarely formalised, which raises the risk of stale valuations; its [conflicts of interest review (23 Oct 2026)](https://www.fca.org.uk/publications/multi-firm-reviews/conflicts-interest-fair-customer-treatment-private-markets) is a separate review of conflicts and fair customer treatment, not of valuations.

**Revenue model.** Subscription priced by number of funds monitored, in tiers [boundaries and prices to be defined; competitor prices are not public in the sources collected]. Cost drivers: data licences if public sources are replaced by paid feeds, hosting, support.

**Competitors and prior art** (descriptions are the vendors' own; prices [to verify]). [Morningstar Direct](https://morningstardirect.morningstar.com/clientcomm/Watchlists_Descriptions.pdf): screening, fund monitoring and due-diligence watchlists, with an agentic assistant announced in [Oct 2026](https://www.morningstar.com/news/business-wire/20261006744754/morningstar-launches-direct-ai-an-agentic-platform-built-on-independent-insights). [AlternativeSoft](https://www.thewealthmosaic.com/vendors/alternativesoft/hedge-funds-due-diligence-solution/): fund data, risk analytics and customisable scoring models. [SEI Novus](https://seic.com/node/10596): over- and under-performance analytics within a manager's book. Closest prior art: [HFMRD](https://de.mathworks.com/matlabcentral/fileexchange/68683-hfmrd-hedge-funds-misreported-returns-detector), a free MATLAB detector of misreported hedge fund returns using style factors and the T-bill rate. Competitors therefore exist; FundTruth's intended differences, to be tested and not claimed as proven, are a fixed, versioned reference so that scores are comparable over time, a separate reliability level, cautious wording and open, reproducible code. Automated tools flag anomalies but determined fraud can evade them.

**Regulatory position.** The product is statistical decision-support, not advice or an accusation. [Exact scope of the Consumer Duty and in-force dates: to confirm in FCA PS22/9; proposed scope changes in CP26/23 are a consultation, not final.]
"""
)

# ------------------------------------------------------------------ 5
md(r"""
# Implementation plan, limits and sources

**Roadmap (proposal).**

| Phase | Content | Status |
|:----------|:--------------------------------------------------|:----------------|
| 1 | Library, validation module, automated tests, synthetic validation | done |
| 2 | Real public fund data (40 mutual funds and ETFs), real-fund reference, hand review of extreme scores, autocorrelation refinement (excess returns, significance gate) | done |
| 3 | Larger real reference (category-specific), API and dashboard | planned |
| 4 | Private and hedge funds, subject to data access | future development, not demonstrated |

**Limits (stated openly).**

* Detection power is validated on synthetic data only; on real funds only false positives were audited (funds presumed honest, mostly market-priced). Independent reproduction is not independent validation.
* The real reference has 40 funds, pooled across categories: rank-based tests flag the most extreme fund by construction, so a flag needs a manual check of magnitude (done above).
* Reference and thresholds are provisional; the autocorrelation refinement was chosen after seeing the real funds. False positives are expected for funds with legitimately illiquid holdings; a reference of N funds cannot give p-values below about 1/(N+1).
* Near-zero test: low power with few returns in +/-0.5%. Benford: unsuited to returns, experimental. Public data: survivorship bias, prices not audited NAVs, US factors weak for non-US and bond funds; private funds not covered.

**Organisation and quality control [TO CONFIRM WITH THE TEAM].** Work in three streams (data and validation; statistical tests; score, report and demo) with each module reviewed by a different person. The repository holds automated checks: units and date errors raise explicit errors; edge cases (constant series, short history, missing values) do not crash; invariants (deterministic score, independence from the group analysed, bounds, shuffling control, monotonicity under progressive smoothing, similar scores for two share classes) are tested.

**Data and code.** Fund prices: Yahoo Finance through the open-source yfinance library; factors: Kenneth R. French Data Library (monthly Fama/French three factors). Code, tests and results: [ANONYMOUS REPOSITORY LINK TO ADD].

**References.** Getmansky, M., Lo, A. W. and Makarov, I. (2004), An econometric model of serial correlation and illiquidity in hedge fund returns, *Journal of Financial Economics* 74(3), 529-609. Bollen, N. P. B. and Pool, V. K. (2009), Do hedge fund managers misreport returns? Evidence from the pooled distribution, *Journal of Finance* 64(5), 2257-2288. Fama, E. F. and French, K. R. (1993), Common risk factors in the returns on stocks and bonds, *Journal of Financial Economics* 33(1), 3-56.
""")

nb.cells = C
nbf.write(nb, "FundTruth_report.ipynb")
print("notebook written:", len(C), "cells")
