# FundTruth

**Funds make promises. FundTruth checks them against the evidence.**

FundTruth tests a fund's reported returns and its ESG promises against public data. For each fund it gives a plain
verdict (Nothing unusual, Worth a look, Unusual), the evidence behind every check, and the question to put to the
manager. A status is a reason to ask a question, never proof of misreporting or greenwashing.

Business proposal (5 pages): [`report/FundTruth_report.pdf`](report/FundTruth_report.pdf)

## The two modules

| Module | Question | Checks |
|---|---|---|
| Integrity of returns | Are the reported returns credible? | Price regularity (smoothing), Style and benchmark (closet indexing), Return beyond the market (alpha), Small gains and losses (hidden losses), Position among peers (Isolation Forest, k-means), Shape of risk (descriptive) |
| ESG promises | Does the fund do what its own documents promise? | Promise vs portfolio (holdings vs exclusions), Promise vs behaviour (do returns still move with energy?), Label vs index (difference from the ordinary index fund and the cost of the label) |

Methods: Getmansky, Lo & Makarov (2004); Bollen & Pool (2009); Cremers & Petajisto (2009); Fama & French (1993);
Newey & West (1987); Liu, Ting & Zhou (2008). P-values are empirical, against a fixed reference set of 40 real funds.

## Run it

Python 3.11 or 3.12.

```bash
cd app
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Or double-click `app/Start FundTruth.command` (Mac) or `app/Start FundTruth (Windows).bat` (Windows): the first start
installs what is needed. Everything runs locally; uploaded files stay on your computer.

## Reproduce the numbers in the report

```bash
cd app
python -m pytest -q                 # 161 tests, including a hash test that freezes the statistical engine
python fund_report.py --all         # one-page PDF per fund + results/overview.csv
python negative_control.py          # honest simulated funds through the same pipeline
python learning_curve.py --chart    # value of more comparable funds (20 repetitions)
```

Key outputs are in `app/results/` (`overview.csv`, `negative_control.csv`, `learning_curve.csv`, `esg/`).

## Repository layout

```
app/                  Streamlit app, statistical engine (fundtruth.py), ESG module (esg_promises.py), tests, data, results
  data/               public data used by the app (see data_note.txt)
  sample_data/        examples to upload; sample_data/bbg* are synthetic, Bloomberg-style test files
report/               the business proposal: PDFs, HTML source, build.py (rebuilds the report and its charts)
```

## Data

Public sources only: Yahoo Finance (monthly total returns, Jan 2016 to Sep 2026), Kenneth R. French Data Library
(Fama-French factors), issuer holdings files and fund pages (State Street, iShares, Vanguard), and the Norges Bank
Investment Management exclusion list (via OpenSanctions). The repository contains **no Bloomberg data**: the
`sample_data/bbg*` folders are synthetic and only test the Bloomberg import format.

## Notes

- Coursework project (Data Science for Asset Management). Not investment advice.
- Thresholds are provisional; detection is validated on planted and simulated anomalies, not on labelled real cases.
- Fonts in `report/fonts` and `app/static/fonts` are under the SIL Open Font License.
