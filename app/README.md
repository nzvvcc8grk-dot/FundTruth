# FundTruth

Checks the integrity of a fund's reported monthly returns and returns a 0-100
integrity score (100 = no concern) plus a separate reliability level and a
plain-English explanation per test. **A flag is a statistical signal to
investigate, never proof of irregularity or fraud.**

## Files
| File | Purpose |
|---|---|
| `fundtruth.py` | `FundTruth` class: 5 tests, `score()`, `report()`, `plot()` (contract v2) |
| `validation.py` | `valida_dati()`: unit / date / NaN checks, raises `ValueError` on wrong units |
| `features.py` | autocorrelation, Ljung-Box, factor regression (statsmodels OLS), features |
| `synthetic.py` | 4 labelled synthetic series with known anomalies + placeholder reference |
| `validate_synthetic.py` | power, false positives, +/-20% weight sensitivity, parameter grid |
| `data.py` | yfinance + Fama-French download, cleaning, CSVs (run on your machine) |
| `score_funds.py` | score all real funds in `data/` |
| `tickers.csv` | proposed 40 US-listed funds/ETFs (all TO VERIFY on Yahoo Finance) |
| `build_report.py`, `export_report.py` | build the anonymised report notebook and export an A4 PDF; the export script FAILS unless the PDF has exactly 5 pages |
| `FundTruth_report.ipynb` | report notebook; numbers are read from `results/`, real-fund tables fill in when `results/real_funds_scores_*.csv` exist |
| `analysis.py` | real-fund diagnostics used in the report: near-zero audit (month-clustered), carry check (raw vs excess returns), weight/cap scenarios, graded-smoothing power curve |
| `fund_profile.py` | `FundProfile`, factor-set choice by region, peer choice by (asset class, region) with displayed fallback. Not part of the score |
| `context.py` | context layer: picks factors/peers from the profile, calls the frozen engine, builds versioned references (never overwrites) |
| `csv_connector.py` | Bloomberg-style CSV connector (`returns.csv` long format, `meta.csv`), explicit validation, `anonymise()` |
| `run_all.py` | **ONE COMMAND**: ingestion, universe, scoring, real validation, tables, charts, three-fund card, numeri_report, audit. Modes: Bloomberg, `--yahoo` (degraded), `--dry-run` (rehearsal) |
| `ingest.py` | strict ingestion of `bbg/` (hard stops, exclusions listed one by one, anonymised IDs, `data_quality_report.txt`) |
| `universe.py` | profiles, factors, groups (asset class x region, fallback <20), peer distributions, reference v2 (frozen, leave-one-out), per-fund findings, local snapshot |
| `real_validation.py` | flag rates per test/group/pricing/dead-vs-live (Wilson 95%), anomalies injected into real series; NO retuning |
| `three_funds.py` | comparison card (US exchange-traded ETF, European fund, appraised-NAV fund): table + chart |
| `output_audit.py` | confidentiality audit of a results folder (raw series, non-anonymous IDs, name mapping) |
| `make_sample_bbg.py`, `sample_data/bbg`, `sample_data/bbg_bad` | SYNTHETIC rehearsal universe (84 funds incl. EU, appraised NAV, dead funds, 4 to be excluded) and a copy with a deliberate percent-unit fund |
| `make_bloomberg_template.py`, `bbg_from_excel.py`, `BLOOMBERG_EXTRACTION.md` | Excel BDP/BDH template (UNTESTED on a Terminal, mnemonics are candidates), converter to the `bbg/` schema, step-by-step guide |
| `validate_closet_rule.py` | synthetic validation of the derived closet-indexing rule (`results/closet_rule_validation.csv`) |
| `connector.py` | live Yahoo Finance + French library downloads with cache (from the app version), region factors, benchmark proposals |
| `diagnostics.py`, `readings.py`, `panel.py` | diagnostics, evidence readings (3-level status, applicability, false-positive rates), panel builder (no Streamlit) |
| `app.py`, `charts.py`, `narrative.py` | Streamlit app (readings first, score in Technical details), Altair charts, plain-English text for the technical report |
| `make_sample_csv.py`, `sample_data/` | SYNTHETIC sample in the connector schema (no real data) |
| `tests/` | 161 pytest checks (engine hash and golden scores included; `test_infrastructure.py` covers ingestion stops, universe, frozen reference, end-to-end `run_all`, audit, converter) |


## Module 2: ESG promises
FundTruth now has two modules: **1. Integrity of returns** (are the reported numbers credible?) and
**2. ESG promises** (does an ESG fund do what its documents promise?). Module 2 never uses ESG ratings:
it compares each fund's own written promises with public facts.

| File | Purpose |
|---|---|
| `esg_claims.json` | claim register: verbatim promises, exclusions, fossil-fuel promise type, parent index fund, fees, sources (issuer pages, checked 2026-11-13) |
| `esg_promises.py` | three readings per fund: promise vs portfolio (holdings vs energy sector and Norges Bank exclusion list), promise vs behaviour (returns-based energy exposure, Newey-West), label vs index (R^2, tracking error, active share, fee gap in GBP per 10,000); claim cards; `CLAIM_PROMPT` + `extract_claims()` for LLM extraction |
| `app_esg.py` | Streamlit page of module 2 (sidebar: Module) |
| `public_data.py` | builds `data/` from the public downloads of `scarica_dati.sh` (Yahoo Finance, Kenneth French, issuer holdings, Norges Bank list via OpenSanctions) |
| `fund_report.py` | one-page PDF report per fund (both modules + questions for the manager); `--all` writes every report and `results/overview.csv` |
| `app_overview.py` | Streamlit overview page: every fund, both modules, sorted by the number of questions for the manager |
| `Start FundTruth.command` | macOS double-click launcher: installs Python 3.12 with uv the first time, then opens the app at http://localhost:8501 |
| `upload_help.py` | plain-language upload instructions, downloadable templates, and a converter that accepts two-column files (date + return or NAV) in CSV or Excel |
| `sample_data/esg_example_*` | example files to try 'Check your own ESG fund' (returns, holdings, promise text) |
| `tests/test_esg_promises.py` | readings behave as designed on series with a known answer |

```
python public_data.py --raw ../fundtruth_dati/raw --out data   # builds data/ (40 funds + ESG funds)
python run_all.py --yahoo                                        # module 1 results -> results/yahoo_run/
python esg_promises.py                                            # module 2 results -> results/esg/
python fund_report.py --all                                       # one PDF per fund -> results/reports/
streamlit run app.py
```
Users can also check their own ESG fund in the app (module 2 > 'Check your own ESG fund'): monthly returns, the comparison index fund, the promise text (read automatically, then confirmed by the user), fees and an optional holdings file.
ESG universe: SPYX, EFIV (vs SPY), ESGU, SUSA (vs IVV), ESGV (vs VTI), CRBN (vs ACWI); London-listed: SUSW.L (vs SWDA.L), V3AB.L (vs VWRL.L), analysed in GBP. Energy proxies XLE, IXC (IXC converted to GBP for London-listed funds).
Holdings files used: SPY, SPYX, EFIV, XLE (State Street). iShares holdings could not be downloaded automatically.

## Run
```
pip install -r requirements.txt
python -m pytest -q                      # 161 checks
python validate_synthetic.py 300         # writes results/*.csv
python data.py --tickers tickers.csv     # on your machine; or add --ff-file <zip>
python score_funds.py                    # synthetic reference
python score_funds.py --reference real   # real-fund reference, leave-one-out
```
```python
from fundtruth import FundTruth
ft = FundTruth(returns, factors, category="equity", style="active")
ft.score(); print(ft.report()); ft.plot()
```

## Frozen design (contract v2)
* Weights: autocorrelation 30, factor regression 30, near-zero 20, peer anomaly 20, Benford 0 (shown, not scored). Sum = 100, checked.
* Subscore = `100 * min(1, p / 0.10)` (p from a fixed reference set, so the score never depends on which other funds are analysed). Peer anomaly: 100 above the 5th percentile, linear below.
* Missing tests are excluded and weights renormalised; more than 1/3 of the weight missing -> no score. Reliability (high/medium/low) is a separate output.
* Final = `min(weighted average, worst subscore + 30)`.
* Factor regression flags (a) R2 >= 0.95 with tracking error in the lower tail of the category (skipped when `style="passive"`), (b) high AND significant (t >= 2) alpha in the upper tail. Negative alpha is never a concern.
* Wrong units (percent instead of decimal) raise `ValueError`; no score is produced.

## Known limits (state them in the report)
* **Provisional parameters.** The p-scale 0.10, the +30 cap, the flag level 0.05, the near-zero window, the index-likeness thresholds (R2 0.95, tracking error 2%) and the ingestion zero-run rule are assumptions, not calibrated facts. They were NOT retuned on the real data used to evaluate them.
* **AR(1) de-smoothing** (diagnostic) assumes first-order smoothing and floors the autocorrelation at zero; it is a rough estimate of unsmoothed volatility, not a correction of the data.
* **Dependence on the retrieval date.** Vendors revise NAVs and adjusted prices; scores computed on a later extract can differ (the Yahoo regression test uses a stated tolerance for this reason). Every run logs the data-received date and input checksums.
* **Closet indexing requires expenses.** Without the fund's expense ratio, and the peer median expense, closet indexing is NOT detected (0% detection in `results/closet_rule_validation.csv`). For bond, sector and thematic funds, factor R2 cannot establish index-likeness; a declared benchmark is needed.
* **Real validation has no known cases.** There are no labelled fraud or smoothing cases in real data. A flag rate on real funds is an upper bound on the false-positive rate, not a measurement; detection is shown only on simulated series and on anomalies injected into real series (which are simulations on real funds, not real cases).
* **Survivorship.** Without dead funds, flag rates are likely understated; the run states how many dead funds the universe contains.
* Groups below 20 funds use the displayed fallback; with fewer than 20 reference funds a high score is NOT reassurance. A real reference with N funds cannot give p-values below about 1/(N+1).
* Benford on returns is unsuited (about 25-37% of honest series flagged); experimental, weight 0.
* Factors are USD-based (non-USD classes carry currency effects); Global uses Developed markets as a proxy; EM is not mapped; equity factors explain little of bond returns; appraised-NAV, real-asset and "other" funds are scored through the multi-asset category as the closest available.
* Bloomberg fields and Excel formulas are untested here (no Terminal): confirm mnemonics with FLDS.

## Frozen score engine (Checkpoint A)
`fundtruth.py`, `features.py` are frozen: `tests/test_profile_context.py` checks their hashes and compares scores with
`tests/golden_*` (synthetic always; the 40 real funds when `data/` is present). Profile and context never feed the score.
Do not commit raw vendor data (licence); cite aggregated results only ("Bloomberg Terminal, retrieved <date>").
Regional factor files: `python data.py --check-ff <downloaded zip>` prints the layout summary; run it on the real file.

## One command (Bloomberg extract)
```
python run_all.py --bbg bbg --ff-us F-F_Research_Data_Factors_CSV.zip \
    --ff-europe Europe_3_Factors_CSV.zip --ff-global Developed_3_Factors_CSV.zip
```
Degraded mode on the public Yahoo universe: `python run_all.py --yahoo`. Rehearsal on the synthetic sample: `python run_all.py --dry-run`.
Exit codes: 0 ok, 2 ingestion stopped (message says why; nothing is corrected silently), 3 engine files changed, 4 audit found issues.
Everything runs locally. **Share only `results/bbg_run/`** (aggregated, anonymised, audited: `audit_report.txt`). Never share or commit
`bbg/`, `private/` (ticker-to-code mapping), `snapshot/` (cleaned series used by the app). Cite as *Bloomberg Terminal, retrieved <date>*.
App on the processed snapshot: `streamlit run app.py` (source "Universe snapshot").
Extraction instructions: `BLOOMBERG_EXTRACTION.md`. Every figure of the report is in `numeri_report.csv/json` (key, figure, unit, source file, note).

## Install and run on macOS (Terminal)
```
cd FundTruth                         # the folder with app.py
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                  # 161 checks
streamlit run app.py                 # opens in the browser
```
The app needs internet only for live data (Yahoo Finance, Kenneth French Data Library). Upload CSV works offline
with the factor files already cached. Nothing you choose changes a result without an on-screen reason: there is no
style or reference selector (index-likeness is derived; the reference is chosen automatically and shown).
