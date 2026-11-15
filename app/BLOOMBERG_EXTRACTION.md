# Bloomberg extraction guide (macOS)

**Status: nothing in this guide has been tested against a Bloomberg Terminal** (none was available
where it was written). Every field mnemonic is a *candidate*: confirm each with `FLDS` before use. The
Excel formulas use standard Bloomberg add-in functions (BDP, BDH) that read the mnemonics from the
`README` sheet of the workbook, so a corrected mnemonic is changed in one cell.
Whether the Terminal licence allows these exports for coursework, and whether the Excel add-in works
on a Mac, must be checked with the university library or Bloomberg support (**to verify**). If the
add-in is not available on the Mac, use a Windows Bloomberg machine, or export the series from the
Terminal (`FLDS`/`HP`/`EXPORT`-style screens) to CSV and place them in the same schema.

Raw data never leave your computer: `bbg/`, `private/`, `snapshot/` and every `.xlsx` are git-ignored.
Share only the audited folder `results/bbg_run/`. Cite results as *Bloomberg Terminal, retrieved <date>*.

## 1. Which funds (aim for 25 per group, so that 20 survive the data checks)

The peer tests need at least 20 funds per group (asset class x region). Pick, ideally with Terminal fund
search (`FSRCH`; screening filters and the option to include inactive/dead funds are **to confirm**):

| Group | Target | Notes |
|---|---|---|
| equity x US | 25 | mix of ETFs (pricing `exchange_traded`) and mutual funds (`nav_listed_holdings`); include a few index funds |
| equity x Europe | 25 | UCITS equity funds; keep the share-class currency (EUR/GBP/...) as reported |
| bond x US | 25 | |
| bond x Europe | 25 (optional) | |
| appraised NAV (property, private credit, fund of hedge funds) | as many as the Terminal returns, ideally 20+ | pricing `appraised_nav`; if fewer than 20 the fallback rule is shown and read with care |

Rules that keep the study honest: take every fund the screen returns, not the ones that look good; include
dead/merged funds if the screen allows (otherwise survivorship bias is declared); at least 36 monthly
observations each; do not choose funds you already know are problematic. Fewer funds in a group is fine:
the tool displays the fallback; more is better.

## 2. Discover the fields with FLDS

On the Terminal type `FLDS <GO>`, search each concept below, open the field, check it returns data for one
fund of each kind (ETF, UCITS fund, property fund), and write the mnemonic in the `README` sheet (yellow cells).

| Need | Candidate mnemonic (to confirm with FLDS) |
|---|---|
| NAV history | `FUND_NET_ASSET_VAL` |
| Total-return index history (if NAV ignores distributions) | `TOT_RETURN_INDEX_GROSS_DVDS` |
| Currency | `CRNCY` |
| Expense ratio (reported in percent) | `FUND_EXPENSE_RATIO` |
| Inception | `FUND_INCEPT_DT` |
| Category / style label | `FUND_ASSET_CLASS_FOCUS` |
| Geographic focus | `FUND_GEO_FOCUS` |
| Domicile | `FUND_DOMICILE` |
| Fund type | `FUND_TYP` |
| Active / dead status | `FUND_ACTIVE_STATUS` (existence unknown: otherwise mark dead funds by hand in column F) |

Use **total return** (distributions reinvested) when available: a plain NAV drops on distributions and
would look like a loss. State in `README!B6` which sheet (`NAV` or `TRI`) is to be converted; if you use
the total-return sheet, run the converter with `--kind nav` too (it is an index level, converted to returns).

## 3. Fill the workbook

1. `python make_bloomberg_template.py` creates `bloomberg_template.xlsx` (150 fund slots).
2. `README`: type the retrieval date (B3), check start/end dates, and the mnemonics.
3. `Universe`: column A tickers exactly as the Terminal writes them (e.g. `XXXX US Equity`); B region
   (US / Europe / Global / EM / Other); C asset_class (equity / bond / multi-asset / real_assets / other);
   D pricing (exchange_traded / nav_listed_holdings / appraised_nav); E benchmark ticker (an index or ETF
   ticker, optional but needed for the closet rule on non-equity funds); F `1` for dead/merged funds.
   Columns G-N fill by BDP.
4. The sheets `NAV`, `TRI`, `Benchmarks` fill with BDH (monthly, calendar days, no gap filling: missing months stay visible).
5. Wait for the formulas to finish, then **select all, copy, paste special as VALUES**, save. (The converter reads saved values.)

## 4. Convert to the `bbg/` schema and check

```
python bbg_from_excel.py bloomberg_filled.xlsx --out bbg/
python ingest.py bbg/ --out results/bbg_check
```
The converter assigns anonymised codes F001... (mapping in `private/id_map.csv`, local only), converts the
expense ratio from percent to decimal, moves a date to month end only if it is in the same month (it prints
how many), and drops `#N/A`. `ingest.py` then **stops with a clear message** on missing columns, duplicates,
non-monthly dates, percent instead of decimals, empty currency, and lists funds it excludes (under 36 months,
gaps, repeated zero returns) one by one in `data_quality_report.txt`.

## 5. Factors (Kenneth French library, downloaded by you from the library site)

Download `F-F_Research_Data_Factors_CSV.zip`, `Europe_3_Factors_CSV.zip`, `Developed_3_Factors_CSV.zip`
and pass them to run_all (command in the README). The loader checks the layout and stops if it changed.

## 6. Known limits (read before you extract)

- A field may be empty for some funds (typically expense ratio, benchmark, inception for private or
  appraised funds). The data check lists them; **without the expense ratio closet indexing is not detected**,
  and without a declared benchmark the index-likeness of bond/sector funds cannot be established.
- Appraised-NAV and real-asset funds are scored through the multi-asset category as the closest available
  (visible note); their peer sets are usually small.
- Factors are USD-based; non-USD share classes carry currency effects (warning shown).
- Bloomberg "dead fund" coverage depends on the screen; if none are included, survivorship bias is declared.
- Values depend on the retrieval date (vendors revise NAVs): rerunning on a later extract can change figures.
