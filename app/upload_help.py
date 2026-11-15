"""Make fund uploads easy: plain-language instructions, downloadable templates, and a converter that
accepts the simplest files people actually have (two columns: date + return or NAV; CSV or Excel)
and turns them into the connector's long format (date, fund_id, value, kind). Validation stays in
csv_connector: this module only reshapes and explains."""
from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

TEMPLATE_SIMPLE = ("date,return\n2021-01-31,0.0125\n2021-02-28,-0.0083\n2021-03-31,0.0210\n"
                   "2021-04-30,0.0047\n2021-05-31,-0.0151\n")
TEMPLATE_NAV = "date,nav\n2020-12-31,100.00\n2021-01-31,101.25\n2021-02-28,100.41\n2021-03-31,102.52\n"
TEMPLATE_LONG = ("date,fund_id,value,kind\n2021-01-31,MY_FUND,0.0125,return\n2021-02-28,MY_FUND,-0.0083,return\n"
                 "2021-03-31,MY_FUND,0.0210,return\n")
_NAV_WORDS = ("nav", "price", "close", "value per share", "valore quota", "prezzo")


def show_help(key):
    """Instructions + templates, shown next to every fund uploader."""
    with st.expander("How to prepare your file (2 minutes)", expanded=False):
        st.markdown(
            "**The easiest way: two columns**, one row per month, in a CSV or Excel file.\n\n"
            "| date | return |\n|---|---|\n| 2021-01-31 | 0.0125 |\n| 2021-02-28 | -0.0083 |\n| 2021-03-31 | 0.0210 |\n\n"
            "- **date**: any day of the month is fine (it is moved to the month end).\n"
            "- **return** as a decimal: **0.0125 means +1.25%**. If your numbers are percentages (1.25), divide them by 100.\n"
            "- No returns? Use a **nav** column instead (the fund's price per unit, e.g. 101.25): we compute the returns.\n"
            "- At least **36 months** (3 years); the longer the better. Monthly data only.\n"
            "- Returns should be **after fees** and include reinvested dividends (total return), as in the fund's factsheet.\n\n"
            "Several funds in one file? Use the long format: `date, fund_id, value, kind` (kind = return or nav).")
        c = st.columns(3)
        c[0].download_button("Template: returns", TEMPLATE_SIMPLE, "fundtruth_template_returns.csv", "text/csv", key=key + "t1")
        c[1].download_button("Template: NAV", TEMPLATE_NAV, "fundtruth_template_nav.csv", "text/csv", key=key + "t2")
        c[2].download_button("Template: several funds", TEMPLATE_LONG, "fundtruth_template_several_funds.csv", "text/csv", key=key + "t3")


def _read_any(uploaded):
    name = getattr(uploaded, "name", "fund.csv")
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    else:
        text = data.decode("utf-8-sig", errors="replace")
        sep = ";" if text.count(";") > text.count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep)
        if sep == ";":                                  # European decimals: 0,0125
            for col in df.columns[1:]:
                if df[col].dtype == object:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(".", "", regex=False)
                                            .str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return name, df


def normalise_upload(uploaded):
    """Return a CSV file object in the long format the connector expects.
    Raises ValueError with a plain-language message when the file cannot be understood."""
    name, df = _read_any(uploaded)
    low = [c.lower() for c in df.columns]
    if {"date", "fund_id", "value", "kind"} <= set(low):
        out = df.rename(columns=dict(zip(df.columns, low)))
    elif len(df.columns) == 2:
        dcol, vcol = df.columns
        kind = "nav" if any(w in vcol.lower() for w in _NAV_WORDS) else "return"
        fid = os.path.splitext(os.path.basename(name))[0].strip().replace(" ", "_")[:40] or "MY_FUND"
        out = pd.DataFrame({"date": df[dcol], "fund_id": fid, "value": df[vcol], "kind": kind})
    else:
        raise ValueError("We could not read this file. Use two columns, 'date' and 'return' (or 'nav'), "
                         "one row per month - see 'How to prepare your file' for a template.")
    try:
        out["date"] = pd.to_datetime(out["date"], errors="raise", dayfirst=False)
    except (ValueError, TypeError):
        raise ValueError("Some dates could not be read. Write them as 2021-01-31 (year-month-day).") from None
    v = pd.to_numeric(out["value"], errors="coerce")
    if v.isna().all():
        raise ValueError("The value column has no numbers. Check that it contains returns or NAVs.")
    r = out["kind"].astype(str).str.lower() == "return"
    if r.any() and v[r].abs().median() > 0.2:
        raise ValueError("Your returns look like percentages (for example 1.25 for +1.25%). "
                         "Divide them by 100 (0.0125) and upload again.")
    out["value"] = v
    buf = io.StringIO()
    out.to_csv(buf, index=False, date_format="%Y-%m-%d")
    buf.seek(0)
    return buf


# =====================================================================================================
# Easier input: paste from Excel or a factsheet, upload any simple file, or try an example.
# The parser accepts what people actually have and says how it read it; the user can correct the
# reading. Everything ends in the long format above, so csv_connector still does the validation.
# =====================================================================================================
import re

import numpy as np

from csv_connector import load_csv_universe

_MONTHS = {"jan": 1, "gen": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "mag": 5, "jun": 6, "giu": 6, "jul": 7,
           "lug": 7, "aug": 8, "ago": 8, "sep": 9, "set": 9, "oct": 10, "ott": 10, "nov": 11, "dec": 12, "dic": 12}
UNITS = {"pct": "Monthly returns in % (1.25 means +1.25%)",
         "dec": "Monthly returns as decimals (0.0125 means +1.25%)",
         "nav": "Prices per unit (NAV), not returns"}
EXAMPLES = {
    "Property fund valued by appraisal (simulated)": ("returns_sample.csv", "SYN_APPRAISED"),
    "US equity ETF (simulated)": ("returns_sample.csv", "SYN_US_ETF"),
    "Bond fund (simulated)": ("returns_sample.csv", "SYN_BOND"),
}
ESG_EXAMPLES = {"Fossil-free S&P 500 fund (real prices, SPYX)": ("esg_example_returns.csv", None)}
_SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")


def _num(cell):
    """'1.25%', '-0,8', '(0.4)', '1,234.5', '−2.1' -> (float or nan, had a % sign)."""
    s = str(cell).strip().replace("−", "-").replace("–", "-").replace("\xa0", "").replace(" ", "")
    if s in ("", "-", "--", "n/a", "na", "nan", "None"):
        return np.nan, False
    pct = s.endswith("%")
    s = s.rstrip("%")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s), pct
    except ValueError:
        return np.nan, pct


def _rows(text):
    """Split pasted text into rows of cells: tab (Excel), semicolon, comma, or spaces (PDF)."""
    lines = [l for l in str(text).replace("\r", "").split("\n") if l.strip()]
    for sep in ("\t", ";"):
        if any(sep in l for l in lines):
            return [[c.strip() for c in l.split(sep)] for l in lines]
    if any("," in l for l in lines):
        decimal_comma = sum(bool(re.search(r"\s[-(]?\d+,\d+\)?%?(\s|$)", l)) for l in lines) > len(lines) / 2
        if not decimal_comma:
            return [[c.strip() for c in l.split(",")] for l in lines]
    return [re.split(r"\s+", l.strip()) for l in lines]


def _month_end(y, m):
    return pd.Timestamp(year=int(y), month=int(m), day=1) + pd.offsets.MonthEnd(0)


def _parse_grid(rows):
    """Factsheet grid: a header row with month names, then one row per year. None if not a grid."""
    for h, row in enumerate(rows):
        cols = {i: _MONTHS[c.strip().lower()[:3]] for i, c in enumerate(row) if c.strip().lower()[:3] in _MONTHS}
        if len(cols) >= 10:
            out, pct = [], False
            for r in rows[h + 1:]:
                if not r or not re.fullmatch(r"(19|20)\d\d", r[0].strip()):
                    continue
                shift = 1 if min(cols) == 0 else 0      # header without a "Year" cell: data rows start with it
                for i, m in cols.items():
                    j = i + shift
                    if j < len(r):
                        v, p = _num(r[j])
                        pct |= p
                        if np.isfinite(v):
                            out.append((_month_end(r[0], m), v))
            if out:
                return pd.DataFrame(out, columns=["date", "value"]), pct, "a table of monthly returns by year"
    return None


def _dates(cells):
    """Parse a column of dates; day-first when the first number is above 12 (31/01/2021)."""
    s = pd.Series([str(c).strip() for c in cells])
    parts = s.str.extract(r"^(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{2,4})")
    dayfirst = False
    if parts[0].notna().any() and not parts[0].dropna().str.len().eq(4).all():
        dayfirst = bool((parts[0].dropna().astype(int) > 12).any())
    d = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    return d


def guess_units(values, pct_sign=False, header=""):
    """'nav', 'pct' or 'dec' from the numbers themselves."""
    v = pd.Series(values, dtype=float).dropna()
    if any(w in header.lower() for w in _NAV_WORDS):
        return "nav"
    if pct_sign:
        return "pct"
    if len(v) and (v > 0).all() and v.median() > 5:
        return "nav"
    if len(v) and v.abs().median() > 0.2:
        return "pct"
    return "dec"


def parse_history(rows):
    """Rows of cells -> (DataFrame date,value | long-format DataFrame, units guess, description).
    Raises ValueError with a plain-language message."""
    if not rows:
        raise ValueError("Nothing to read yet: paste the figures or choose a file.")
    low = [c.strip().lower() for c in rows[0]]
    if {"date", "fund_id", "value", "kind"} <= set(low):                      # several funds, long format
        df = pd.DataFrame(rows[1:], columns=low)[["date", "fund_id", "value", "kind"]]
        return df, "long", "the several-funds format"
    g = _parse_grid(rows)
    if g is not None:
        df, pct, what = g
        return df.sort_values("date"), guess_units(df["value"], pct), what
    # two columns: date + number (a header row is optional)
    header = ""
    body = rows
    if _dates([rows[0][0]]).isna().all():
        header, body = " ".join(rows[0]), rows[1:]
    dcells, vals, pct = [], [], False
    for r in body:
        cells = [c for c in r if c.strip()]
        if len(cells) < 2:
            continue
        # the date may take several cells when separated by spaces ("31 Jan 2021 1.25")
        for k in (1, 2, 3):
            if len(cells) > k and _dates([" ".join(cells[:k])]).notna().all():
                v, p = _num(cells[k])
                dcells.append(" ".join(cells[:k])); vals.append(v); pct |= p
                break
    if not dcells:
        raise ValueError("We could not find dates and numbers. Paste two columns (date and value), or a "
                         "factsheet table with the months across the top and one row per year.")
    d = _dates(dcells)
    if d.isna().mean() > 0.2:
        raise ValueError("Some dates could not be read. Write them as 2021-01-31 or 31/01/2021.")
    df = pd.DataFrame({"date": d, "value": vals}).dropna()
    return df.sort_values("date"), guess_units(df["value"], pct, header), "two columns: date and value"


def rows_from_upload(uploaded):
    name = getattr(uploaded, "name", "fund.csv")
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    if name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        rows = [[("" if pd.isna(c) else str(c)) for c in r] for r in df.itertuples(index=False)]
        rows = [[c[:10] if re.match(r"^\d{4}-\d{2}-\d{2} 00:00:00$", c) else c for c in r] for r in rows]
        return name, [r for r in rows if any(c.strip() for c in r)]
    return name, _rows(data.decode("utf-8-sig", errors="replace"))


def to_long(df, units, fund_id):
    """Parsed table + the units the user confirmed -> CSV buffer in the connector's long format."""
    if units == "long":
        out = df.copy()
    else:
        v = pd.to_numeric(df["value"], errors="coerce") * (0.01 if units == "pct" else 1.0)
        out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"), "fund_id": fund_id,
                            "value": v, "kind": "nav" if units == "nav" else "return"})
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def where_to_find():
    with st.expander("Where to find this data", icon=":material/travel_explore:"):
        st.markdown(
            "- **A listed fund or ETF?** Use *Look up a ticker online*: there is nothing to prepare.\n"
            "- **The fund's factsheet (PDF).** The performance section often has a table of monthly returns, "
            "one row per year. Select the whole table, copy it and paste it here: FundTruth reads it as it is.\n"
            "- **The fund's website.** *Prices* or *Performance* pages usually let you download the price (NAV) "
            "history. Upload that file unchanged; daily prices are fine, the last price of each month is used.\n"
            "- **A fund you hold privately** (property, private credit, hedge fund). Ask the manager or the "
            "administrator for the monthly NAV history since launch. Investors are normally entitled to it.\n"
            "- **Your platform, Morningstar or FT.com.** Their historical prices pages can be exported or copied.\n\n"
            "At least 36 months are needed; returns after fees, with dividends reinvested. "
            "Your data stays on this computer.")
        c = st.columns(3)
        c[0].download_button("Template: returns", TEMPLATE_SIMPLE, "fundtruth_template_returns.csv", "text/csv",
                             key="wtf_t1_" + str(id(c)))
        c[1].download_button("Template: prices (NAV)", TEMPLATE_NAV, "fundtruth_template_nav.csv", "text/csv",
                             key="wtf_t2_" + str(id(c)))
        c[2].download_button("Template: several funds", TEMPLATE_LONG, "fundtruth_template_several_funds.csv",
                             "text/csv", key="wtf_t3_" + str(id(c)))


PASTE_HINT = ("Paste two columns from Excel, for example\n2021-01-31    1.25%\n2021-02-28   -0.83%\n\n"
              "or a factsheet table:\nYear  Jan   Feb   Mar  ...  Dec\n2021  1.2  -0.8   2.1  ...  0.5")


def returns_input(key, examples=None):
    """Three ways in. Returns (fund_id, monthly return Series, prefill dict) or (None, None, {})."""
    examples = EXAMPLES if examples is None else examples
    where_to_find()
    ways = ["Paste from Excel or a factsheet", "Upload a file", "Try an example"]
    how = st.segmented_control("How would you like to add the data?", ways, default=ways[0], key=key + "_how")
    how = how or ways[0]
    rows, name, prefill = None, None, {}
    if how == ways[0]:
        text = st.text_area("Monthly figures", height=170, key=key + "_paste", placeholder=PASTE_HINT,
                            help="Copy cells from Excel or a table from a PDF factsheet and paste them here.")
        name = st.text_input("Fund name", "My fund", key=key + "_name", help="Used in the results and the report.")
        if text.strip():
            rows = _rows(text)
    elif how == ways[1]:
        f = st.file_uploader("CSV or Excel file with the fund's monthly returns or prices",
                             type=["csv", "xlsx", "xls", "txt"], key=key + "_file")
        if f is not None:
            name, rows = rows_from_upload(f)
            name = os.path.splitext(os.path.basename(name))[0]
    else:
        pick = st.selectbox("Example", list(examples), key=key + "_ex")
        fname, fid = examples[pick]
        df = pd.read_csv(os.path.join(_SAMPLE, fname))
        if fid is not None:
            df = df[df["fund_id"] == fid][["date", "value"]]
            meta = os.path.join(_SAMPLE, "meta_sample.csv")
            if os.path.exists(meta):
                m = pd.read_csv(meta).set_index("fund_id")
                if fid in m.index:
                    prefill = {k: v for k, v in m.loc[fid].items() if isinstance(v, str) or pd.notna(v)}
        else:
            df.columns = ["date", "value"]
            prom = os.path.join(_SAMPLE, "esg_example_promise.txt")
            if os.path.exists(prom):
                prefill["promise"] = open(prom).read()
        rows = [["date", "value"]] + df.astype(str).values.tolist()
        short = {"SYN_APPRAISED": "Example property fund", "SYN_US_ETF": "Example equity ETF", "SYN_BOND": "Example bond fund"}
        name = short.get(fid, fid) if fid else pick.split("(")[-1].rstrip(")").split(",")[-1].strip()
    if not rows:
        st.info("Add the fund's monthly figures to continue. Not sure where to find them? "
                "Open *Where to find this data* above.", icon=":material/info:")
        return None, None, {}
    try:
        table, guess, what = parse_history(rows)
    except ValueError as e:
        st.error(str(e), icon=":material/error:")
        return None, None, {}
    fid = re.sub(r"[^A-Za-z0-9_.-]+", "_", (name or "MY_FUND").strip())[:40] or "MY_FUND"
    if guess != "long":
        ukey = key + "_units"
        sig = f"{len(table)}|{guess}|{how}|{fid}"
        if st.session_state.get(ukey + "_sig") != sig:      # new data: show our reading again
            st.session_state[ukey + "_sig"] = sig
            st.session_state[ukey] = guess
        units = st.radio("We read these numbers as", list(UNITS), format_func=UNITS.get, key=ukey,
                         horizontal=True, help="Change this if our reading is wrong.")
    else:
        units = "long"
    try:
        uni = load_csv_universe(to_long(table, units, fid), None)
    except ValueError as e:
        st.error(str(e), icon=":material/error:")
        return None, None, {}
    for _, row in uni.rejected.iterrows():
        st.error(f"{row.fund_id} cannot be analysed: {row.reason}", icon=":material/error:")
    if not uni.returns:
        return None, None, {}
    if len(uni.returns) > 1:
        fid = st.selectbox("Fund in the file", list(uni.returns), key=key + "_pick")
    else:
        fid = list(uni.returns)[0]
    r = uni.returns[fid]
    with st.container(border=True):
        st.markdown(f"**{len(r)} months read**, {r.index[0]:%b %Y} to {r.index[-1]:%b %Y}, from {what}. "
                    "Check the line below looks like your fund.")
        import altair as alt
        g = ((1 + r.fillna(0)).cumprod() * 100).rename("value").rename_axis("month").reset_index()
        st.altair_chart(alt.Chart(g, height=150).mark_line(strokeWidth=2, color="#1E4D78").encode(
            x=alt.X("month:T", title=None), y=alt.Y("value:Q", title="100 invested", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("value:Q", format=".1f")]), use_container_width=True)
        for w in uni.warnings.get(fid, []):
            st.caption(w)
    return fid, r, prefill
