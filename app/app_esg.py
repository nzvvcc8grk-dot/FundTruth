"""Streamlit page of module 2 (ESG promises): ESG promises vs holdings, behaviour and cost.
Two ways in: a built-in ESG fund, or the user's own ESG fund (returns + promises + optional holdings)."""
import os
import tempfile

import pandas as pd
import streamlit as st

import esg_promises as ml
import upload_help as uh
from csv_connector import load_csv_universe
from readings import NA

# Comparison ("parent") funds available in the built-in data: label, fee, currency, energy proxy
PARENTS = {"SPY": ("S&P 500 - US large companies (SPDR, USD)", 0.000945, "USD", "XLE"),
           "IVV": ("S&P 500 - US large companies (iShares, USD)", 0.0003, "USD", "XLE"),
           "VTI": ("US total stock market (Vanguard, USD)", 0.0003, "USD", "XLE"),
           "ACWI": ("Global equities, all countries (iShares, USD)", 0.0032, "USD", "IXC"),
           "SWDA.L": ("MSCI World, developed markets (iShares, London, GBP)", 0.0020, "GBP", "IXC_GBP"),
           "VWRL.L": ("FTSE All-World, global equities (Vanguard, London, GBP)", 0.0019, "GBP", "IXC_GBP")}


@st.cache_data(show_spinner=False)
def _returns(path):
    return pd.read_csv(path, index_col="date", parse_dates=True)


@st.cache_data(show_spinner=False)
def _exclusions(path):
    return ml.load_exclusions(path)


def render_card(card, R, level_color, pill, source_note, pdf=True):
    t, c = card.ticker, card.claim
    st.markdown(f"### {t}: {card.headline}")
    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="ft-label">What the fund promises (verbatim)</div>', unsafe_allow_html=True)
        for q in c["quotes"] or ["(no quote given)"]:
            st.markdown(f"> {q}")
        st.caption(source_note)
    with right:
        st.markdown('<div class="ft-label">Facts</div>', unsafe_allow_html=True)
        st.write(f"Index: {c.get('index') or 'not stated'}")
        st.write(f"Compared with: {c['parent']} - {c['parent_name']}")
        st.write(f"Fees: {c['expense_ratio']*100:.4g}% vs {c['parent_expense_ratio']*100:.4g}%")
        st.write(f"Fossil-fuel promise: {ml.PROMISE_LABEL[c['fossil_promise']]}")
    both = pd.concat([R[t].dropna(), R[c["parent"]].dropna()], axis=1, join="inner").dropna()
    growth = (1 + both).cumprod() * 10000
    growth.columns = [f"{t} (ESG)", f"{c['parent']} (parent)"]
    st.markdown('<div class="ft-label">Growth of 10,000 (fund currency), after fees</div>', unsafe_allow_html=True)
    st.line_chart(growth, height=260)
    for x in card.readings:
        with st.container(border=True):
            tag = f" &nbsp; {pill(x.tag, '#1f4e79')}" if x.tag else ""
            st.markdown(f"**{x.title}** &nbsp; {pill(x.level, level_color[x.level])}{tag}", unsafe_allow_html=True)
            if x.level == NA:
                st.write("Not applicable: " + x.reason)
                continue
            for lab, v in (("What we measured", x.measured), ("What we found", x.found), ("How sure we are", x.uncertainty),
                           ("Why it may be innocent", x.innocent), ("What to ask the manager", x.ask)):
                if v:
                    st.markdown(f'<div class="ft-label">{lab}</div>', unsafe_allow_html=True)
                    st.write(v)
            if x.reason:
                st.caption("Why this reading is " + x.tag + ": " + x.reason)
            with st.expander("Limits of this check"):
                st.write("Not validated: " + x.not_validated)
    if card.questions:
        st.markdown("**Questions for the manager**")
        for i, q in enumerate(card.questions, 1):
            st.write(f"{i}. {q}")
    c1, c2 = st.columns(2)
    c1.download_button("Download claim card (Markdown)", ml.card_markdown(card),
                       f"fundtruth_esg_{t.replace('.', '_')}.md", "text/markdown", key="md_" + t)
    if pdf:
        with c2:
            import app_overview
            app_overview.download_button(t, "esg_pdf_" + t)


def builtin(R, claims, hdir, excl, level_color, pill):
    funds = [t for t, v in claims["funds"].items() if t in R and v["parent"] in R]   # funds with data only
    t = st.selectbox("ESG fund", funds, format_func=lambda x: f"{x} - {claims['funds'][x]['name']}")
    try:
        card = ml.analyse_fund(t, R, claims, hdir, excl)
    except ValueError as e:
        st.error(f"Not covered: {e}")
        return
    render_card(card, R, level_color, pill,
                f"Source: {card.claim['source']} (checked {claims['as_of']}). Extracted with an LLM.")
    with st.expander("All ESG funds at a glance"):
        rows = []
        for u in funds:
            try:
                k = ml.analyse_fund(u, R, claims, hdir, excl)
            except ValueError:
                continue
            m = k.metrics
            rows.append({"Fund": u, "Compared with": k.claim["parent"], "Headline": k.headline,
                         **{r.title.split(":")[0]: r.level for r in k.readings},
                         "R² with parent": round(m["r2"], 3), "Extra fee per 10,000": round(m["fee_gap_gbp_per_10k"]),
                         "Energy underweight implied by returns": f"{m.get('fossil_implied_underweight', float('nan')):.1%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)


def own_fund(R, hdir, excl, level_color, pill):
    st.markdown("Check **your own ESG fund** in four steps. Your files stay on this computer.")
    # 1. returns
    st.markdown("#### 1. The fund's monthly history")
    fid, series, prefill = uh.returns_input("esgret", examples=uh.ESG_EXAMPLES)
    if prefill.get("promise") and not st.session_state.get("esg_txt"):
        st.session_state["esg_txt"] = prefill["promise"]          # the example comes with its promise
    # 2. comparison fund
    st.markdown("#### 2. The non-ESG fund to compare with")
    opts = [p for p in PARENTS if p in R]
    parent = st.selectbox("Closest ordinary index fund (same market, no ESG screens)", opts,
                          format_func=lambda p: f"{p} - {PARENTS[p][0]}",
                          help="The ESG fund is compared with this fund. Pick the one tracking the same market; "
                               "its currency must be the currency of your returns.")
    # 3. promises
    st.markdown("#### 3. What the fund promises")
    txt = st.text_area("Paste the fund's ESG description from its KID, factsheet or website",
                       height=120, key="esg_txt",
                       placeholder="e.g. The index excludes companies involved in thermal coal, tobacco and controversial weapons...")
    pre = ml.keyword_claims(txt) if txt.strip() else dict(exclusions=[], fossil_promise="none", quotes=[])
    if st.session_state.get("esg_txt_last") != txt:          # new text: refresh the pre-filled fields
        st.session_state["esg_txt_last"] = txt
        st.session_state["esg_ex"] = pre["exclusions"]
        st.session_state["esg_pr"] = pre["fossil_promise"]
        st.session_state["esg_q"] = "\n".join(pre["quotes"])
    if txt.strip():
        st.caption("We read the text and pre-filled the promises below. Check them and correct them if needed.")
    excls = st.multiselect("Activities the fund says it excludes", list(ml.EXCLUSION_TERMS), key="esg_ex")
    promise = st.radio("Fossil-fuel promise", list(ml.PROMISE_LABEL), format_func=lambda k: ml.PROMISE_LABEL[k], key="esg_pr")
    quotes = [q for q in st.text_area("Sentences that state the promise (one per line, verbatim)", height=90,
                                      key="esg_q").splitlines() if q.strip()]
    # 4. fees and optional holdings
    st.markdown("#### 4. Fees and, optionally, holdings")
    c = st.columns(2)
    er = c[0].number_input("Ongoing charge of the ESG fund, % a year", 0.0, 3.0, 0.20, 0.01, key="esg_er") / 100
    per = c[1].number_input(f"Ongoing charge of {parent}, % a year", 0.0, 3.0, round(PARENTS[parent][1] * 100, 4), 0.01, key="esg_per") / 100
    fh = st.file_uploader("Holdings file (optional: columns name and weight; ticker and sector if you have them)",
                          type=["csv", "xlsx"], key="esg_h")
    if not st.button("Check the promises", type="primary", disabled=series is None, icon=":material/fact_check:"):
        if series is None:
            st.info("Upload the fund's returns to start.")
        elif st.session_state.get("esg_result") and st.session_state["esg_result"][0] == fid:
            import fund_card as fc
            f0, c0, R0 = st.session_state["esg_result"]
            fc.render(f0, None, c0, R=R0, name=f0.replace("_", " "), pdf=False, source_note="Text pasted by the user and checked on screen.",
                      key="own_esg")
        return
    override = None
    if fh is not None:
        from public_data import read_holdings
        suffix = ".xlsx" if fh.name.lower().endswith(".xlsx") else ".csv"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(fh.getvalue())
        try:
            override = {fid: read_holdings(tmp.name)}
        except ValueError:
            st.warning("The holdings file could not be read (it needs a 'name' or 'ticker' column and a 'weight' "
                       "column): the portfolio check is skipped.")
        finally:
            os.unlink(tmp.name)
    claim = dict(name=fid, index="", parent=parent, parent_name=PARENTS[parent][0], energy_proxy=PARENTS[parent][3],
                 expense_ratio=er, parent_expense_ratio=per, fossil_promise=promise,
                 exclusions=excls or ["none stated"], quotes=quotes, source="text pasted by the user")
    Rx = R.copy()
    Rx[fid] = series.reindex(Rx.index.union(series.index))
    try:
        card = ml.analyse_fund(fid, Rx, {"funds": {fid: claim}}, hdir, excl, holdings_override=override)
    except ValueError as e:
        st.error(f"Not covered: {e}. Check that the dates overlap with {parent} (data until "
                 f"{R[parent].dropna().index[-1]:%b %Y}) and that you have at least {ml.MIN_MONTHS} common months.")
        return
    import fund_card as fc
    st.session_state["esg_result"] = (fid, card, Rx)
    fc.render(fid, None, card, R=Rx, name=fid.replace("_", " "), pdf=False, source_note="Text pasted by the user and checked on screen.",
              key="own_esg")


def page(here, level_color, pill):
    st.subheader("ESG promises")
    st.caption("Does an ESG fund do what its documents promise? The fund's own written promises are compared with "
               "its holdings, its returns and its cost. No ESG ratings are used. A status is a reason to ask the "
               "manager a question, never proof of greenwashing.")
    data_dir = os.path.join(here, "data")
    rpath = os.path.join(data_dir, "esg_returns.csv")
    if not os.path.exists(rpath):
        st.error("data/esg_returns.csv is missing: run `python public_data.py` after downloading the public data.")
        return
    R = _returns(rpath)
    claims = ml.load_claims()
    excl = _exclusions(os.path.join(data_dir, "nbim_exclusions.csv"))
    hdir = os.path.join(data_dir, "holdings")
    how = st.radio("Fund", ["Choose a built-in ESG fund", "Check your own ESG fund"], horizontal=True, key="esg_how")
    if how.startswith("Choose"):
        builtin(R, claims, hdir, excl, level_color, pill)
    else:
        own_fund(R, hdir, excl, level_color, pill)
