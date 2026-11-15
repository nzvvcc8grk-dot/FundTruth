"""Check your own fund: upload returns (module 1), an ESG fund (module 2), or a ticker looked up online."""
import os

import streamlit as st

import app_esg
import connector as cn
import fund_card as fc
import panel as pn
import ui
import upload_help as uh
from fund_profile import UNKNOWN, factor_set_for

ui.page_header("Check your own fund",
               "Run the same checks on a fund that is not in the watchlist. Your files stay on this computer.")
ctx = ui.context()


def contribute_box(r, prof, key):
    """Continuous improvement: the fund just checked can join the comparison set used for the next checks."""
    counts, added = ui.comparison_counts()
    with st.container(border=True):
        st.markdown("**Make the next check sharper.** Add this fund to FundTruth's comparison set. The checks compare "
                    "each fund with similar ones, and today's groups are small ("
                    + ", ".join(f"{dict(bond='bond', equity='equity').get(k, 'mixed')} {v}" for k, v in sorted(counts.items()))
                    + " funds). "
                    "With about 40 per group the checks reach their full power (see *How it works*). Only summary "
                    "statistics are kept, never the returns or the fund's name.")
        if st.button("Add to the comparison set", key="contrib_" + key, icon=":material/group_add:"):
            ok, msg = ui.contribute(r, prof)
            (st.success if ok else st.info)(msg)
UV = ctx["UV"]
t1, t2, t3 = st.tabs(["Add a fund's monthly returns", "Check an ESG fund's promises", "Look up a ticker online"])

# ---------------------------------------------------------------- 1. upload
with t1:
    st.markdown("#### 1. Add the fund's monthly history")
    fid, r, prefill = uh.returns_input("own")
    if r is not None:
        st.markdown("#### 2. Tell us about the fund")
        st.caption("Three quick questions decide which market factors and which similar funds the fund is compared with. "
                   "Not sure? Leave 'Not sure': the checks that need the answer are skipped and say so.")
        prof, bench, start = fc.profile_form(fid, prefill, "up_" + fid)
        fs = factor_set_for(prof)
        if prof.region != UNKNOWN and (not fs.usable or fs.key not in UV.factors):
            st.warning(f"No market factors for region {prof.region} in the built-in data: the factor-based checks are skipped.")
        if st.button("Run the checks", type="primary", key="run_up", icon=":material/play_arrow:"):
            ref, ref_label = ui.reference_for_checks()
            st.session_state["up_result"] = (fid, pn.analyse_panel(r, prof, UV.factors, ref, ref_label,
                                                                   None, "", UV.peer_stats, None))
        res = st.session_state.get("up_result")
        if res and res[0] == fid:
            st.divider()
            fc.render(fid, res[1], None, name=fid.replace("_", " "), pdf=False, key="up")
            contribute_box(r, prof, "up_" + fid)

# ---------------------------------------------------------------- 2. ESG
with t2:
    R = ctx["R"]
    if R is None:
        st.error("The ESG comparison data is missing: run `python public_data.py` first.")
    else:
        app_esg.own_fund(R, ctx["hdir"], ctx["excl"], None, None)

# ---------------------------------------------------------------- 3. ticker
with t3:
    st.markdown("Download a fund's prices from Yahoo Finance and the market factors from the Kenneth French Data Library. "
                "Only the ticker leaves this computer.")
    ticker = st.text_input("Ticker", "", placeholder="For example VTI or VWRL.L", key="live_t").strip().upper()
    known = cn.lookup(ticker) if ticker else None
    legacy = {}
    if known:
        from fund_profile import propose_pricing
        legacy = dict(region="US", currency="USD", asset_class=known[0],
                      pricing=propose_pricing("ETF" if ticker in ("SPY", "QQQ", "VTI", "IWM") else None) or "")
    if ticker:
        prof, bench, start = fc.profile_form(ticker, legacy, "live")
        if st.button("Download and run the checks", type="primary", key="run_live", icon=":material/cloud_download:"):
            try:
                with st.spinner("Downloading data and running the checks..."):
                    r = cn.fetch_series(ticker, start)
                    fs = factor_set_for(prof)
                    facs = {fs.key: cn.get_region_factors(fs.key)} if fs.usable else {}
                    b = cn.fetch_series(bench, start) if bench else None
                    ref, ref_label = ui.reference_for_checks()
                    st.session_state["live_result"] = (ticker, pn.analyse_panel(r, prof, facs, ref, ref_label,
                                                                                b, bench, UV.peer_stats, None), r)
            except cn.ConnectorError as e:
                st.error(f"{e} Check the ticker and your internet connection.")
        res = st.session_state.get("live_result")
        if res and res[0] == ticker:
            st.divider()
            fc.render(ticker, res[1], None, name=ticker, pdf=False, key="live")
            contribute_box(res[2], prof, "live_" + ticker)
    else:
        st.info("Type a ticker to start. The fund's facts are pre-filled when the ticker is known.")

ui.footer()
