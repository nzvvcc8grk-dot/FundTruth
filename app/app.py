"""FundTruth app.

    streamlit run app.py

An opening screen is shown once per visit. Pages: Home, Watchlist, Fund card, Check your own fund, How it works. The analysis is unchanged
(engine, readings, ESG module); this file only sets the identity and the navigation. The original
single-page screens now live in fund_card.py (fund card, charts, facts form) and views/check.py
(upload, ESG fund, ticker). A flag is a statistical signal to investigate, never proof.
"""
import os
import warnings

import streamlit as st

import ui

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="FundTruth", page_icon=os.path.join(HERE, "static", "mark.svg"), layout="wide",
                   initial_sidebar_state="auto")
ui.inject_css()
st.logo(os.path.join(HERE, "static", "logo.svg"), size="large", icon_image=os.path.join(HERE, "static", "mark.svg"))

P = {
    "home": st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
    "watchlist": st.Page("views/watchlist.py", title="Watchlist", icon=":material/list_alt:", url_path="watchlist"),
    "fund": st.Page("views/fund.py", title="Fund card", icon=":material/query_stats:", url_path="fund"),
    "check": st.Page("views/check.py", title="Check your own fund", icon=":material/upload_file:", url_path="check"),
    "method": st.Page("views/method.py", title="How it works", icon=":material/menu_book:", url_path="method"),
}
ui.PAGES.update(P)
nav = st.navigation({"": [P["home"]], "Screen": [P["watchlist"], P["fund"]], "Your funds": [P["check"]],
                     "About": [P["method"]]})
if nav is P["home"] and not st.session_state.get("entered") and not st.query_params:
    import splash                                  # opening screen, once per visit, before the home page
    splash.render()
else:
    st.session_state["entered"] = True
    with st.sidebar:
        st.caption("Public data to September 2026. A status is a reason to ask a question, never proof.")
    nav.run()
