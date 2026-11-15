"""Watchlist: every screened fund, the ones that need a question first."""
import pandas as pd
import streamlit as st

import ui

ui.page_header("Watchlist",
               "Every fund in FundTruth, screened on both modules. Funds that raise a question for the manager come first. "
               "Select a row to open its fund card.")
ui.screen_all()
df = ui.watchlist_frame()

n_q = int((df["Questions"] > 0).sum())
n_unusual = int((df["Status"] == "Unusual").sum())
ui.kpis([("Funds screened", f"{len(df)}", "built-in, public data"),
         ("Need a question", f"{n_q}", "at least one check worth a look"),
         ("Unusual", f"{n_unusual}", "strong signals"),
         ("ESG funds", f"{int(df['ESG fund'].sum())}", "also checked against their promises")])

f1, f2, f3 = st.columns([1.6, 1, 1.2], vertical_alignment="bottom")
show = f1.segmented_control("Show", ["All funds", "With questions", "ESG funds"], default="All funds", key="wl_show")
assets = ["All asset classes"] + sorted(df["Asset class"].unique())
asset = f2.selectbox("Asset class", assets, key="wl_asset")
q = f3.text_input("Search", placeholder="Ticker or name", key="wl_q")

view = df.copy()
if show == "With questions":
    view = view[view["Questions"] > 0]
elif show == "ESG funds":
    view = view[view["ESG fund"]]
if asset != "All asset classes":
    view = view[view["Asset class"] == asset]
if q:
    m = view["Fund"].str.contains(q, case=False, regex=False) | view["Name"].str.contains(q, case=False, regex=False)
    view = view[m]
view = view.reset_index(drop=True)

COLS = {"Nothing unusual": ("#2E7D5B", "#E4F1EA"), "Worth a look": ("#A86A12", "#FAEFD9"),
        "Unusual": ("#B42318", "#FBE6E3"), "Not applicable": ("#5F6B7A", "#EDF0F3"), "Descriptive": ("#5F6B7A", "#EDF0F3")}


def paint(v):
    c = COLS.get(v)
    return f"color:{c[0]}; background-color:{c[1]}; font-weight:600" if c else "color:#8A94A0"


shown = view[["Fund", "Name", "Status", "Questions", "Returns check", "ESG check", "Raised by", "Asset class"]]
styled = shown.style.map(paint, subset=["Status", "Returns check", "ESG check"])

if view.empty:
    st.info("No fund matches these filters. Clear the search or choose another view.")
else:
    ev = st.dataframe(
        styled, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row", key="wl_table",
        height=min(38 + 35 * len(shown), 640),
        column_config={"Fund": st.column_config.TextColumn("Fund", width="small"),
                       "Name": st.column_config.TextColumn("Name", width="large"),
                       "Questions": st.column_config.NumberColumn("Questions", format="%d", width="small",
                                                                  help="Checks that raise a question for the manager"),
                       "Returns check": st.column_config.TextColumn("Returns credible?"),
                       "ESG check": st.column_config.TextColumn("ESG promises kept?"),
                       "Raised by": st.column_config.TextColumn("Raised by", width="medium")})
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        fid = view.loc[rows[0], "Fund"]
        c1, c2 = st.columns([1.3, 3], vertical_alignment="center")
        if c1.button(f"Open the fund card for {fid}", type="primary", icon=":material/open_in_new:"):
            ui.go("fund", fund=fid)
        c2.caption(view.loc[rows[0], "Name"])
    else:
        st.caption("Select a row to open the fund card.")

ui.footer()
