"""Streamlit overview page: every fund, both modules, sorted by the number of questions for the manager."""
import pandas as pd
import streamlit as st

import fund_report as fr


@st.cache_resource(show_spinner=False)
def context():
    return fr.load_context()


@st.cache_data(show_spinner="Analysing every fund (first time only)...")
def table():
    return fr.overview(context())


@st.cache_data(show_spinner="Building the PDF...")
def pdf(fid):
    return fr.build_pdf(fid, context())


def download_button(fid, key):
    """PDF report button for a fund covered by the built-in data; nothing for other funds."""
    if fid not in fr.all_funds(context()):
        return
    st.download_button("Download fund report (PDF, one page)", pdf(fid), f"FundTruth_{fid.replace('.', '_')}.pdf",
                       "application/pdf", key=key)


def page():
    st.subheader("Overview: which funds need a question to the manager?")
    st.caption("Every fund in the built-in data, both modules. Funds are sorted by the number of checks that raise a "
               "question; ESG funds also go through module 2. A question is a reason to ask, never proof.")
    df = table()
    n, esg = len(df), int(df["ESG fund"].sum())
    c = st.columns(3)
    c[0].metric("Funds analysed", n)
    c[1].metric("ESG funds (module 2)", esg)
    c[2].metric("Funds with at least one question", int((df["Questions for the manager"] > 0).sum()))
    st.dataframe(df, hide_index=True, use_container_width=True,
                 column_config={"Questions for the manager": st.column_config.NumberColumn(format="%d"),
                                "ESG fund": st.column_config.CheckboxColumn()})
    fid = st.selectbox("Fund report", list(df["Fund"]),
                       format_func=lambda x: f"{x} ({int(df.set_index('Fund').loc[x, 'Questions for the manager'])} questions)")
    download_button(fid, "ov_pdf")
