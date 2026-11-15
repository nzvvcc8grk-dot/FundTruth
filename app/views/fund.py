"""Fund card for a built-in fund: both modules, verdict first."""
import streamlit as st

import fund_card as fc
import ui

ids = ui.fund_ids()
current = st.session_state.get("fund") or st.query_params.get("fund") or "SPYX"
if current not in ids:
    current = ids[0]
st.session_state["fund"] = current
st.session_state["fund_pick"] = current          # keeps the picker in step with the watchlist and the home page


def _picked():
    st.session_state["fund"] = st.session_state["fund_pick"]


label = {f: f"{ui.fund_name(f)} ({f})" for f in ids}
st.selectbox("Fund", ids, format_func=label.get, key="fund_pick", on_change=_picked,
             help="Type to search by name or ticker.")
st.query_params["fund"] = current

p1, card = ui.one(current)
fc.render(current, p1, card, R=ui.context()["R"], key=current)
ui.footer()
