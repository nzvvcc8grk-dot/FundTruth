"""Home: what FundTruth is, what it checks, how to read it, and a live example."""
import streamlit as st

import readings as rd
import ui

hn = ui.headline_numbers()

left, right = st.columns([1.35, 1], gap="large", vertical_alignment="center")
with left:
    st.title("Is this fund what it claims to be?")
    st.markdown(
        '<p class="ft-lede">FundTruth checks a fund\'s reported returns and its sustainability promises against public '
        'data, then lists the questions to put to the manager, with the evidence behind each one. It is built for '
        'fund selectors, wealth managers and advisers who have to show their due diligence.</p>', unsafe_allow_html=True)
    b1, b2, _ = st.columns([1.1, 1.1, 0.6])
    if b1.button("Open the watchlist", type="primary", use_container_width=True, icon=":material/list_alt:"):
        ui.go("watchlist")
    if b2.button("Check your own fund", use_container_width=True, icon=":material/upload_file:"):
        ui.go("check")
    st.caption(f"{hn['funds']} funds screened, {hn['esg']} of them ESG funds. "
               f"{hn['with_q']} raise at least one question for the manager.")

with right:
    fid = "SPYX" if "SPYX" in ui.names() else None
    if fid:
        p1, card = ui.one(fid)
        s = ui.summary(fid, p1, card)
        rows = "".join(
            f'<div class="ft-row"><span>{ui.esc(lbl)}</span>{ui.chip(lv)}</div>'
            for lbl, lv in (("Are the returns credible?", s["returns"]), ("Does it keep its ESG promises?", s["esg"])) if lv)
        q = ", ".join(dict.fromkeys(t.split(":")[0] for _, t, _ in s["questions"]))
        title = f"{s['n']} question{'s' if s['n'] != 1 else ''} for the manager"
        st.markdown(
            f'<div class="ft-specimen"><div class="cap">Example from the watchlist</div>'
            f'<div style="font-weight:600;font-size:1.05rem">{ui.esc(ui.fund_name(fid))}</div>'
            f'<div class="ft-claim" style="font-size:.95rem;margin:10px 0 12px">{ui.esc(card.claim["quotes"][0])}</div>'
            f'<div style="font-size:1.25rem;font-weight:600;margin:4px 0 6px">{ui.esc(title)}</div>{rows}'
            + (f'<div class="ft-muted" style="font-size:.85rem;margin-top:8px">Raised by: {ui.esc(q)}</div>' if q else "")
            + '</div>', unsafe_allow_html=True)
        if st.button("Open this fund card", key="spec", icon=":material/arrow_forward:"):
            ui.go("fund", fund=fid)

st.write("")
st.subheader("What FundTruth checks")
m1, m2 = st.columns(2, gap="medium")
with m1:
    st.markdown(
        '<div class="ft-module"><h3>Are the reported returns credible?</h3>'
        '<span class="ft-muted">Five statistical checks on the monthly returns a fund reports, plus one descriptive view.</span>'
        '<ul><li>Returns too smooth for market prices, a sign of stale or estimated valuations</li>'
        '<li>An index tracker charging active fees (closet indexing)</li>'
        '<li>Returns beyond the market that look too good to be true</li>'
        '<li>Too few small losses compared with small gains</li>'
        '<li>A risk profile unlike similar funds</li></ul></div>', unsafe_allow_html=True)
with m2:
    st.markdown(
        '<div class="ft-module"><h3>Does an ESG fund keep its promises?</h3>'
        '<span class="ft-muted">Three checks comparing what the fund says with what it holds, how it behaves and what it costs.</span>'
        '<ul><li>Holdings against the exclusions it promises, and against Norway\'s sovereign-fund exclusion list</li>'
        '<li>For fossil-free promises, whether the returns still move with oil and gas</li>'
        '<li>How different the ESG version really is from its ordinary index fund, and the extra cost in pounds</li></ul>'
        '</div>', unsafe_allow_html=True)

st.write("")
st.subheader("How to read the results")
cells = "".join(f'<div>{ui.chip(lv)}<div class="d">{ui.esc(ui.STATUS_HELP[lv])}</div></div>'
                for lv in (rd.NOTHING, rd.LOOK, rd.UNUSUAL, rd.DESC, rd.NA))
st.markdown(f'<div class="ft-legend">{cells}</div>', unsafe_allow_html=True)
st.caption("A status is a reason to ask a question, never proof of wrongdoing.")

st.write("")
st.subheader("How it works")
st.markdown(
    '<div class="ft-steps">'
    '<div class="ft-step"><b>Pick a fund</b><span>From the watchlist of screened funds, or upload the monthly returns of '
    'your own. For an ESG fund, paste its promise from the KID or factsheet.</span></div>'
    '<div class="ft-step"><b>FundTruth runs the checks</b><span>On public data: prices with dividends reinvested, '
    'Fama-French market factors, issuer holdings files and Norway\'s exclusion list.</span></div>'
    '<div class="ft-step"><b>Ask the right questions</b><span>Read the verdict, open the evidence behind each check '
    'and export a one-page report for your due-diligence file.</span></div>'
    '<div class="ft-step"><b>FundTruth gets sharper</b><span>Each fund you check can join the comparison set, so the '
    'next fund is compared with more peers. The more funds, the more precise the checks.</span></div></div>',
    unsafe_allow_html=True)

ui.footer()
