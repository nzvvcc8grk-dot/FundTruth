"""The FundTruth fund card: verdict first, evidence one click away.
Used for built-in funds (both modules) and for funds a user uploads."""
from __future__ import annotations

import os

import altair as alt
import pandas as pd
import streamlit as st

import charts
import connector as cn
import diagnostics as dg
import esg_promises as ml
import panel as pn
import readings as rd
import ui
from fund_profile import ASSET_CLASSES, PRICING, REGIONS, UNKNOWN, build_profile

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- header, verdict, numbers
def _verdict_text(p1, card, s):
    names = [t.split(":")[0] for _, t, _ in s["questions"]]
    if not names:
        bits = []
        if p1:
            bits.append("the reported returns behave as expected for this type of fund")
        if card:
            bits.append("the ESG promises are visible in the data that could be checked")
        return "Every check that applies found nothing that needs a question: " + " and ".join(bits) + "."
    txt = "Raised by " + ", ".join(names) + "."
    if p1 and s["returns"] == rd.NOTHING:
        txt += " The reported returns themselves behave as expected."
    if card and s["esg"] == rd.NOTHING:
        txt += " The ESG promises are visible in the data."
    return txt


def header(fid, p1, card, name=None, export=None):
    prof = p1.res.profile if p1 else None
    c1, c2 = st.columns([5, 1.25], vertical_alignment="bottom")
    with c1:
        st.markdown(f"## {ui.esc(name or ui.fund_name(fid))}")
        items = [] if name and fid.replace("_", " ") == name else [fid]   # no chip that repeats the title
        if prof:
            items += [prof.asset_class.replace("_", " ").capitalize() if prof.asset_class != UNKNOWN else "",
                      prof.region if prof.region != UNKNOWN else "", prof.currency if prof.currency != UNKNOWN else "",
                      ui.PRICING_WORDS.get(prof.pricing, prof.pricing)]
        elif card:
            items += ["Equity", card.claim.get("currency", "")]
        if card:
            items.append("ESG fund")
        st.markdown(ui.facts(items, strong={fid, "ESG fund"}), unsafe_allow_html=True)
    with c2:
        if export:
            with st.popover("Export", use_container_width=True, icon=":material/download:"):
                export()
    s = ui.summary(fid, p1, card)
    title = ("No questions for the manager" if s["n"] == 0 else
             f"{s['n']} question{'s' if s['n'] > 1 else ''} for the manager")
    col = ui.STATUS_HEX.get(s["level"], "#8A94A0")
    st.markdown(f'<div class="ft-verdict" style="--c:{col}">{ui.chip(s["level"])}'
                f'<div class="t">{ui.esc(title)}</div><div class="w">{ui.esc(_verdict_text(p1, card, s))}</div></div>',
                unsafe_allow_html=True)
    return s


def numbers(p1, card):
    items = []
    r = p1.returns if p1 else None
    if r is not None and len(r):
        stt = ui.return_stats(r)
        items += [("History analysed", f"{stt['n']} months", f"{stt['first']:%b %Y} to {stt['last']:%b %Y}"),
                  ("Return a year", f"{stt['ann']:+.1%}", "after fees, dividends reinvested"),
                  ("Volatility a year", f"{stt['vol']:.1%}", "standard deviation of returns"),
                  ("Worst fall", f"{stt['mdd']:.1%}", "peak to trough")]
    if card:
        m = card.metrics
        gap = m.get("fee_gap_gbp_per_10k", float("nan"))
        items.append(("Extra cost of the label", ("none" if abs(gap) < 0.5 else f"£{gap:,.0f} a year") if gap == gap else "n/a",
                      f"per £10,000 vs {card.claim['parent']}"))
        u = m.get("fossil_implied_underweight", float("nan"))
        if u == u:
            items.append(("Oil and gas vs index", f"{-u:+.1%}", "exposure implied by returns"))
    if items:
        ui.kpis(items)


def questions_panel(s):
    if not s["questions"]:
        st.markdown('<div class="ft-none">No question for the manager: nothing unusual in the checks that apply.</div>',
                    unsafe_allow_html=True)
        return
    lis = "".join(f'<li><span class="c">{ui.esc(t.split(":")[0])}.</span> {ui.esc(a)}</li>' for _, t, a in s["questions"])
    st.markdown(f'<div class="ft-qs"><h4>Questions for the manager</h4><ol>{lis}</ol></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- module 1: integrity of returns
def chart_for(p, x):
    """Evidence chart for a reading (moved from the original app; same charts)."""
    ctx = p.ctx
    if x.key == "price_regularity":
        roll = dg.rolling_autocorr(dg.excess(p.returns, p.factors))
        if len(roll):
            st.altair_chart(charts.rolling_ac_chart(roll, p.peer_band_ac1, "Rolling 36-month lag-1 autocorrelation"))
            st.caption("Green band: where most peers sit (5th to 95th percentile of their full-period value)."
                       if p.peer_band_ac1 else "No peer band available.")
    elif x.key == "style":
        fl = dg.factor_loadings(p.returns, p.factors)
        if fl:
            st.altair_chart(charts.loadings_chart(fl, p.peer_loadings, "Factor loadings with 95% intervals | " + p.res.factor_set.label))
            st.caption("Diamonds: peer median." if p.peer_loadings else "Peer median loadings are not available in this reference.")
        ref = p.benchmark if p.benchmark is not None and len(p.benchmark) >= 24 else p.market
        if ref is not None:
            g = charts.cumulative(p.returns, ref, p.benchmark_label or "Market", p.peer_median_returns, "Growth of 100")
            if g is not None:
                st.altair_chart(g)
    elif x.key == "alpha":
        fl = dg.factor_loadings(p.returns, p.factors)
        if fl:
            sub = dg.alpha_subperiods(p.returns, p.factors)
            rows = [dict(test="Full period", score=fl["alpha"] * 100, flag=False)] + [
                dict(test=k, score=v["alpha"] * 100, flag=False) for k, v in sub.items() if v["alpha"] == v["alpha"]]
            st.altair_chart(charts.subscores(pd.DataFrame(rows)).properties(title="Annualised alpha, % (bars are not scores)"))
    elif x.key == "risk_shape":
        st.altair_chart(charts.drawdown_chart(dg.drawdown(p.returns), "Fall from the previous peak"))
    elif x.key == "distribution":
        if x.tag == "primary":
            st.altair_chart(charts.returns_hist(p.returns.to_numpy(), rd.NEAR_ZERO_W))
            st.caption("The shaded strip is the ±0.5% window. An honest fund shows a smooth hump.")
    elif x.key == "peers":
        g = charts.peer_bars(p.peer_pcts, "Percentile within the peer set")
        if g is not None:
            st.altair_chart(g)


def _evidence(x, extra=None, show_peers=True):
    rows = [("What we measured", x.measured), ("What we found", x.found)]
    if show_peers:
        rows.append(("Compared with similar funds", getattr(x, "peer_range", "")))
    rows += [("How sure we are", x.uncertainty), ("Why it may be innocent", x.innocent), ("What to ask the manager", x.ask)]
    for lab, v in rows:
        if v:
            st.markdown(f'<div class="ft-label">{lab}</div>', unsafe_allow_html=True)
            st.write(v)
    if x.reason:
        st.caption((f"Why this reading is {x.tag}: " if x.tag else "") + x.reason)
    if extra:
        extra()
    lim = []
    if getattr(x, "fp_rate", None) is not None:
        lim.append(f"On simulated honest funds this check raised a flag about {x.fp_rate:.0f}% of the time "
                   "(synthetic data; the rate on real funds is unknown).")
    if x.not_validated:
        lim.append("Not validated: " + x.not_validated)
    if lim:
        st.caption(" ".join(lim))


def reading_row(x, extra=None, show_peers=True, key=""):
    name = ui.CHECK_NAMES.get(x.key, x.title.split(":")[0])
    label = f"{ui.badge_md(x.level)} **{name}** · {ui.in_short(x)}"
    if x.level == rd.NA:
        st.markdown(f"{ui.badge_md(x.level)} **{name}** · {x.reason or 'see data quality.'}")
        return
    with st.expander(label, expanded=False):
        st.markdown(f"**{x.title}**")
        _evidence(x, extra, show_peers)


def module1_tab(p):
    st.caption(f"{p.ctx}. Reference set chosen automatically: {p.reference_label}.")
    for x in p.readings:
        reading_row(x, extra=(lambda x=x: chart_for(p, x)))


def limits_tab(p, card):
    if p is not None:
        q = p.quality
        st.write(f"Months analysed: {q['months']} ({q['first']} to {q['last']}). Reference funds: {q['reference_size']}"
                 + ("" if q["reference_adequate"] else " (fewer than 20: peer-based readings are not meaningful)") + ".")
        st.write(f"Factors are in {q['factor_currency']} (US T-bill as the risk-free rate); fund currency: {p.res.profile.currency}.")
        for w in q["warnings"]:
            st.warning(w)
        for w in p.notes:
            st.caption(w.replace("FALLBACK: ", "Fewer similar funds than ideal: "))
        fp = q["fp_rates"]
        if fp:
            st.markdown("**How often each check raises a flag on simulated honest funds**")
            st.dataframe(pd.DataFrame({"Check": ["Price regularity", "Style and alpha", "Small gains and losses", "Position among peers",
                                                 "Benford (experimental, not used)"],
                                       "Flagged, %": [fp.get("flag_autocorrelation"), fp.get("flag_factor_regression"),
                                                      fp.get("flag_near_zero"), fp.get("flag_peer_anomaly"), fp.get("flag_benford")]}).round(1),
                         hide_index=True)
    st.markdown("**What is not validated.** There are no labelled fraud cases: detection is tested on simulated and "
                "injected anomalies, while on real funds only false alarms were audited. Closet indexing needs the expense "
                "ratio. Equity factors explain little of bond returns. Public data have survivorship bias.")
    if card is not None:
        st.markdown("**ESG checks.** Promises are read from the issuer's pages; holdings are a one-day "
                    "snapshot; Norway's exclusion list is an outside reference whose criteria differ from the fund's. "
                    "No ESG rating is used.")


def technical_tab(p):
    t = p.technical
    if not t:
        st.write("No score for this asset class.")
        return
    st.metric("Priority index (0-100)", f"{t['priority_index']:.0f}", help=t["label"])
    st.caption(f"{t['label'].capitalize()}. Reliability: {t['reliability']}.")
    st.dataframe(pd.DataFrame({"Test": list(t["subscores"]), "Subscore": [round(v) for v in t["subscores"].values()],
                               "Weight %": [round(t["weights"].get(k, 0)) for k in t["subscores"]]}), hide_index=True)
    st.code(t["report"], language=None)


# ---------------------------------------------------------------- module 2: ESG promises
def growth_chart(R, fid, parent):
    both = pd.concat([R[fid].dropna(), R[parent].dropna()], axis=1, join="inner").dropna()
    if len(both) < 12:
        return None
    g = (1 + both).cumprod() * 10000
    g.columns = [f"{fid} (ESG version)", f"{parent} (ordinary index fund)"]
    long = g.reset_index(names="date").melt("date", var_name="series", value_name="value")
    dom = list(g.columns)
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title="Value of 10,000 invested", scale=alt.Scale(zero=False), axis=alt.Axis(format=",.0f")),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=dom, range=[charts.NAVY, charts.GREY]),
                        legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("series:N", legend=None, scale=alt.Scale(domain=dom, range=[[1, 0], [5, 4]])),
        tooltip=[alt.Tooltip("date:T", format="%b %Y"), "series:N", alt.Tooltip("value:Q", format=",.0f")])
    return charts._finish(ch, 250)


def esg_tab(card, R, source_note):
    c = card.claim
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="ft-label">What the fund says</div>', unsafe_allow_html=True)
        for q in c["quotes"] or ["No promise text was given."]:
            st.markdown(f'<div class="ft-claim">{ui.esc(q)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ft-source">{ui.esc(source_note)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ft-label">What we compare it with</div>', unsafe_allow_html=True)
        st.markdown(f"**{c['parent']}**, {c['parent_name']}")
        st.markdown('<div class="ft-label">Fossil-fuel promise</div>', unsafe_allow_html=True)
        st.write(ml.PROMISE_LABEL[c["fossil_promise"]].capitalize())
        st.markdown('<div class="ft-label">Ongoing charges</div>', unsafe_allow_html=True)
        st.write(f"{c['expense_ratio']*100:.4g}% for the ESG fund, {c['parent_expense_ratio']*100:.4g}% for {c['parent']}")
    g = growth_chart(R, card.ticker, c["parent"]) if R is not None else None
    if g is not None:
        st.markdown('<div class="ft-label">What the data shows: the ESG version against its ordinary index fund</div>',
                    unsafe_allow_html=True)
        st.altair_chart(g)
    for x in card.readings:
        reading_row(x, show_peers=False)


# ---------------------------------------------------------------- the whole card
def render(fid, p1, card, *, R=None, name=None, pdf=True, source_note=None, key=""):
    def export():
        if pdf:
            try:
                st.download_button("One-page report (PDF)", ui.pdf(fid), f"FundTruth_{fid.replace('.', '_')}.pdf",
                                   "application/pdf", use_container_width=True, key="pdf" + key, icon=":material/picture_as_pdf:")
            except ValueError:
                pass
        if p1 is not None:
            st.download_button("Readings (CSV)", pn.panel_to_frame(p1).to_csv(index=False), f"fundtruth_{fid}.csv",
                               "text/csv", use_container_width=True, key="csv" + key, icon=":material/table:")
            st.download_button("Printable page (HTML)", pn.panel_to_html(p1), f"fundtruth_{fid}.html", "text/html",
                               use_container_width=True, key="html" + key, icon=":material/print:")
        if card is not None:
            st.download_button("ESG claim card (Markdown)", ml.card_markdown(card), f"fundtruth_esg_{fid.replace('.', '_')}.md",
                               "text/markdown", use_container_width=True, key="md" + key, icon=":material/description:")

    s = header(fid, p1, card, name=name, export=export)
    numbers(p1, card)
    questions_panel(s)
    tabs, bodies = [], []
    if p1 is not None:
        n1 = len(p1.readings)
        tabs.append(f"Are the returns credible? ({n1} checks)")
        bodies.append(lambda: module1_tab(p1))
    if card is not None:
        tabs.append(f"Does it keep its ESG promises? ({len(card.readings)} checks)")
        note = source_note or (f"Source: {card.claim.get('source', '')}. Extracted with an LLM.")
        bodies.append(lambda: esg_tab(card, R, note))
    tabs.append("Data and limits")
    bodies.append(lambda: limits_tab(p1, card))
    if p1 is not None and p1.technical:
        tabs.append("Technical details")
        bodies.append(lambda: technical_tab(p1))
    for t, body in zip(st.tabs(tabs), bodies):
        with t:
            body()


# ---------------------------------------------------------------- facts form (for uploads and tickers)
ASSET_WORDS = {"equity": "Shares", "bond": "Bonds", "multi-asset": "A mix of shares and bonds",
               "real_assets": "Property or infrastructure", "other": "Something else", UNKNOWN: "Not sure"}
REGION_WORDS = {"US": "United States", "Europe": "Europe", "Global": "Worldwide", "EM": "Emerging markets",
                "Other": "Elsewhere", UNKNOWN: "Not sure"}
PRICING_PLAIN = {"exchange_traded": "On a stock exchange (ETF, investment trust)",
                 "nav_listed_holdings": "Daily, from listed holdings (most mutual funds)",
                 "appraised_nav": "By appraisal or a model (property, private credit, private equity)",
                 UNKNOWN: "Not sure"}


def _choice(col, label, options, words, current, key, help=None):
    opts = list(options) + [UNKNOWN]
    cur = current if current in opts else UNKNOWN
    return col.radio(label, opts, index=opts.index(cur), format_func=words.get, key=key, help=help)


def profile_form(fid, prefill, key):
    """Three plain questions, then optional details; pre-filled when known (same fields as before)."""
    c = st.columns(3, gap="large")
    asset = _choice(c[0], "What does the fund mainly invest in?", ASSET_CLASSES, ASSET_WORDS,
                    prefill.get("asset_class", UNKNOWN), key + "a")
    region = _choice(c[1], "Where does it invest?", REGIONS, REGION_WORDS, prefill.get("region", UNKNOWN), key + "r",
                     help="Decides which market factors the fund is compared with.")
    pricing = _choice(c[2], "How is its price set?", PRICING, PRICING_PLAIN, prefill.get("pricing") or UNKNOWN, key + "p",
                      help="Prices set by appraisal are estimates: smooth returns are then expected, and read differently.")
    with st.expander("More details (optional): currency, benchmark, fees, start date"):
        d = st.columns(4)
        currency = d[0].text_input("Currency of the figures", prefill.get("currency", ""), key=key + "c",
                                   placeholder="USD, GBP, EUR").strip().upper()
        bproposal = cn.propose_benchmark(asset, region) or ""
        bench = d[1].text_input("Benchmark ticker", prefill.get("benchmark_id") or bproposal, key=key + "b",
                                help="An ETF that stands for the fund's declared benchmark. Leave empty if unknown.")
        er = prefill.get("expense_ratio", "")
        exp = d[2].text_input("Ongoing charge, % a year", f"{float(er) * 100:.2f}" if str(er).strip() not in ("", "nan") else "",
                              key=key + "e", placeholder="0.75", help="From the KID or factsheet. Needed to detect closet indexing.")
        start = d[3].text_input("Analyse from", prefill.get("start", cn.DEFAULT_START), key=key + "s")
    try:
        exp_dec = float(exp.replace(",", ".").rstrip("%")) / 100 if exp.strip() else None
    except ValueError:
        st.warning("The ongoing charge must be a number, for example 0.75 for 0.75% a year: it is ignored.")
        exp_dec = None
    meta = dict(region=region, currency=currency, asset_class=asset, pricing=pricing, benchmark_id=bench or None,
                expense_ratio=exp_dec)
    return build_profile(fid, meta), bench, start
