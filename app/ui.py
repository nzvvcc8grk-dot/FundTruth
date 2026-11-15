"""FundTruth interface layer: visual identity, shared components and cached data access.
Presentation only: every number comes from the engine (fundtruth.py), the readings
(readings.py, esg_promises.py) and the report builder (fund_report.py)."""
from __future__ import annotations

import html
import os

import numpy as np
import pandas as pd
import streamlit as st

import diagnostics as dg
import fund_report as fr
import readings as rd

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- vocabulary
# One word per status, everywhere in the product.
STATUS = {rd.NOTHING: ("Nothing unusual", "ok", "green"),
          rd.LOOK: ("Worth a look", "ask", "orange"),
          rd.UNUSUAL: ("Unusual", "bad", "red"),
          rd.DESC: ("Descriptive", "desc", "gray"),
          rd.NA: ("Not applicable", "na", "gray")}
STATUS_HELP = {rd.NOTHING: "The check found nothing that needs a question.",
               rd.LOOK: "Something deserves a question to the manager.",
               rd.UNUSUAL: "A strong signal: ask before relying on the fund.",
               rd.DESC: "Context for the reader, not a judgement.",
               rd.NA: "The check does not apply to this fund, or the data is missing."}
ORDER = {rd.UNUSUAL: 0, rd.LOOK: 1, rd.NOTHING: 2, rd.DESC: 3, rd.NA: 4}
STATUS_HEX = {rd.NOTHING: "#2E7D5B", rd.LOOK: "#A86A12", rd.UNUSUAL: "#B42318", rd.DESC: "#5F6B7A", rd.NA: "#8A94A0"}
PRICING_WORDS = {"exchange_traded": "Exchange traded", "nav_listed_holdings": "NAV from listed holdings",
                 "appraised_nav": "Appraised NAV", "unknown": "Pricing not stated"}
CHECK_NAMES = {"price_regularity": "Price regularity", "style": "Style and benchmark", "alpha": "Return beyond the market",
               "risk_shape": "Shape of risk", "distribution": "Small gains and losses", "peers": "Position among peers",
               "promise_vs_holdings": "Promise vs portfolio", "fossil_behaviour": "Promise vs behaviour",
               "label_vs_index": "Label vs index"}

# Plain-language one-liners shown before the evidence (key, level) -> sentence
IN_SHORT = {
    ("price_regularity", rd.NOTHING): "Month-to-month returns show no sign of smoothing or stale prices.",
    ("price_regularity", rd.LOOK): "Returns look smoother than market prices would allow: ask how assets are valued.",
    ("price_regularity", rd.UNUSUAL): "Returns are much smoother than market prices allow: ask how assets are valued.",
    ("style", rd.NOTHING): "Market exposure is consistent with the fund's declared style.",
    ("style", rd.LOOK): "The fund moves almost like its index: check that the fee matches the management.",
    ("style", rd.UNUSUAL): "The fund behaves like an index tracker: check that the fee matches the management.",
    ("alpha", rd.NOTHING): "Returns beyond the market are in the normal range for similar funds.",
    ("alpha", rd.LOOK): "Returns beyond the market are high for similar funds: ask where they come from.",
    ("alpha", rd.UNUSUAL): "Returns beyond the market look too good: ask where they come from.",
    ("risk_shape", rd.DESC): "Falls, skew and fat tails of the returns, shown as context.",
    ("distribution", rd.NOTHING): "Small gains and small losses occur in normal proportions.",
    ("distribution", rd.LOOK): "Fewer small losses than expected: ask how month-end values are set.",
    ("distribution", rd.UNUSUAL): "Far fewer small losses than expected: ask how month-end values are set.",
    ("peers", rd.NOTHING): "Risk and return profile sits within its peer group.",
    ("peers", rd.LOOK): "The profile stands out from similar funds: ask why.",
    ("peers", rd.UNUSUAL): "The profile is far from similar funds: ask why.",
    ("promise_vs_holdings", rd.NOTHING): "Holdings are consistent with the exclusions the fund promises.",
    ("promise_vs_holdings", rd.LOOK): "Some holdings may conflict with the promised exclusions.",
    ("promise_vs_holdings", rd.NA): "No holdings file for this fund.",
    ("fossil_behaviour", rd.NOTHING): "Returns confirm less oil and gas than the ordinary index fund.",
    ("fossil_behaviour", rd.LOOK): "Returns still move with oil and gas about as much as the ordinary index fund.",
    ("fossil_behaviour", rd.UNUSUAL): "Returns move with oil and gas more than the ordinary index fund.",
    ("fossil_behaviour", rd.DESC): "No oil-and-gas promise: exposure shown for information.",
    ("label_vs_index", rd.NOTHING): "The ESG version differs from its index enough, or costs no more.",
    ("label_vs_index", rd.LOOK): "Almost identical to its ordinary index fund, but more expensive.",
}


def status_label(level):
    return STATUS.get(level, (str(level).capitalize(), "na", "gray"))[0]


def badge_md(level):
    """Native Streamlit badge (works inside expander labels)."""
    label, _, colour = STATUS.get(level, ("Not applicable", "na", "gray"))
    return f":{colour}-badge[{label}]"


def in_short(x):
    s = IN_SHORT.get((x.key, x.level))
    if s is None:
        s = (x.reason if x.level == rd.NA else "") or "See the evidence."
    if getattr(x, "tag", "") == "informational" and x.level in (rd.LOOK, rd.UNUSUAL):
        s += " Informational for this type of fund."
    return s


# ---------------------------------------------------------------- style
CSS = """
<style>
:root { --ink:#14213D; --ledger:#1E4D78; --rule:#DCE2E4; --mist:#F3F6F5; --muted:#5B6878;
        --ok:#2E7D5B; --ok-bg:#E4F1EA; --ask:#A86A12; --ask-bg:#FAEFD9; --bad:#B42318; --bad-bg:#FBE6E3;
        --desc:#5F6B7A; --desc-bg:#EDF0F3; }
.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }
p, li { font-feature-settings: "tnum" 1; }
h1 { letter-spacing: -0.012em; }
.ft-muted { color: var(--muted); }
.ft-lede { font-size: 1.08rem; line-height: 1.6; color: #2B3A52; max-width: 62ch; }
.ft-chip { display:inline-flex; align-items:center; gap:7px; padding:2px 11px 2px 9px; border-radius:999px;
           font-size:.8rem; font-weight:600; line-height:1.65; white-space:nowrap; }
.ft-chip::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }
.ft-chip.ok { color:var(--ok); background:var(--ok-bg); } .ft-chip.ask { color:var(--ask); background:var(--ask-bg); }
.ft-chip.bad { color:var(--bad); background:var(--bad-bg); } .ft-chip.desc, .ft-chip.na { color:var(--desc); background:var(--desc-bg); }
.ft-facts { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0 2px; }
.ft-fact { display:inline-block; padding:1px 9px; border:1px solid var(--rule); border-radius:6px; font-size:.8rem;
           color:#3B4859; background:#fff; line-height:1.7; }
.ft-fact.strong { border-color:#BFD0E0; color:var(--ledger); background:#F2F6FA; font-weight:600; }
.ft-verdict { border:1px solid var(--rule); border-left:6px solid var(--c); border-radius:10px; padding:18px 22px 16px;
              background:#fff; margin:14px 0 4px; }
.ft-verdict .t { font-size:1.55rem; font-weight:600; color:var(--ink); line-height:1.25; margin:10px 0 6px; }
.ft-verdict .w { color:var(--muted); font-size:.98rem; line-height:1.55; max-width:80ch; }
.ft-kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); border:1px solid var(--rule);
           border-radius:10px; background:#fff; overflow:hidden; margin:14px 0 6px; }
.ft-kpi { padding:13px 18px 12px; border-right:1px solid var(--rule); }
.ft-kpi:last-child { border-right:none; }
.ft-kpi .l { font-size:.8rem; color:var(--muted); }
.ft-kpi .v { font-size:1.4rem; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; line-height:1.35; }
.ft-kpi .s { font-size:.76rem; color:var(--muted); }
.ft-claim { font-family:"IBM Plex Serif", Georgia, serif; font-size:1.06rem; line-height:1.62; color:var(--ink);
            border-left:3px solid var(--ledger); padding:2px 0 2px 16px; margin:10px 0; max-width:72ch; }
.ft-claim::before { content:"\\201C"; } .ft-claim::after { content:"\\201D"; }
.ft-source { font-size:.8rem; color:var(--muted); margin:-2px 0 8px 19px; }
.ft-label { font-size:.82rem; font-weight:600; color:var(--muted); margin:16px 0 3px; }
.ft-qs { border:1px solid #EBD9B6; background:#FFFBF3; border-radius:10px; padding:14px 20px 8px; margin:12px 0 8px; }
.ft-qs h4 { margin:0 0 6px; font-size:1rem; color:var(--ink); }
.ft-qs ol { margin:0 0 6px 1.1rem; padding:0; } .ft-qs li { margin:0 0 8px; line-height:1.5; }
.ft-qs .c { font-weight:600; color:var(--ask); }
.ft-none { border:1px solid var(--rule); background:var(--mist); border-radius:10px; padding:12px 18px; margin:12px 0 8px;
           color:#3B4859; }
.ft-legend { display:grid; grid-template-columns:repeat(auto-fit, minmax(165px, 1fr)); gap:10px 22px; margin:8px 0; }
.ft-legend .d { font-size:.86rem; color:#3B4859; margin-top:4px; line-height:1.45; }
.ft-steps { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:18px; margin:8px 0; counter-reset:s; }
.ft-step { border-top:2px solid var(--ledger); padding-top:10px; }
.ft-step b { display:block; color:var(--ink); margin-bottom:3px; }
.ft-step b::before { counter-increment:s; content:counter(s) ". "; color:var(--ledger); }
.ft-step span { color:#3B4859; font-size:.92rem; line-height:1.5; }
.ft-module { border:1px solid var(--rule); border-radius:12px; padding:18px 22px 10px; background:#fff; height:100%; }
.ft-module h3 { margin-top:0; }
.ft-module ul { padding-left:1.1rem; margin:6px 0 10px; } .ft-module li { margin:0 0 6px; color:#2B3A52; }
.ft-specimen { border:1px solid var(--rule); border-radius:12px; padding:16px 18px; background:#fff;
               box-shadow:0 1px 0 rgba(20,33,61,.04), 0 12px 32px -18px rgba(20,33,61,.28); }
.ft-specimen .cap { font-size:.8rem; color:var(--muted); margin-bottom:8px; }
.ft-row { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:8px 0;
          border-top:1px solid #EEF1F3; font-size:.92rem; }
.ft-row:first-of-type { border-top:none; }
.ft-foot { font-size:.82rem; color:var(--muted); border-top:1px solid var(--rule); margin-top:36px; padding-top:12px; }
[data-testid="stSidebarNav"] a span { font-size:.95rem; }
[data-testid="stExpander"] details { border-radius:10px; }
[data-testid="stExpander"] summary p { font-size:.95rem; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(t):
    return html.escape(str(t))


def chip(level, text=None):
    label, cls, _ = STATUS.get(level, ("Not applicable", "na", "gray"))
    return f'<span class="ft-chip {cls}">{esc(text or label)}</span>'


def facts(items, strong=()):
    return '<div class="ft-facts">' + "".join(
        f'<span class="ft-fact{" strong" if i in strong else ""}">{esc(i)}</span>' for i in items if i) + "</div>"


def kpis(items):
    """items: list of (label, value, sub)"""
    cells = "".join(f'<div class="ft-kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div>'
                    f'<div class="s">{esc(s)}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="ft-kpis">{cells}</div>', unsafe_allow_html=True)


def page_header(title, lede=None):
    st.title(title)
    if lede:
        st.markdown(f'<p class="ft-lede">{lede}</p>', unsafe_allow_html=True)


def footer(text="A status is a reason to ask a question, never proof of wrongdoing. Public data; thresholds are provisional."):
    st.markdown(f'<div class="ft-foot">{esc(text)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- data (cached)
@st.cache_resource(show_spinner=False)
def context():
    return fr.load_context()


@st.cache_data(show_spinner=False)
def names():
    p = os.path.join(HERE, "fund_names.csv")
    out = dict(pd.read_csv(p).values) if os.path.exists(p) else {}
    for t, v in context()["claims"]["funds"].items():
        out[t] = v["name"]
    return out


def fund_name(fid):
    return names().get(fid, fid)


@st.cache_resource(show_spinner=False)
def one(fid):
    """(module-1 panel, module-2 claim card) for a built-in fund; computed once per session server."""
    ctx = context()
    return fr.module1(fid, ctx), fr.module2(fid, ctx)


def screen_all():
    ids = fr.all_funds(context())
    if "screened" not in st.session_state:
        with st.spinner("Screening every fund (first time only, about 20 seconds)..."):
            out = {fid: one(fid) for fid in ids}
        st.session_state["screened"] = True
        return out
    return {fid: one(fid) for fid in ids}


@st.cache_data(show_spinner=False)
def headline_numbers():
    """Counts for the home page, from the saved overview (fast) or by screening (slow)."""
    p = os.path.join(HERE, "results", "overview.csv")
    if os.path.exists(p):
        o = pd.read_csv(p)
        return dict(funds=len(o), with_q=int((o["Questions for the manager"] > 0).sum()), esg=int(o["ESG fund"].sum()))
    df = watchlist_frame()
    return dict(funds=len(df), with_q=int((df["Questions"] > 0).sum()), esg=int(df["ESG fund"].sum()))


def funds():
    return list(screen_all())


def fund_ids():
    """All built-in funds, sorted by display name (no screening needed)."""
    return sorted(fr.all_funds(context()), key=lambda f: fund_name(f).lower())


@st.cache_data(show_spinner="Building the one-page report...")
def pdf(fid):
    return fr.build_pdf(fid, context())


def primary(readings):
    return [x for x in readings if getattr(x, "tag", "") != "informational" and x.level in (rd.NOTHING, rd.LOOK, rd.UNUSUAL)]


def worst(readings):
    lv = [x.level for x in primary(readings)]
    return min(lv, key=ORDER.get) if lv else None


def summary(fid, p1, card):
    """Fund-level verdict in the product's vocabulary."""
    q = fr.questions(p1, card)
    r1 = p1.readings if p1 else []
    r2 = card.readings if card else []
    w1, w2 = worst(r1), worst(r2)
    allw = [w for w in (w1, w2) if w]
    level = min(allw, key=ORDER.get) if allw else rd.NA
    prof = p1.res.profile if p1 else None
    return dict(fund=fid, name=fund_name(fid), level=level, questions=q, n=len(q), returns=w1, esg=w2,
                esg_fund=card is not None, asset=(prof.asset_class if prof else "equity"))


@st.cache_data(show_spinner=False)
def watchlist_frame():
    rows = []
    for fid, (p1, card) in screen_all().items():
        s = summary(fid, p1, card)
        rows.append({"Fund": fid, "Name": s["name"],
                     "Asset class": s["asset"].replace("_", " ").capitalize(),
                     "Status": status_label(s["level"]), "Questions": s["n"],
                     "Returns check": status_label(s["returns"]) if s["returns"] else "",
                     "ESG check": status_label(s["esg"]) if s["esg"] else "",
                     "Raised by": ", ".join(t.split(":")[0] for _, t, _ in s["questions"]),
                     "ESG fund": s["esg_fund"], "_order": ORDER.get(s["level"], 9)})
    df = pd.DataFrame(rows).sort_values(["_order", "Questions", "ESG fund", "Fund"], ascending=[True, False, False, True])
    return df.drop(columns="_order").reset_index(drop=True)


def return_stats(r):
    r = r.dropna()
    n = len(r)
    ann = (1 + r).prod() ** (12 / n) - 1 if n else np.nan
    vol = r.std(ddof=1) * np.sqrt(12) if n > 1 else np.nan
    mdd = float(dg.drawdown(r).min()) if n else np.nan
    return dict(n=n, first=r.index[0] if n else None, last=r.index[-1] if n else None, ann=ann, vol=vol, mdd=mdd)


# ---------------------------------------------------------------- navigation registry
PAGES = {}


def go(page_key, fund=None):
    if fund:
        st.session_state["fund"] = fund
    st.switch_page(PAGES[page_key])


# ---------------------------------------------------------------- continuous improvement: funds added by users
CONTRIB = os.path.join(HERE, "data", "contributed_reference.csv")


def contributions():
    """Comparison-set rows added by users (summary statistics only, anonymous ids)."""
    if os.path.exists(CONTRIB):
        try:
            return pd.read_csv(CONTRIB, index_col=0)
        except (ValueError, OSError):
            pass
    return pd.DataFrame()


def reference_for_checks():
    """Frozen public reference + the funds users added. The watchlist keeps the frozen one (reproducible)."""
    UV = context()["UV"]
    base, c = UV.reference, contributions()
    if c.empty:
        return base, UV.reference_label
    c = c[[k for k in base.columns if k in c.columns]]
    return pd.concat([base, c]), f"{UV.reference_label} + {len(c)} added by users"


def comparison_counts():
    """Funds per group today: public reference + added by users."""
    ref, _ = reference_for_checks()
    return ref["category"].value_counts().to_dict(), len(contributions())


def contribute(r, prof):
    """Add one fund's summary statistics to the comparison set. Returns (added, message)."""
    import hashlib
    from context import build_reference
    key = "U_" + hashlib.sha1(np.round(r.to_numpy(dtype=float), 8).tobytes()).hexdigest()[:10]
    old = contributions()
    if key in old.index:
        return False, "This fund is already in the comparison set."
    row = build_reference({key: r}, {key: prof}, context()["UV"].factors)
    if row.empty:
        return False, ("Only equity, bond and multi-asset funds with at least 36 months can join the comparison set: "
                       "tell us what the fund invests in, above.")
    os.makedirs(os.path.dirname(CONTRIB), exist_ok=True)
    pd.concat([old, row]).to_csv(CONTRIB, index_label="fund")
    n = len(old) + 1
    return True, f"Added. The comparison set now includes {n} fund{'s' if n != 1 else ''} added by users."
