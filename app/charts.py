"""Charts for the FundTruth app (Altair). Presentation only."""
from __future__ import annotations

import math

import altair as alt
import numpy as np
import pandas as pd

GREEN, AMBER, RED = "#2E7D5B", "#A86A12", "#B42318"      # FundTruth status colours
NAVY, GREY, INK = "#1E4D78", "#8A94A0", "#14213D"
_FILL = {"High concern": "#f1d5d5", "Review": "#f6e8c3", "No major concern": "#d3e8dd"}


def _finish(chart, height):
    return (chart.properties(width="container", height=height)
            .configure(font="IBM Plex Sans")
            .configure_axis(labelColor="#3B4859", titleColor="#3B4859", gridColor="#E9EDF0", domainColor="#C9D1D6",
                            tickColor="#C9D1D6", labelFontSize=11.5, titleFontSize=11.5, titleFontWeight=500)
            .configure_legend(labelColor="#3B4859", labelFontSize=12, titleFontSize=12)
            .configure_title(color=INK, fontSize=12.5, fontWeight=500, anchor="start")
            .configure_view(strokeWidth=0))


def gauge(score):
    """Score on the 0-100 scale with the three provisional bands."""
    bands = pd.DataFrame({"start": [0, 50, 70], "end": [50, 70, 100],
                          "mid": [25, 60, 85],
                          "label": ["High concern", "Review", "No major concern"]})
    x = alt.Scale(domain=[0, 100])
    base = alt.Chart(bands).mark_rect().encode(
        x=alt.X("start:Q", scale=x, title=None, axis=alt.Axis(values=[0, 50, 70, 100])),
        x2="end:Q",
        color=alt.Color("label:N", legend=None,
                        scale=alt.Scale(domain=list(_FILL), range=list(_FILL.values()))))
    names = alt.Chart(bands).mark_text(fontSize=12, color=INK).encode(
        x=alt.X("mid:Q", scale=x), y=alt.value(18), text="label:N")
    mark = pd.DataFrame({"s": [float(score)]})
    rule = alt.Chart(mark).mark_rule(color=NAVY, size=4).encode(x=alt.X("s:Q", scale=x))
    return _finish(alt.layer(base, names, rule), 56)


def subscores(rows):
    """rows: DataFrame with test, score, flag (bool). Red = signal."""
    df = rows.copy()
    bars = alt.Chart(df).mark_bar(size=22).encode(
        y=alt.Y("test:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]),
                title="Subscore (100 = nothing unusual)"),
        color=alt.condition("datum.flag", alt.value(RED), alt.value(GREEN)),
        tooltip=["test", "score"])
    labels = alt.Chart(df).mark_text(align="left", dx=4, color=INK).encode(
        y=alt.Y("test:N", sort=None), x="score:Q", text=alt.Text("score:Q", format=".0f"))
    return _finish(alt.layer(bars, labels), 34 * max(len(df), 1) + 40)


def returns_hist(r, window):
    df = pd.DataFrame({"r": np.asarray(r, dtype=float)})
    hist = alt.Chart(df).mark_bar(color=NAVY, opacity=0.85).encode(
        x=alt.X("r:Q", bin=alt.Bin(maxbins=30), title="Monthly return",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("count():Q", title="Months"))
    win = alt.Chart(pd.DataFrame({"a": [-window], "b": [window]})).mark_rect(
        color=AMBER, opacity=0.35).encode(x="a:Q", x2="b:Q")
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color=INK, size=1).encode(x="z:Q")
    return _finish(alt.layer(hist, win, zero), 260)


def autocorr(ac, n):
    ac = [float(v) for v in ac]
    guide = 1.96 / math.sqrt(max(n, 1))
    m = max(max(abs(v) for v in ac), guide) * 1.35
    bars = alt.Chart(pd.DataFrame({"lag": ["1", "2", "3"], "ac": ac})).mark_bar(
        color=NAVY, size=40).encode(
        x=alt.X("lag:N", title="Lag (months)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("ac:Q", title="Autocorrelation", scale=alt.Scale(domain=[-m, m])))
    rules = alt.Chart(pd.DataFrame({"y": [guide, -guide]})).mark_rule(
        color=AMBER, strokeDash=[5, 4]).encode(y="y:Q")
    return _finish(alt.layer(bars, rules), 260)


def growth(r, factors):
    """Growth of 100 invested: the fund against the US market factor."""
    if factors is None or "Mkt-RF" not in factors or "RF" not in factors:
        return None
    f = factors.reindex(r.index)
    df = pd.DataFrame({"fund": r, "market": f["Mkt-RF"] + f["RF"]}).dropna()
    if len(df) < 12:
        return None
    g = (1.0 + df).cumprod() * 100.0
    long = g.reset_index(names="date").melt("date", var_name="series", value_name="value")
    long["series"] = long["series"].map({"fund": "This fund", "market": "US market (Fama-French)"})
    line = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title="Growth of 100", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=["This fund", "US market (Fama-French)"],
                                        range=[NAVY, GREY])))
    return _finish(line, 260)


# ---------------------------------------------------------------- context-aware charts
def _title(text):
    return alt.TitleParams(text, fontSize=12, anchor="start", color=INK)


def cumulative(fund, ref, ref_label, peer_median=None, ctx=""):
    """Growth of 100: fund vs benchmark (or regional market) and, if given, the peer median."""
    cols = {"This fund": fund, ref_label: ref}
    if peer_median is not None:
        cols["Peer median"] = peer_median
    df = pd.DataFrame(cols).dropna()
    if len(df) < 12:
        return None
    g = (1.0 + df).cumprod() * 100.0
    long = g.reset_index(names="date").melt("date", var_name="series", value_name="value")
    dom = list(cols)
    rng = [NAVY, GREY, AMBER][: len(dom)]
    ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="Growth of 100", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=dom, range=rng)))
    return _finish(ch.properties(title=_title(ctx)), 240)


def drawdown_chart(dd, ctx=""):
    df = dd.reset_index(); df.columns = ["date", "dd"]
    ch = alt.Chart(df).mark_area(color=RED, opacity=0.45).encode(
        x=alt.X("date:T", title=None), y=alt.Y("dd:Q", title="Drawdown", axis=alt.Axis(format=".0%")))
    return _finish(ch.properties(title=_title(ctx)), 200)


def rolling_ac_chart(roll, band, ctx=""):
    """Rolling 36-month lag-1 autocorrelation with the peer band (5th-95th percentile of
    peers' full-period lag-1 autocorrelation) if available."""
    df = roll.reset_index(); df.columns = ["date", "ac"]
    line = alt.Chart(df).mark_line(color=NAVY, strokeWidth=2).encode(
        x=alt.X("date:T", title=None), y=alt.Y("ac:Q", title="Lag-1 autocorrelation"))
    zero = alt.Chart(pd.DataFrame({"z": [0.0]})).mark_rule(color=GREY, size=1).encode(y="z:Q")
    layers = [line, zero]
    if band is not None:
        b = pd.DataFrame({"lo": [band[0]], "hi": [band[1]]})
        layers.insert(0, alt.Chart(b).mark_rect(color=GREEN, opacity=0.18).encode(y="lo:Q", y2="hi:Q"))
    return _finish(alt.layer(*layers).properties(title=_title(ctx)), 240)


def loadings_chart(fl, peer_median, ctx=""):
    """Factor betas with 95% intervals against the peer median (diamonds) when available."""
    rows = [dict(f=f, beta=fl[f"beta_{f}"], lo=fl[f"lo_{f}"], hi=fl[f"hi_{f}"]) for f in ("Mkt-RF", "SMB", "HML")]
    df = pd.DataFrame(rows)
    pts = alt.Chart(df).mark_point(color=NAVY, size=90, filled=True).encode(
        y=alt.Y("f:N", title=None, sort=None), x=alt.X("beta:Q", title="Factor loading"))
    bars = alt.Chart(df).mark_rule(color=NAVY, size=3).encode(y=alt.Y("f:N", sort=None), x="lo:Q", x2="hi:Q")
    layers = [bars, pts]
    if peer_median:
        pm = pd.DataFrame([dict(f=f, m=peer_median[f]) for f in ("Mkt-RF", "SMB", "HML") if peer_median.get(f) == peer_median.get(f)])
        if len(pm):
            layers.append(alt.Chart(pm).mark_point(shape="diamond", color=AMBER, size=120, filled=True).encode(
                y=alt.Y("f:N", sort=None), x="m:Q"))
    return _finish(alt.layer(*layers).properties(title=_title(ctx)), 150)


def peer_bars(pcts, ctx=""):
    """pcts: dict label -> percentile (0-1) within the peer set."""
    df = pd.DataFrame([dict(measure=k, pct=v) for k, v in pcts.items() if v == v])
    if df.empty:
        return None
    ch = alt.Chart(df).mark_bar(color=NAVY, size=18).encode(
        y=alt.Y("measure:N", sort=None, title=None),
        x=alt.X("pct:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%"), title="Percentile within the peer set"))
    return _finish(ch.properties(title=_title(ctx)), 28 * len(df) + 50)
