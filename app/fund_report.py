"""FundTruth one-page fund report (PDF): module 1 (integrity of returns) + module 2 (ESG promises)
+ the questions to ask the manager. This page is the due-diligence evidence a client files.

    python fund_report.py SPYX            -> results/reports/FundTruth_SPYX.pdf
    python fund_report.py --all           -> one PDF per fund + results/overview.csv
"""
from __future__ import annotations

import argparse
import io
import os
import re
import warnings
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

import esg_promises as ml
import panel as pn
import universe as UN
from fund_profile import build_profile
from readings import DESC, LOOK, NA, NOTHING, UNUSUAL

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
FLAG = (LOOK, UNUSUAL)
LEVEL_HEX = {NOTHING: "#2e7d5b", LOOK: "#b07a00", UNUSUAL: "#b33a3a", DESC: "#6b7785", NA: "#6b7785"}
SNAP = os.path.join(HERE, "snapshot_yahoo")


# ---------------------------------------------------------------- data

def _promise_quote(quotes):
    """The sentence that states the exclusions (the last one that says 'exclu...' or 'eliminat...'), else the last."""
    hits = [q for q in quotes if "exclu" in q.lower() or "eliminat" in q.lower()]
    return (hits or quotes or [""])[-1]

def load_context():
    """Universe snapshot (40 funds, factors, frozen reference, peers) + ESG returns, claims, holdings, exclusions."""
    UV = UN.load_snapshot(SNAP) if os.path.exists(os.path.join(SNAP, "snapshot.json")) else None
    d = os.path.join(HERE, "data")
    R = pd.read_csv(os.path.join(d, "esg_returns.csv"), index_col="date", parse_dates=True) \
        if os.path.exists(os.path.join(d, "esg_returns.csv")) else None
    return dict(UV=UV, R=R, claims=ml.load_claims(), hdir=os.path.join(d, "holdings"),
                excl=ml.load_exclusions(os.path.join(d, "nbim_exclusions.csv")))


def all_funds(ctx):
    a = sorted(ctx["UV"].returns) if ctx["UV"] is not None else []
    b = [t for t in ctx["claims"]["funds"] if ctx["R"] is not None and t in ctx["R"]]
    return a + [t for t in b if t not in a]


def module1(fid, ctx):
    """Integrity readings for a fund of the 40-fund universe, or for an ESG fund (equity, exchange traded)."""
    UV, R = ctx["UV"], ctx["R"]
    if UV is None:
        return None
    if fid in UV.returns:
        prof = UV.profiles[fid]
        bl = UV.meta.loc[fid].get("benchmark_id")
        lab = UV.meta.loc[fid].get("category_label")
        return pn.analyse_panel(UV.returns[fid], prof, UV.factors, UV.reference, UV.reference_label,
                                UV.benchmarks.get(bl), bl if isinstance(bl, str) else "", UV.peer_stats, None,
                                category_label=lab if isinstance(lab, str) else None)
    c = ctx["claims"]["funds"].get(fid)
    if c is None or R is None or fid not in R:
        return None
    region = "US" if c.get("currency", "USD") == "USD" else ("Europe" if fid == "UKEL.L" else "Global")
    prof = build_profile(fid, dict(region=region, currency=c.get("currency", "USD"), asset_class="equity",
                                   pricing="exchange_traded", benchmark_id=c["parent"],
                                   expense_ratio=c["expense_ratio"]))
    bench = R[c["parent"]].dropna() if c["parent"] in R else None
    return pn.analyse_panel(R[fid].dropna(), prof, UV.factors, UV.reference, UV.reference_label,
                            bench, c["parent"], UV.peer_stats, None)


def module2(fid, ctx):
    if fid not in ctx["claims"]["funds"] or ctx["R"] is None:
        return None
    try:
        return ml.analyse_fund(fid, ctx["R"], ctx["claims"], ctx["hdir"], ctx["excl"])
    except ValueError:
        return None


def questions(p1, card):
    q = [("Integrity", x.title, x.ask) for x in (p1.readings if p1 else [])
         if x.level in FLAG and x.ask and x.tag != "informational"]   # same rule as the headline
    q += [("ESG", r.title.split(":")[0], r.ask) for r in (card.readings if card else []) if r.level in FLAG and r.ask]
    return q


def overview(ctx):
    rows = []
    for fid in all_funds(ctx):
        p1, card = module1(fid, ctx), module2(fid, ctx)
        q = questions(p1, card)
        name = ctx["claims"]["funds"].get(fid, {}).get("name", "")
        rows.append({"Fund": fid, "Name": name, "ESG fund": card is not None,
                     "Integrity of returns": p1.headline if p1 else "not analysed",
                     "ESG promises": card.headline if card else "",
                     "Questions for the manager": len(q),
                     "Flagged checks": "; ".join(t for _, t, _ in q)})
    df = pd.DataFrame(rows)[["Fund", "Questions for the manager", "Integrity of returns", "ESG promises",
                             "Flagged checks", "Name", "ESG fund"]]
    return df.sort_values(["Questions for the manager", "ESG fund", "Fund"], ascending=[False, False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- PDF
def _short(text, n=330):
    """First sentences of a reading, cut at a sentence boundary under n characters."""
    s = re.split(r"(?<=[.!?])\s+", str(text).strip())
    out = ""
    for x in s:
        if len(out) + len(x) + 1 > n:
            break
        out = (out + " " + x).strip()
    return out or (str(text)[:n].rsplit(" ", 1)[0] + "...")


def _esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_pdf(fid, ctx=None):
    ctx = ctx or load_context()
    p1, card = module1(fid, ctx), module2(fid, ctx)
    if p1 is None and card is None:
        raise ValueError(f"{fid}: no data for this fund")
    c = ctx["claims"]["funds"].get(fid, {})
    ink, quiet, line = colors.HexColor("#1b2733"), colors.HexColor("#5b6773"), colors.HexColor("#c9d1da")
    st = {k: ParagraphStyle(k, fontName=f, fontSize=s, leading=l, textColor=col, spaceAfter=a)
          for k, f, s, l, col, a in [("h1", "Helvetica-Bold", 16, 19, ink, 2), ("sub", "Helvetica", 8.5, 11, quiet, 6),
                                     ("h2", "Helvetica-Bold", 10.5, 13, colors.HexColor("#1f4e79"), 3),
                                     ("body", "Helvetica", 8, 10.2, ink, 2), ("small", "Helvetica", 7, 9, quiet, 0),
                                     ("cell", "Helvetica", 7.4, 9.2, ink, 0), ("cellb", "Helvetica-Bold", 7.4, 9.2, ink, 0),
                                     ("big", "Helvetica-Bold", 11, 14, ink, 4)]}
    q = questions(p1, card)
    story = [Paragraph(f"FundTruth fund report: {_esc(fid)}", st["h1"]),
             Paragraph(_esc(f"{c.get('name', '')}{' | ' if c.get('name') else ''}"
                            f"{p1.ctx if p1 else ''} | generated {date.today():%d %b %Y}"), st["sub"]),
             Paragraph((f"{len(q)} question{'s' if len(q) != 1 else ''} for the manager" if q else "No questions for the manager")
                       + (f" &nbsp;|&nbsp; Integrity: {_esc(p1.headline)}" if p1 else "")
                       + (f" &nbsp;|&nbsp; ESG: {_esc(card.headline)}" if card else ""), st["big"])]

    def table(rows):
        data = [[Paragraph("<b>Check</b>", st["cell"]), Paragraph("<b>Status</b>", st["cell"]),
                 Paragraph("<b>What we found</b>", st["cell"])]]
        for title, level, found in rows:
            data.append([Paragraph(_esc(title), st["cellb"]),
                         Paragraph(f'<font color="{LEVEL_HEX.get(level, "#6b7785")}"><b>{_esc(level)}</b></font>', st["cell"]),
                         Paragraph(_esc(found), st["cell"])])
        t = Table(data, colWidths=[42 * mm, 22 * mm, 112 * mm], repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f6")),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        return t

    if p1:
        rows = [(x.title.split(":")[0], x.level, (x.reason if x.level == NA else _short(x.found, 800 if x.level in FLAG else 250)))
                for x in p1.readings]
        story += [Paragraph("1. Integrity of returns: are the reported returns credible?", st["h2"]), table(rows), Spacer(1, 5)]
    if card:
        story += [Paragraph("2. ESG promises: does the fund do what its documents promise?", st["h2"]),
                  Paragraph("<i>Promise (verbatim):</i> &ldquo;" + _esc(_promise_quote(c["quotes"]))
                            + f"&rdquo; &nbsp;<font color='#5b6773'>Source: {_esc(c['source'])}</font>", st["body"])]
        rows = [(r.title.split(":")[0], r.level, r.reason if r.level == NA else _short(r.found, 900 if r.level in FLAG else 420)) for r in card.readings]
        story += [table(rows), Spacer(1, 5)]
    if q:
        story.append(Paragraph("Questions for the manager", st["h2"]))
        story += [Paragraph(f"{i}. <b>{_esc(t)}</b> ({m}): {_esc(a)}", st["body"]) for i, (m, t, a) in enumerate(q, 1)]
    else:
        story.append(Paragraph("No check raised a question for this fund.", st["body"]))
    story += [Spacer(1, 6), KeepTogether([Paragraph(
        "A status is a statistical signal and a reason to ask a question, never proof of misreporting or greenwashing. "
        "Data: Yahoo Finance prices (dividends reinvested), Kenneth R. French factors, issuer holdings files and fund pages, "
        "Norges Bank exclusion list (via OpenSanctions). Thresholds are provisional; detection is validated on simulated and "
        "injected anomalies, not on labelled real cases.", st["small"])])]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=13 * mm, bottomMargin=12 * mm,
                      title=f"FundTruth report {fid}", author="FundTruth").build(story)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("funds", nargs="*")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    ctx = load_context()
    out = os.path.join(HERE, "results", "reports")
    os.makedirs(out, exist_ok=True)
    funds = all_funds(ctx) if a.all else a.funds
    for f in funds:
        with open(os.path.join(out, f"FundTruth_{f.replace('.', '_')}.pdf"), "wb") as fh:
            fh.write(build_pdf(f, ctx))
    if a.all:
        overview(ctx).to_csv(os.path.join(HERE, "results", "overview.csv"), index=False)
    print(f"{len(funds)} report(s) in {out}")


if __name__ == "__main__":
    main()
