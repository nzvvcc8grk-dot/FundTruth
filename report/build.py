"""Builds report.html (5 A4 pages) for the FundTruth business proposal.

Run from this folder: python build.py  (FUNDTRUTH_DIR = the FundTruth app folder, default ../01_app_FundTruth)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FUNDTRUTH_DIR") or next(
    (p for p in (os.path.join(HERE, "..", "app"), os.path.join(HERE, "..", "01_app_FundTruth")) if os.path.isdir(p)),
    os.path.join(HERE, "..", "app"))
REPO_URL = os.environ.get("REPO_URL", "")   # e.g. https://github.com/<account>/FundTruth
LINK = (f'<a class="repo" href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a>' if REPO_URL
        else '<span class="todo">[link to the anonymous GitHub repository]</span>')
NAVY, BLUE, SKY, INK, MUTED, RULE, HAIR = "#0B2A4A", "#2457C5", "#9DB7E0", "#0F1B2D", "#5B6878", "#C9D2DD", "#E2E7EE"
OK, LOOK, BAD = "#1F7A55", "#AD6800", "#B42318"

_ex = [0]


def ex(title, light=""):
    """Numbered exhibit heading."""
    _ex[0] += 1
    lt = f' <span class="light">{light}</span>' if light else ""
    return f'<h3><span class="n">Exhibit {_ex[0]}</span>{title}{lt}</h3>'


def st(level):
    cls = {"Nothing unusual": "ok", "Worth a look": "look", "Unusual": "bad"}[level]
    return f'<span class="st {cls}">{level}</span>'


def T(x, y, s, size=9.5, w=400, fill=INK, anchor="start", italic=False):
    it = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{fill}" '
            f'text-anchor="{anchor}"{it}>{s}</text>')


MARK = ('<svg viewBox="0 0 40 40"><rect width="40" height="40" rx="8" fill="{bg}"/>'
        '<path d="M10 28.5 H30" stroke="#fff" stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>'
        '<path d="M12.5 17.5 L18 23 L28.5 11.5" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round" '
        'stroke-linejoin="round"/></svg>')

SECTIONS = ["Problem", "Product", "Method", "Evidence", "Business"]


def top(n):
    tr = "".join(f'<span class="{"on" if i + 1 == n else ""}"><b>{i + 1}</b>{s}</span>' for i, s in enumerate(SECTIONS))
    return (f'<div class="top"><div class="wm">{MARK.format(bg=NAVY)}FundTruth</div>'
            f'<nav class="track">{tr}</nav></div>')


def head(n, label, title, lead):
    return (f'{top(n)}<p class="label"><b>{n}</b>{label}</p><h1 class="action">{title}</h1>'
            f'<p class="lead">{lead}</p>')


def footer(n, mid=""):
    return (f'<div class="foot"><span>FundTruth business proposal, November 2026</span><span>{mid}</span>'
            f'<span>{n} / 5</span></div>')


# ------------------------------------------------------------------ cover chart: SPYX growth, nine checks
R = pd.read_csv(f"{DATA}/data/esg_returns.csv", index_col="date", parse_dates=True)
g = (1 + R["SPYX"].dropna()).cumprod().to_numpy()
ov = pd.read_csv(f"{DATA}/results/overview.csv").set_index("Fund").loc["SPYX"]
flagged = [x.strip() for x in str(ov["Flagged checks"]).split(";") if x.strip() and x.strip() != "nan"]
CHECKS = ["Price regularity", "Style and benchmark", "Return beyond the market", "Shape of risk",
          "Small gains and losses", "Position among peers", "Promise vs portfolio", "Promise vs behaviour",
          "Label vs index"]
W, H = 1000, 122
x = np.linspace(0, W, len(g))
y = H - 9 - (g - g.min()) / (g.max() - g.min()) * (H - 30)
line = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in zip(x, y))
dots = []
for k, c in enumerate(CHECKS):
    f = 0.03 + k * 0.94 / (len(CHECKS) - 1)
    yy = float(np.interp(f * W, x, y))
    col = "#9AA8B8" if c == "Shape of risk" else "#F2B04A" if c in flagged else "#4FC08D"
    ring = (f'<circle cx="{f*W:.1f}" cy="{yy:.1f}" r="14" fill="none" stroke="{col}" stroke-opacity=".55" '
            f'stroke-width="2.4"/>') if c in flagged else ""
    dots.append(f'{ring}<circle cx="{f*W:.1f}" cy="{yy:.1f}" r="7" fill="{col}" stroke="{NAVY}" stroke-width="3"/>')
HERO = (f'<svg viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">'
        f'<path d="{line} L{W} {H} L0 {H} Z" fill="#13355A"/>'
        f'<path d="{line}" fill="none" stroke="#DCE6F2" stroke-width="1.7" stroke-linejoin="round"/>{"".join(dots)}</svg>')


# ------------------------------------------------------------------ exhibits (SVG units = CSS px: 1 unit = 0.75pt)
F = 8.4   # chart text size in px (6.3pt)


def chart_controls():
    w, x0, bw = 272, 96, 108
    s = [T(0, 9, "Found when planted in the 40 real funds", 8.8, 600)]
    rows = [("Strong smoothing", 100, "100% (40 of 40)"), ("Hidden small losses", 92.5, "92.5% (37 of 40)"),
            ("Mild smoothing", 32.5, "32.5% (13 of 40)")]
    yy = 15
    for lab, v, txt in rows:
        s.append(T(0, yy + 7.2, lab, F))
        s.append(f'<rect x="{x0}" y="{yy}" width="{bw * v / 100:.1f}" height="9" fill="{NAVY}"/>')
        s.append(T(x0 + bw * v / 100 + 3, yy + 7.2, txt, F, 600, NAVY))
        yy += 13
    yy += 11
    s.append(T(0, yy, "Flagged by mistake on honest funds (average per check)", 8.8, 600))
    yy += 6
    sc, top_y = 23.0, yy
    for lab, v in [("Price regularity", 2.5), ("Style and benchmark", 3.75), ("Small gains and losses", 3.3),
                   ("Position among peers", 4.2)]:
        s.append(T(0, yy + 7.2, lab, F))
        s.append(f'<rect x="{x0}" y="{yy}" width="{sc * v:.1f}" height="9" fill="#7F9CC8"/>')
        s.append(T(x0 + sc * v - 3, yy + 7.2, f"{v:g}%", F - .4, 600, "#fff", "end"))
        yy += 13
    lx = x0 + sc * 5
    s.append(f'<line x1="{lx:.1f}" x2="{lx:.1f}" y1="{top_y - 2}" y2="{yy - 2}" stroke="{INK}" stroke-width=".9" '
             f'stroke-dasharray="3 2"/>')
    s.append(T(lx + 4, top_y + 7, "5%: the most", F - .6, 400, MUTED))
    s.append(T(lx + 4, top_y + 16, "a fair test", F - .6, 400, MUTED))
    s.append(T(lx + 4, top_y + 25, "allows", F - .6, 400, MUTED))
    return f'<svg viewBox="0 0 {w} {yy + 1}" width="100%">{"".join(s)}</svg>'


def chart_spyx():
    w, x0, k = 283, 98, 42.0
    s = []
    for v in range(5):
        xx = x0 + v * k
        s.append(f'<line x1="{xx}" x2="{xx}" y1="8" y2="56" stroke="{HAIR}" stroke-width="1"/>')
        s.append(T(xx, 66, f"{v}", F, 400, MUTED, "middle"))
    s.append(T(x0 + 2 * k, 77, "Energy underweight vs SPY, percentage points", F, 400, MUTED, "middle"))
    s.append(T(0, 22, "From returns", F + .4, 600))
    s.append(T(0, 31, "129 months, t = −7.5", F - .4, 400, MUTED))
    s.append(f'<line x1="{x0 + 2.0 * k}" x2="{x0 + 3.3 * k}" y1="24" y2="24" stroke="{NAVY}" stroke-width="1.8"/>')
    for e in (2.0, 3.3):
        s.append(f'<line x1="{x0 + e * k}" x2="{x0 + e * k}" y1="20.5" y2="27.5" stroke="{NAVY}" stroke-width="1.3"/>')
    s.append(f'<circle cx="{x0 + 2.6 * k}" cy="24" r="4.2" fill="{NAVY}"/>')
    s.append(T(x0 + 2.6 * k, 14.5, "2.6% (2.0 to 3.3)", F, 600, NAVY, "middle"))
    s.append(T(0, 45, "From holdings", F + .4, 600))
    s.append(T(0, 54, "one day, 1.2% vs 3.6%", F - .4, 400, MUTED))
    c = x0 + 2.4 * k
    s.append(f'<path d="M{c} 42.5 L{c + 4.6} 47 L{c} 51.5 L{c - 4.6} 47 Z" fill="{LOOK}"/>')
    s.append(T(c + 8, 50, "2.4%", F, 600, LOOK))
    return f'<svg viewBox="0 0 {w} 81" width="100%">{"".join(s)}</svg>'


def chart_learning():
    d = pd.read_csv(f"{DATA}/results/learning_curve.csv")
    w, h, x0, x1, yt, yb = 272, 124, 24, 166, 13, 96
    ns = [10, 20, 40, 80, 160]
    X = {n: x0 + i * (x1 - x0) / 4 for i, n in enumerate(ns)}
    Y = lambda v: yb - v / 100 * (yb - yt)
    s = []
    sx = x0 + np.log2(1.6) * (x1 - x0) / 4
    s.append(f'<rect x="{x0}" y="{yt}" width="{sx - x0:.1f}" height="{yb - yt}" fill="#E6EDF8"/>')
    s.append(T(x0 + 1, yt - 4, "today: 10 to 16 per type", F - .2, 600, BLUE))
    for v in (0, 50, 100):
        s.append(f'<line x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{HAIR}" stroke-width="1"/>')
        s.append(T(x0 - 3, Y(v) + 3, f"{v}%", F - .4, 400, MUTED, "end"))
    for n in ns:
        s.append(T(X[n], yb + 10, str(n), F - .2, 400, MUTED, "middle"))
    s.append(T((x0 + x1) / 2, yb + 21, "Funds of the same type in the comparison set (log scale)", F - .6, 400, MUTED,
               "middle"))
    m = d[d.kind == "mild smoothing"].set_index("n_ref")
    band = [(X[n], Y(m.loc[n, "smoothing_unusual_pct_hi"])) for n in ns] + \
           [(X[n], Y(m.loc[n, "smoothing_unusual_pct_lo"])) for n in reversed(ns)]
    s.append('<path d="M' + " L".join(f"{a:.1f} {b:.1f}" for a, b in band) + ' Z" fill="#7F9CC8" fill-opacity=".25"/>')
    spec = [("strong smoothing", NAVY, "Strong smoothing", "100% flagged"),
            ("mild smoothing", "#4F74B8", "Mild smoothing", "62% flagged"),
            ("honest", LOOK, "Honest funds", "1.5% flagged by mistake")]
    for kind, col, l1, l2 in spec:
        q = d[d.kind == kind].set_index("n_ref")["smoothing_unusual_pct"]
        pts = [(X[n], Y(q.loc[n])) for n in ns]
        s.append('<path d="M' + " L".join(f"{a:.1f} {b:.1f}" for a, b in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="1.8"/>')
        s += [f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.3" fill="{col}"/>' for a, b in pts]
        ly = pts[-1][1] + (2 if kind != "honest" else -6)
        s.append(T(x1 + 6, ly, l1, F, 600, col))
        s.append(T(x1 + 6, ly + 9, l2, F - .4, 400, MUTED))
    return f'<svg viewBox="0 0 {w} {h}" width="100%">{"".join(s)}</svg>'


def chart_plan():
    w, lab_w = 272, 88
    cx = [lab_w + 30, lab_w + 92, lab_w + 154]
    base, hmax, vmax, bw = 78, 56, 1470, 30
    stacks = [(90, 45, 0), (360, 180, 40), (900, 450, 120)]
    costs = [220, 450, 750]
    cols = [NAVY, BLUE, SKY]
    s = []
    lx = 0
    for name, col in zip(["Professional", "Compliance", "Platform"], cols):
        s.append(f'<rect x="{lx}" y="2.5" width="7" height="7" fill="{col}"/>')
        s.append(T(lx + 10, 9, name, F))
        lx += 10 + len(name) * 4.1 + 10
    s.append(f'<line x1="{lx}" x2="{lx + 13}" y1="6" y2="6" stroke="{INK}" stroke-width="1.2" stroke-dasharray="3 2"/>')
    s.append(T(lx + 16, 9, "Costs", F))
    for c, st_, cost in zip(cx, stacks, costs):
        yy = base
        for v, col in zip(st_, cols):
            hh = v / vmax * hmax
            if hh > 0:
                s.append(f'<rect x="{c - bw / 2}" y="{yy - hh:.1f}" width="{bw}" height="{hh:.1f}" fill="{col}"/>')
            yy -= hh
        tot = sum(st_)
        cy = base - cost / vmax * hmax
        s.append(f'<line x1="{c - bw / 2 - 5}" x2="{c + bw / 2 + 5}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{INK}" '
                 f'stroke-width="1.2" stroke-dasharray="3 2"/>')
        ty = min(yy, cy) - 3.5
        s.append(T(c, ty, f"£{tot}k" if tot < 1000 else f"£{tot / 1000:.2f}m", F + .4, 600, NAVY, "middle"))
    s.append(f'<line x1="0" x2="{w}" y1="{base}" y2="{base}" stroke="{RULE}" stroke-width="1"/>')
    rows = [("", ["Year 1", "Year 2", "Year 3"], 600),
            ("Professional users", ["50", "200", "500"], 400),
            ("Compliance clients", ["3", "12", "30"], 400),
            ("Platform licences", ["0", "1", "3"], 400),
            ("Revenue", ["£135k", "£580k", "£1.47m"], 600),
            ("Costs", ["£220k", "£450k", "£750k"], 400)]
    yy = base + 10.5
    for lab, vals, wt in rows:
        s.append(T(0, yy, lab, F, 600 if lab == "Revenue" else 400))
        s += [T(c, yy, v, F, wt, INK, "middle") for c, v in zip(cx, vals)]
        yy += 10
    s.append(f'<line x1="0" x2="{w}" y1="{yy - 6.5}" y2="{yy - 6.5}" stroke="{NAVY}" stroke-width="1"/>')
    yy += 3
    s.append(T(0, yy, "Operating result", F, 600))
    for c, v in zip(cx, ["−£85k", "+£130k", "+£720k"]):
        s.append(T(c, yy, v, F + .4, 600, BAD if v.startswith("−") else OK, "middle"))
    return f'<svg viewBox="0 0 {w} {yy + 3}" width="100%">{"".join(s)}</svg>'


def hb(v):
    c = f'<circle cx="5" cy="5" r="4.1" fill="{"%s" % NAVY if v == 1 else "#fff"}" stroke="{NAVY}" stroke-width="1"/>'
    if v == 0.5:
        c += f'<path d="M5 0.9 A4.1 4.1 0 0 1 5 9.1 Z" fill="{NAVY}"/>'
    return f'<svg viewBox="0 0 10 10">{c}</svg>'


CSS = open(os.path.join(HERE, "style.css")).read()

# ================================================================== PAGE 1
P1 = f"""
<section class="page">
 <div class="hero">
  <div class="row"><div class="brand">{MARK.format(bg="#2A5F91")}<span>FundTruth</span></div>
   <div class="meta"><b>Business proposal</b><br>Data Science for Asset Management<br>London, November 2026</div></div>
  <h1>Funds make promises.<br>We check them against the evidence.</h1>
  <div class="pillars">
   <div><b>Returns and ESG promises</b>FundTruth tests a fund's reported returns and its own ESG promises against
   public data: prices, market factors, issuer holdings, exclusion lists.</div>
   <div><b>Evidence behind every signal</b>Each check shows the numbers, the peer range and why the signal may be
   innocent. No paid ratings, no black box.</div>
   <div><b>A question, not an accusation</b>Every warning sign becomes a question for the manager, ready for the
   due-diligence file.</div>
  </div>
  <div class="hero-cap">SPYX, Jan 2016 to Sep 2026. Each dot is one of the nine checks:<i style="background:#4FC08D"></i>nothing
   unusual<i style="background:#F2B04A"></i>worth a look<i style="background:#9AA8B8"></i>descriptive</div>
  <div class="hero-chart">{HERO}</div>
 </div>
 <div class="kpis">
  <div><b>48</b><span>real funds screened, 8 of them ESG funds</span></div>
  <div><b>9</b><span>evidence checks in two modules</span></div>
  <div><b>100%</b><span>of strong return smoothing caught when planted in real funds</span></div>
  <div><b>2.5-4.2%</b><span>false-alarm rate per check on honest funds</span></div>
  <div><b>7 of 48</b><span>funds raise at least one question for the manager</span></div>
 </div>
 <p class="label"><b>1</b>The problem</p>
 <h1 class="action">Fund claims are cheap to make and expensive to check</h1>
 <p class="lead">Every fund makes two kinds of promise. <b>Numbers</b>: the returns and the risk it reports.
 <b>Words</b>: its style, its benchmark and, increasingly, its ESG exclusions. Advisers, fund selectors and investors
 rely on both, but almost nobody tests them. Checking today means reading factsheets by hand or buying ESG ratings
 that measure something else.</p>
 <div class="g c2">
  <div class="mod"><h3>Failure 1: returns that are too smooth or too good</h3>
  <p>Funds holding illiquid assets (property, private credit, hedge strategies) are valued by appraisal or by model.
  Stale or managed valuations make returns look smoother and safer than the assets really are (Getmansky, Lo &amp;
  Makarov, 2004). Managers also avoid reporting small losses (Bollen &amp; Pool, 2009). Elsewhere, “active” funds
  charge active fees while hugging an index (Cremers &amp; Petajisto, 2009). The extreme case, Madoff, was famous for
  an implausibly smooth track record.</p></div>
  <div class="mod"><h3>Failure 2: ESG promises the portfolio does not keep</h3>
  <p>Greenwashing is now an enforcement issue. The SEC fined DWS <b>$19 million</b> in September 2023 for misleading
  ESG statements, its largest greenwashing penalty, after fines on Goldman Sachs Asset Management ($4m, 2022) and BNY
  Mellon ($1.5m, 2022). ESG ratings do not settle the question: providers disagree widely (Berg, Kölbel &amp;
  Rigobon, 2022), and a rating says how “green” a fund looks, not whether it does what its own documents promise.</p></div>
 </div>
 <div class="mod mt">{ex("Why now: in the UK and EU, ESG claims have become a matter of proof")}
  <div class="timeline">
   <div><b>31 Jul 2023</b>FCA Consumer Duty: firms must evidence fair value for the products they recommend</div>
   <div><b>31 May 2024</b>FCA anti-greenwashing rule: sustainability claims must be fair, clear and capable of being substantiated</div>
   <div><b>21 Nov 2024</b>ESMA guidelines on fund names apply to new EU funds using ESG terms</div>
   <div><b>2 Dec 2024</b>FCA SDR naming and marketing rules for UK funds</div>
   <div><b>21 May 2025</b>ESMA fund-name guidelines extend to existing EU funds</div>
  </div>
  <p class="src">Source: FCA; ESMA (see references).</p>
 </div>
 <div class="g c75 mt">
  <div class="mod"><h3>A large market that has to document its checks</h3>
   <div class="market">
    <div><b>$3.51tn</b><span>in sustainable funds at the end of March 2026, about 85% of it in Europe</span></div>
    <div><b>£11.1tn</b><span>managed in the UK in 2025, a record</span></div>
    <div><b>31,000</b><span>UK financial advisers in about 5,500 firms, advising on £1tn of client assets</span></div>
   </div>
   <p class="note" style="margin-top:2mm">Every one of them must show that the funds they recommend are what they
   claim to be.</p>
   <p class="src">Sources: Morningstar Sustainalytics (2026); Investment Association (2026); FCA (2025 survey).</p>
  </div>
  <div class="answer"><p class="lab">Our answer in one line</p>
   <p>A tool that reads what a fund says, measures what it does, and turns every gap into a precise, polite question
   for the manager: never an accusation, always a documented reason to ask.</p></div>
 </div>
 {footer(1)}
</section>"""

# ================================================================== PAGE 2
ARR = f'<div class="arr"><svg viewBox="0 0 8 14"><path d="M1 1 L7 7 L1 13" fill="none" stroke="{NAVY}" stroke-width="1.4"/></svg></div>'
P2 = f"""
<section class="page">
 {head(2, "The product", "One screen: a verdict, the evidence and a question for the manager",
 "FundTruth is a web app on a Python engine. A user picks a fund from the watchlist or adds their own, and in seconds "
 "receives a plain-English verdict, the numbers behind each check, and the questions to send to the manager.")}
 <div class="mod">{ex("How FundTruth works")}
  <div class="flow">
   <div class="stage"><h4><span class="n">1</span>Inputs</h4><p><b>Public data</b>: monthly prices, market factors,
   issuer holdings files, the fund's own documents (KID, factsheet).</p><p><b>Your data</b>: paste from Excel or a
   factsheet, or upload a file. Units and dates are recognised automatically, and files stay on your computer.</p></div>
   {ARR}
   <div class="stage"><h4><span class="n">2</span>Nine checks in two modules</h4>
    <div class="mods">
     <div><b>Integrity of returns</b>: are the reported numbers credible?<p>Price regularity, Style and benchmark,
     Return beyond the market, Small gains and losses, Position among peers (machine learning), Shape of risk.</p></div>
     <div><b>ESG promises</b>: does the fund do what it promises?<p>Promise vs portfolio, Promise vs behaviour, Label
     vs index. No ESG ratings: the fund's own words are the benchmark.</p></div>
    </div></div>
   {ARR}
   <div class="stage"><h4><span class="n">3</span>Output</h4><div class="stlist">{st("Nothing unusual")}
   {st("Worth a look")}{st("Unusual")}</div><p>Evidence and peer range for every check, <b>questions for the
   manager</b>, a one-page PDF report, and the option to add the fund to the comparison set.</p></div>
  </div>
 </div>
 <div class="shots mt">
  <div class="mod">{ex("The fund card")}<figure><img src="assets/fund_card_SPYX.png"></figure>
   <p class="cap">The verdict comes first, then six headline numbers (including the yearly cost of the ESG label),
   then the questions for the manager. The checks sit in tabs below.</p></div>
  <div class="mod">{ex("The ESG module")}<figure><img src="assets/esg_module_SPYX.png"></figure>
   <p class="cap">The fund's promise is quoted word for word with its source, then compared with its ordinary index
   fund: holdings, behaviour and fees.</p></div>
 </div>
 <div class="mod mt"><h3>The user journey</h3>
  <div class="steps">
   <div><span class="k">1</span><div><b>Screen</b>The watchlist ranks 48 funds by the number of open questions,
   with filters by asset class, ESG status and search.</div></div>
   <div><span class="k">2</span><div><b>Inspect</b>The fund card shows the verdict, the evidence behind each check,
   and what would make the signal innocent.</div></div>
   <div><span class="k">3</span><div><b>Check your own fund</b>Paste a factsheet's return table or upload a file.
   Three simple questions pick the right factors and peers.</div></div>
   <div><span class="k">4</span><div><b>Document</b>Export the one-page PDF, send the questions to the manager and
   file the answer: an audit trail for Consumer Duty and SDR.</div></div>
  </div>
 </div>
 <div class="g c75 mt">
  <div class="mod"><h3>Who uses it, and why they pay</h3>
  <table class="t">
   <tr><th style="width:31%">User</th><th>Job to be done</th></tr>
   <tr><td>Fund selectors, wealth managers</td><td>Screen a fund range and document due diligence before a fund goes
   on the buy list.</td></tr>
   <tr><td>Independent financial advisers</td><td>Show the FCA that recommended funds are what they claim to be, at a
   fraction of the cost of a research platform.</td></tr>
   <tr><td>Compliance teams at asset managers</td><td>Test their own ESG marketing against their own portfolios
   before a regulator or journalist does.</td></tr>
   <tr><td>Fund platforms</td><td>Monitor every fund on the shelf through an API and flag the ones that need a
   question.</td></tr>
  </table></div>
  <div class="mod diff"><h3>What makes it different</h3>
  <ul>
   <li><b>Questions, not scores.</b> Each signal becomes a concrete question, for example: <i>“which holdings does
   the ESG screen actually remove, and how is the extra fee justified?”</i></li>
   <li><b>Promise-based, not rating-based.</b> The fund is tested against its own binding words, so the verdict is
   objective and easy to defend.</li>
   <li><b>Two lenses in one place.</b> Return integrity and ESG claims are usually sold separately; FundTruth checks
   both.</li>
   <li><b>Built for illiquid funds.</b> The smoothing and hidden-loss tests target property, private-credit and hedge
   funds, where the problem lives.</li>
   <li><b>Private by design.</b> Return files stay on the user's computer; only anonymous summary statistics can join
   the comparison set.</li>
  </ul></div>
 </div>
 {footer(2)}
</section>"""

# ================================================================== PAGE 3
P3 = f"""
<section class="page">
 {head(3, "Method and data", "Nine checks, each built on a published method and free public data",
 f"A status needs statistical evidence: {st('Unusual')} if p &lt; 0.05, {st('Worth a look')} if p &lt; 0.10, otherwise {st('Nothing unusual')}. "
 f"P-values come from a <b>fixed reference set</b> of real funds, which changes only when a user adds a fund to it.")}
 <div class="mod">{ex("From public data to a question for the manager")}
  <div class="pipe">
   <div><h4><span class="n">1</span>Data</h4>Monthly returns 2016-2026, Fama-French factors, issuer holdings,
   exclusion list</div>
   <div><h4><span class="n">2</span>Features</h4>Autocorrelation, betas and alpha, returns near zero, volatility,
   skew, energy slope, active share</div>
   <div><h4><span class="n">3</span>Tests and models</h4>Ljung-Box, OLS and Newey-West regressions, binomial
   test, Isolation Forest, k-means</div>
   <div><h4><span class="n">4</span>Calibration</h4>Empirical p-values against 40 real funds; thresholds fixed
   in advance</div>
   <div><h4><span class="n">5</span>Output</h4>Status, evidence, innocent explanations, manager question</div>
  </div>
  <p class="src"><b>Universe:</b> 40 liquid US funds and ETFs (16 equity, 14 bond, 10 multi-asset) plus 8 ESG funds
  compared with their ordinary index fund (SPYX, EFIV, ESGU, SUSA, ESGV, CRBN; London-listed SUSW.L and V3AB.L,
  analysed in GBP).</p>
 </div>
 <div class="mod mt">{ex("The nine checks")}
 <table class="t method">
  <tr><th>Check</th><th>Question</th><th>Method</th><th>Basis</th></tr>
  <tr class="grp"><td colspan="4">Module 1: Integrity of returns</td></tr>
  <tr><td>Price regularity</td><td>Are returns too smooth for market prices?</td>
   <td>Lag 1-3 autocorrelation, Ljung-Box test, rank against peers; AR(1) de-smoothing shows the volatility the fund
   would report on market prices</td><td>Getmansky, Lo &amp; Makarov (2004)</td></tr>
  <tr><td>Style and benchmark</td><td>Is an “active” fund an index tracker in disguise?</td>
   <td>Fama-French three-factor regression; R² ≥ 0.95 with low tracking error and an above-median fee; R² and
   tracking error against the declared benchmark</td><td>Cremers &amp; Petajisto (2009); Fama &amp; French (1993)</td></tr>
  <tr><td>Return beyond the market</td><td>Is the alpha too good to be true?</td><td>Annualised alpha, t ≥ 2 and
   upper tail against peers; stability across sub-periods (2020 crash, 2022-23)</td><td>Fama &amp; French (1993)</td></tr>
  <tr><td>Small gains and losses</td><td>Are small losses being hidden?</td><td>Binomial test: returns just above vs
   just below zero, against the fund's own distribution</td><td>Bollen &amp; Pool (2009)</td></tr>
  <tr><td>Position among peers</td><td>Does the risk profile look unlike similar funds?</td><td>Machine learning:
   Isolation Forest on volatility, skewness, autocorrelation, alpha and R², plus k-means distance</td>
   <td>Liu, Ting &amp; Zhou (2008)</td></tr>
  <tr><td>Shape of risk</td><td>What does the risk look like? (descriptive)</td><td>Maximum drawdown, skewness,
   kurtosis, up- and down-capture</td><td>Context only, no status</td></tr>
  <tr class="grp"><td colspan="4">Module 2: ESG promises</td></tr>
  <tr><td>Promise vs portfolio</td><td>Does the fund hold what it says it excludes?</td><td>Issuer holdings against
   the parent index, the energy sector and the Norges Bank (Norway's sovereign fund) exclusion list, matched to the
   fund's own exclusion wording (owning coal reserves is not the same as burning coal)</td><td>Holdings files;
   NBIM list</td></tr>
  <tr><td>Promise vs behaviour</td><td>Does a “fossil-free” fund still move with oil and gas?</td><td>Regression of
   the fund's return over its ordinary index fund on energy's return over the same fund (formula below), Newey-West
   errors. Works without holdings data</td><td>Newey &amp; West (1987)</td></tr>
  <tr><td>Label vs index</td><td>Is the ESG version really different, and what does the label cost?</td><td>R²,
   tracking error and active share against the ordinary index fund; fee gap in £ per £10,000 a year and compounded
   over 20 years</td><td>Cremers &amp; Petajisto (2009)</td></tr>
 </table></div>
 <div class="g c2 mt">
  <div class="mod"><h3>Two key formulas</h3>
   <p><b>Smoothed reporting.</b> The observed return is a moving average of true returns, creating autocorrelation
   and understating risk.</p>
   <div class="eq">R<sup>obs</sup><sub>t</sub> = θ<sub>0</sub>R<sub>t</sub> + θ<sub>1</sub>R<sub>t−1</sub> +
   θ<sub>2</sub>R<sub>t−2</sub>, &nbsp; θ<sub>0</sub> + θ<sub>1</sub> + θ<sub>2</sub> = 1<br>
   de-smoothed: &nbsp;R̂<sub>t</sub> = (R<sup>obs</sup><sub>t</sub> − ρ R<sup>obs</sup><sub>t−1</sub>) / (1 − ρ)<br>
   ρ: lag-1 autocorrelation</div>
   <p><b>Promise vs behaviour.</b> If the ESG fund drops energy stocks, its return over the parent moves against
   energy; the slope estimates the missing weight.</p>
   <div class="eq" style="margin-bottom:0">r<sub>ESG</sub> − r<sub>parent</sub> = a + b · (r<sub>energy</sub> −
   r<sub>parent</sub>) + ε<br>implied energy underweight: w = −b / (1 − b) ≈ −b</div>
  </div>
  <div><div class="mod"><h3>Engineering you can audit</h3><ul>
   <li><b>Frozen statistical engine</b> protected by a hash test. New features are built on top, never inside.</li>
   <li><b>161 automated tests</b> are all passing. The key numbers were recomputed by separate code and match the app.</li>
   <li><b>Stack:</b> Python (pandas, statsmodels, scikit-learn), Streamlit, ReportLab. Double-click launch on Mac
   and Windows; runs offline after installation.</li></ul></div>
  <div class="mod mt"><h3>Design principles</h3><ul>
   <li><b>A status is a reason to ask, never proof.</b> Each check also lists why the signal may be innocent.</li>
   <li><b>Promises are read, not invented.</b> A language model extracts the exclusions into a fixed format and
   never infers a promise the text does not make; quote and source are always shown.</li>
   <li><b>No silent corrections.</b> Wrong units, gaps or stale prices stop the analysis with a message; nothing
   is interpolated.</li></ul></div></div>
 </div>
 {footer(3)}
</section>"""

# ================================================================== PAGE 4
P4 = f"""
<section class="page">
 {head(4, "Evidence", "FundTruth finds planted problems and rarely flags honest funds",
 "No public dataset labels which funds misreport, so we test FundTruth the way a laboratory tests a diagnostic. We "
 "<b>plant known anomalies</b> in real funds (positive control), count <b>false alarms</b> on simulated honest funds, "
 "run it on <b>liquid funds</b> where little should be found (negative control), and examine a <b>real ESG case</b> "
 "with two independent methods.")}
 <div class="g c57">
  <div class="mod">{ex("Detection and false alarms")}{chart_controls()}
   <p class="src">Source: FundTruth validation on the 40 real US funds (Jan 2016-Sep 2026) and on simulated honest
   funds; tests at the 5% level.</p></div>
  <div class="mod"><h3>What the controls show</h3><ul>
   <li><b>Positive control on real funds.</b> We distorted the returns of all 40 liquid funds. Strong smoothing was
   found in <b>40 of 40</b>, mild smoothing in 13 of 40 and hidden small losses in <b>37 of 40</b>.</li>
   <li><b>False alarms.</b> On simulated honest funds, each check flags on average only <b>2.5-4.2%</b> by mistake at
   the 5% level, below the 5% a fair test allows.</li>
   <li><b>Negative control.</b> On 40 liquid, daily-priced US funds, only <b>3 (7.5%: EMB, FDGRX, TLT)</b> get a
   “Worth a look” and <b>none</b> is “Unusual”. Run through exactly the same pipeline, honest simulated funds reach
   “Unusual” 2% of the time (300 funds). This is the right answer for funds priced every day by the market.</li>
   <li><b>ESG module, 8 real funds.</b> Four keep their promises on all the data that apply (ESGV, SUSA, SUSW.L,
   V3AB.L). Four raise one question each: SPYX, CRBN, EFIV and ESGU.</li>
  </ul></div>
 </div>
 <div class="case mt"><h3>Case study: SPDR S&amp;P 500 Fossil Fuel Reserves Free ETF (SPYX)</h3>
  <div class="grid">
   <div><blockquote>“eliminating companies that own fossil fuel reserves from the S&amp;P 500”</blockquote>
    <ul>
     <li><b>Promise vs behaviour</b> {st("Nothing unusual")} Over 129 months, SPYX's return over SPY falls when energy
     rises (slope −0.027, t = −7.5): the behaviour of a portfolio holding <b>2.6% less energy</b> (range 2.0-3.3%).
     The holdings files give <b>2.4%</b> (1.2% against 3.6%). Two independent methods are consistent.</li>
     <li><b>Promise vs portfolio</b> {st("Nothing unusual")} The energy companies still held (refiners, pipelines)
     and 12 coal-burning utilities (0.60%) do not own fossil-fuel reserves, so they are consistent with the promise.
     FundTruth shows them for information.</li>
     <li><b>Label vs index</b> {st("Worth a look")} R² 99.3% with SPY, tracking error 1.25%, active share 3%. The
     fee is 0.20% against 0.0945%: <b>£11 a year per £10,000</b>, about <b>£632 over 20 years</b>.</li>
    </ul></div>
   <div>{ex("SPYX: two methods are consistent")}{chart_spyx()}
    <div class="ask"><p class="lab">The question FundTruth sends</p><p style="margin:0"><b>Label vs index.</b> Which
    holdings does the ESG screen actually remove, what share of SPY is that, and how is the extra fee justified?</p></div>
    <p class="verdict">The promise is kept. What needs explaining is the price of the label, for a portfolio 97%
    identical to SPY.</p>
    <p class="note">Returns themselves: {st("Nothing unusual")} on all integrity checks. FundTruth separates “the
    numbers are fine” from “something needs explaining”, and raises no question where the promise is kept.</p>
   </div>
  </div>
 </div>
 <div class="g c57 mt">
  <div class="mod">{ex("Where module 1 earns its keep")}<figure><img src="assets/property_example.png"></figure>
   <p class="cap">A property fund valued by appraisal (simulated example, uploaded through “Check your own fund”).
   Price regularity reads <b>Unusual</b>, and the question is about the valuation policy.</p></div>
  <div class="mod"><h3>What the evidence says, honestly</h3><ul>
   <li>On liquid funds module 1 finds little, as it should. Its value is in <b>illiquid funds</b> (property, private
   credit, hedge funds), whose data users upload.</li>
   <li>Same-type comparison groups are still small (10-16 funds), so today each fund is compared with all 40 funds,
   and the checks are many: a few signals will be chance. That is why the output is a question.</li>
   <li>CRBN's question comes from the absence of a clear energy underweight (t = −1.0): a point to clarify with the
   manager, not a broken promise.</li>
   <li>Holdings files are available for 2 of the 8 ESG funds. The returns-based check covers the rest.</li>
   <li>The factor model uses the three US Fama-French factors (no momentum yet), so the two London-listed ESG funds
   skip the factor checks. Thresholds are provisional and were <b>not tuned</b> on the data used to evaluate them.</li>
  </ul></div>
 </div>
 {footer(4)}
</section>"""

# ================================================================== PAGE 5
COMP = [("ESG ratings", "MSCI, Sustainalytics: a “greenness” score; paid licence", 0, 0, 0),
        ("Research platforms", "Morningstar Direct, FE fundinfo: broad data; costly", 0, .5, 0),
        ("NGO screens", "Fossil Free Funds, FinanceMap: fossil exposure; free", .5, 0, 0),
        ("Manual review", "An analyst reads KIDs and factsheets; hours per fund", .5, 0, .5),
        ("FundTruth", "Verdict, evidence, manager questions; £150 a month", 1, 1, 1)]
comp_rows = "".join(
    f'<tr class="{"us" if n == "FundTruth" else ""}"><td>{n}<span class="gv">{gv}</span></td>'
    f'<td class="h">{hb(a)}</td><td class="h">{hb(b)}</td><td class="h">{hb(c)}</td></tr>'
    for n, gv, a, b, c in COMP)

P5 = f"""
<section class="page">
 {head(5, "Business model and growth", "A subscription model that breaks even in year 2, with a data moat",
 "Prices, volumes and costs are illustrative assumptions. The working product, its validation and the code already exist.")}
 <div class="g c75">
  <div class="mod">{ex("Four plans, one engine", "(B2B SaaS, freemium entry)")}
   <div class="tiers">
    <div style="--c:#C9D6E8"><h4>Free</h4><span class="p">£0</span><span class="u">&nbsp;</span><p class="for">Private
    investors, students</p><p>Watchlist, 3 own-fund checks a month</p></div>
    <div style="--c:#7F9CC8"><h4>Professional</h4><span class="p">£150</span><span class="u">per user a month</span>
    <p class="for">IFAs, fund selectors, wealth managers</p><p>Unlimited checks, both modules, PDF reports, alerts</p></div>
    <div style="--c:{BLUE}"><h4>Compliance</h4><span class="p">£15,000</span><span class="u">a year</span>
    <p class="for">Asset managers' compliance and product teams</p><p>5 users, monitoring of the firm's own fund
    range, audit trail, API</p></div>
    <div style="--c:{NAVY}"><h4>Platform</h4><span class="p">£40,000+</span><span class="u">a year</span>
    <p class="for">Fund platforms, DFMs, research houses</p><p>API and white-label watchlist for the whole shelf</p></div>
   </div></div>
  <div class="mod">{ex("Go-to-market")}
   <div class="gtm">
    <b>Land</b><p>London IFAs and fund selectors via the free tier; every exported PDF carries the brand to the
    manager who receives it.</p>
    <b>Expand</b><p>Compliance teams of UK managers under SDR, then EU managers under ESMA's naming rules.</p>
    <b>Scale</b><p>Platform and API licences, partnerships with compliance consultancies.</p>
   </div></div>
 </div>
 <div class="g c57 mt">
  <div class="mod">{ex("Three-year plan", "(£, illustrative)")}{chart_plan()}
   <p class="src">Costs: team, data and hosting (mostly people). Year-3 users are 1.6% of the UK's 31,000 advisers.</p></div>
  <div class="mod">{ex("Competitive landscape")}
   <table class="t cmp">
    <tr><th>Alternative, what it gives and what it costs</th><th class="h">Tests own promise</th>
    <th class="h">Checks returns</th><th class="h">Asks the manager</th></tr>
    {comp_rows}
   </table>
   <p class="src">{hb(1)} covered &nbsp;{hb(.5)} partly &nbsp;{hb(0)} not covered. Assessment by the authors.</p>
  </div>
 </div>
 <div class="g c57 mt">
  <div class="mod">{ex("The moat: FundTruth improves with use")}{chart_learning()}
   <p class="src">1,800 simulated funds with a known truth, 96 months each, 20 repetitions. Below 20 funds of a type,
   no p-value can fall under 0.05, so nothing can reach “Unusual”.</p></div>
  <div>
   <div class="mod"><h3>A data flywheel competitors cannot copy</h3>
    <p>Each fund a user checks can join the comparison set as anonymous summary statistics. Today each fund type has
    10-16 funds, so FundTruth compares every fund with all 40 together: the checks work, but a bond fund is also
    measured against equity funds. Every fund users add moves each type towards 40 comparable funds: at 20 the
    smoothing check catches all strong smoothing and 57% of mild cases; from about 40 it levels off at 61-62%, with
    at most 1.5% false alarms. Corrections to promises and managers' answers add labels that no public dataset
    has.</p></div>
   <div class="mod mt">{ex("Roadmap")}
    <div class="vroad">
     <b>Q1 2027</b><p>Pilot with five London advice firms</p>
     <b>Q2 2027</b><p>Automatic holdings for iShares and Vanguard; KID parsing for UK and EU funds</p>
     <b>H2 2027</b><p>Illiquid-fund data partnerships (property, private credit); momentum and non-US factors;
     Bloomberg connector</p>
     <b>2028</b><p>EU launch and platform API</p>
    </div></div>
  </div>
 </div>
 <div class="mod thin mt"><h3>References and sources <span class="light">Code and data:</span>
  {LINK}</h3>
 <div class="refs">
  <div>
   <p>Berg, F., Kölbel, J. &amp; Rigobon, R. (2022). Aggregate confusion: the divergence of ESG ratings. <i>Review of Finance</i>, 26(6).</p>
   <p>Bollen, N. &amp; Pool, V. (2009). Do hedge fund managers misreport returns? Evidence from the pooled distribution. <i>Journal of Finance</i>, 64(5).</p>
   <p>Cremers, M. &amp; Petajisto, A. (2009). How active is your fund manager? A new measure that predicts performance. <i>Review of Financial Studies</i>, 22(9).</p>
   <p>Fama, E. &amp; French, K. (1993). Common risk factors in the returns on stocks and bonds. <i>Journal of Financial Economics</i>, 33(1).</p>
   <p>Getmansky, M., Lo, A. &amp; Makarov, I. (2004). An econometric model of serial correlation and illiquidity in hedge fund returns. <i>Journal of Financial Economics</i>, 74(3).</p>
   <p>Liu, F., Ting, K. &amp; Zhou, Z. (2008). Isolation forest. <i>IEEE International Conference on Data Mining</i>.</p>
   <p>Newey, W. &amp; West, K. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <i>Econometrica</i>, 55(3).</p>
  </div>
  <div>
   <p>ESMA (2024). Guidelines on funds' names using ESG or sustainability-related terms.</p>
   <p>FCA (2023). PS23/16 Sustainability Disclosure Requirements and investment labels. FCA (2024). FG24/3 Anti-greenwashing rule guidance. FCA (2025). Understanding the financial advice market (survey).</p>
   <p>Investment Association (2026). UK investment management industry assets under management, 2025.</p>
   <p>Morningstar Sustainalytics (2026). Global Sustainable Fund Flows: Q1 2026.</p>
   <p>SEC (2023). DWS to pay $25 million for anti-money laundering violations and misstatements regarding ESG investments, 25 September 2023.</p>
   <p>Data: Yahoo Finance; Kenneth R. French Data Library; State Street SPDR holdings files and issuer fund pages (State Street, iShares, Vanguard); Norges Bank Investment Management exclusion list via OpenSanctions. Data to 30 September 2026.</p>
  </div>
 </div></div>
 {footer(5, "A status is a reason to ask a question, never proof of misreporting or greenwashing.")}
</section>"""

HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FundTruth: business proposal</title>
<style>{CSS}</style></head><body>{P1}{P2}{P3}{P4}{P5}</body></html>"""
open(os.path.join(HERE, "report.html"), "w").write(HTML)
print("ok")
