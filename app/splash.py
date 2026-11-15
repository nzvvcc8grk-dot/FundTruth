"""Opening screen, shown once per visit before the home page.

One orchestrated sequence on the brand's ink blue: the mark draws its tick, the headline rises line by line,
then a real return series (SPYX) is scanned from left to right and each of the nine checks lights up in its
status colour. CSS only, no JavaScript; everything is static for visitors who ask for reduced motion.
The action buttons are ordinary Streamlit buttons, so they switch page inside the app without a reload.
"""
import os
import re

import numpy as np
import pandas as pd
import streamlit as st

import readings as rd
import ui

HERE = ui.HERE
EXAMPLE = "SPYX"
CHECKS = ["Price regularity", "Style and benchmark", "Return beyond the market", "Shape of risk",
          "Small gains and losses", "Position among peers", "Promise vs portfolio", "Promise vs behaviour",
          "Label vs index"]
DOT = {rd.NOTHING: "#5FB98E", rd.LOOK: "#E3A53C", rd.UNUSUAL: "#EF6B5E", rd.DESC: "#A9B3BF", rd.NA: "#A9B3BF"}
W, H = 1200, 240                       # chart viewBox
T0, SCAN = 1.8, 3.2                    # scan starts at T0 seconds and lasts SCAN seconds


@st.cache_data(show_spinner=False)
def _wordmark():
    """The outlined 'FundTruth' lettering from the logo file, so the splash matches the logo exactly."""
    s = open(os.path.join(HERE, "static", "logo.svg")).read()
    return re.findall(r'<path d="([^"]+)" fill="#14213D"', s)[0]


@st.cache_data(show_spinner=False)
def _example():
    """Growth of the example fund and the status of each check, from the saved results (fast, no screening)."""
    R = pd.read_csv(os.path.join(HERE, "data", "esg_returns.csv"), index_col="date", parse_dates=True)
    g = (1 + R[EXAMPLE].dropna()).cumprod()
    flagged = []
    ov = os.path.join(HERE, "results", "overview.csv")
    if os.path.exists(ov):
        row = pd.read_csv(ov).set_index("Fund").loc[EXAMPLE]
        flagged = [x.strip() for x in str(row["Flagged checks"]).split(";") if x.strip() and x.strip() != "nan"]
    status = [(c, rd.DESC if c == "Shape of risk" else rd.LOOK if c in flagged else rd.NOTHING) for c in CHECKS]
    return g.index[0], g.index[-1], g.values.tolist(), status


def _chart():
    start, end, g, status = _example()
    g = np.asarray(g)
    x = np.linspace(0, W, len(g))
    y = H - 18 - (g - g.min()) / (g.max() - g.min()) * (H - 46)
    line = "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in zip(x, y))
    area = line + f" L{W} {H} L0 {H} Z"
    dots = []
    for k, (name, lv) in enumerate(status):
        f = 0.06 + k * 0.88 / (len(status) - 1)
        yy = float(np.interp(f * W, x, y)) / H * 100
        ring = '<i class="ring"></i>' if lv in (rd.LOOK, rd.UNUSUAL) else ""
        dots.append(f'<span class="dot" style="left:{f*100:.2f}%;top:{yy:.2f}%;--c:{DOT[lv]};'
                    f'animation-delay:{T0 + f*SCAN:.2f}s" data-label="{ui.esc(name)}: {ui.esc(ui.status_label(lv))}">{ring}</span>')
    n_q = sum(lv in (rd.LOOK, rd.UNUSUAL) for _, lv in status)
    tail = ("the amber one raised the question for its manager." if n_q == 1 else
            f"the amber ones raised the {dict([(2, 'two'), (3, 'three')]).get(n_q, n_q)} questions for its manager."
            if n_q else "none raised a question.")
    caption = (f"{ui.esc(ui.fund_name(EXAMPLE))}, {start:%b %Y} to {end:%b %Y}. Each dot is one of the nine checks; "
               + tail)
    return f"""
<div class="sp-chart" aria-hidden="true">
  <div class="sp-plot">
    <svg viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      <defs><linearGradient id="spg" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" stop-color="#9CC3E6" stop-opacity=".18"/><stop offset="1" stop-color="#9CC3E6" stop-opacity="0"/>
      </linearGradient></defs>
      <path d="{area}" fill="url(#spg)"/>
      <path d="{line}" fill="none" stroke="#D5E4F1" stroke-width="2" vector-effect="non-scaling-stroke"
            stroke-linejoin="round"/>
    </svg>
  </div>
  <div class="sp-scan"></div>
  {''.join(dots)}
</div>
<div class="sp-cap">{caption}</div>"""


CSS = """
<style>
/* the opening screen owns the whole window: no sidebar, no header */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stHeader"],
[data-testid="stDecoration"] { display:none !important; }
[data-testid="stApp"] {
  background-color:#14213D;
  background-image:
    radial-gradient(ellipse 70% 50% at 50% -5%, rgba(46,104,156,.55), rgba(20,33,61,0) 70%),
    repeating-linear-gradient(to bottom, rgba(255,255,255,0) 0 47px, rgba(255,255,255,.035) 47px 48px);
}
[data-testid="stMainBlockContainer"] { max-width:1120px; padding:6vh 24px 32px; }

.sp { text-align:center; color:#F4F7F5; font-family:"IBM Plex Sans", sans-serif; }
.sp-lock { display:inline-flex; align-items:center; gap:16px; margin-bottom:4.5vh; }
.sp-lock svg { height:56px; width:auto; overflow:visible; }
.sp-tile { transform-box:fill-box; transform-origin:center; animation:spTile .6s .1s cubic-bezier(.2,.8,.2,1) both; }
.sp-tick { stroke-dasharray:1; animation:spDraw .55s .45s cubic-bezier(.6,0,.3,1) both; }
.sp-rule { transform-box:fill-box; transform-origin:left center; animation:spRule .45s .75s ease-out both; }
.sp-word { animation:spWord 1.2s .9s cubic-bezier(.16,1,.3,1) both; }

.sp .sp-h1 { font-family:"IBM Plex Sans", sans-serif; font-weight:600; color:#FFFFFF !important; letter-spacing:-.022em;
  font-size:clamp(2.3rem, 5.4vw, 4.5rem) !important; line-height:1.06 !important; margin:0 0 20px !important;
  padding:0 !important; scroll-margin:0; }
.sp .sp-h1 .ln { display:block; overflow:hidden; padding-bottom:.06em; }
.sp .sp-h1 .ln > span { display:inline-block; animation:spRise 1.4s cubic-bezier(.16,1,.3,1) both; }
.sp .sp-h1 .ln:nth-child(1) > span { animation-delay:1.3s; }
.sp .sp-h1 .ln:nth-child(2) > span { animation-delay:1.75s; }
.sp-sub { max-width:40em; margin:0 auto !important; text-align:center; font-size:1.12rem; line-height:1.6; color:rgba(244,247,245,.74);
  animation:spFade 1.3s 2.4s ease-out both; }

/* actions: real Streamlit buttons, restyled for the dark ground */
.st-key-sp_actions { margin-top:26px; animation:spUp 1.2s 2.8s cubic-bezier(.16,1,.3,1) both; }
.st-key-sp_actions button { min-height:46px; padding:0 20px; border-radius:8px; font-weight:500;
  transition:background-color .15s, border-color .15s, transform .15s; }
.st-key-sp_actions button p { font-size:1rem; color:inherit; }
.st-key-sp_actions button:active { transform:translateY(1px); }
.st-key-sp_actions .st-key-go_watchlist button { background:#F4F7F5; border:1px solid #F4F7F5; color:#14213D; }
.st-key-sp_actions .st-key-go_watchlist button:hover { background:#FFFFFF; border-color:#FFFFFF; color:#14213D; }
.st-key-sp_actions .st-key-go_check button, .st-key-sp_actions .st-key-go_method button {
  background:rgba(255,255,255,.02); border:1px solid rgba(255,255,255,.34); color:#F4F7F5; }
.st-key-sp_actions .st-key-go_check button:hover, .st-key-sp_actions .st-key-go_method button:hover {
  background:rgba(255,255,255,.08); border-color:rgba(255,255,255,.62); color:#FFFFFF; }
.st-key-sp_actions button:focus-visible { outline:2px solid #9CC3E6; outline-offset:3px; box-shadow:none; }
.st-key-sp_home { margin-top:-2px; animation:spFade 1.2s 3.15s ease-out both; }
.st-key-sp_home button { color:rgba(244,247,245,.62); }
.st-key-sp_home button:hover { color:#FFFFFF; }
.st-key-sp_home button p { font-size:.92rem; text-decoration:underline; text-underline-offset:4px;
  text-decoration-color:rgba(244,247,245,.3); }

/* the scanned return series */
.sp-chart { position:relative; height:clamp(110px, 19vh, 190px); margin:4.5vh 0 0; animation:spFade 1.1s 1.5s ease-out both; }
.sp-plot { position:absolute; inset:0; animation:spReveal __SCAN__s __T0__s linear both; }
.sp-plot svg { width:100%; height:100%; display:block; overflow:visible; }
.sp-scan { position:absolute; top:-14px; bottom:-6px; left:0; width:1px; opacity:0;
  background:linear-gradient(to bottom, rgba(156,195,230,0), #9CC3E6 30%, #9CC3E6 70%, rgba(156,195,230,0));
  box-shadow:0 0 18px 2px rgba(156,195,230,.45); animation:spScan __SCAN__s __T0__s linear both; }
.dot { position:absolute; width:12px; height:12px; margin:-6px 0 0 -6px; border-radius:50%; background:var(--c);
  box-shadow:0 0 0 3px #14213D; animation:spDot .5s cubic-bezier(.3,1.6,.5,1) both; cursor:default; }
.dot .ring { position:absolute; inset:-3px; border-radius:50%; border:2px solid var(--c); opacity:0;
  animation:spRing 1.6s .25s ease-out 2; animation-delay:inherit; }
.dot::after { content:attr(data-label); position:absolute; bottom:20px; left:50%; transform:translateX(-50%);
  white-space:nowrap; font-size:.78rem; color:#14213D; background:#F4F7F5; padding:3px 8px; border-radius:6px;
  opacity:0; pointer-events:none; transition:opacity .15s; }
.dot:hover::after { opacity:1; }
.sp-cap { margin:16px auto 0 !important; text-align:center; max-width:46em; font-size:.82rem; line-height:1.5; color:rgba(244,247,245,.5);
  animation:spFade 1.3s __CAP__s ease-out both; }

@keyframes spTile { from { opacity:0; transform:scale(.72); } }
@keyframes spDraw { from { stroke-dashoffset:1; } to { stroke-dashoffset:0; } }
@keyframes spRule { from { transform:scaleX(0); } }
@keyframes spWord { from { opacity:0; transform:translateX(-10px); } }
@keyframes spRise { from { transform:translateY(105%); } }
@keyframes spFade { from { opacity:0; } }
@keyframes spUp { from { opacity:0; transform:translateY(12px); } }
@keyframes spReveal { from { clip-path:inset(-20px 100% -20px 0); } to { clip-path:inset(-20px 0 -20px 0); } }
@keyframes spScan { 0% { left:0; opacity:0; } 6% { opacity:1; } 94% { opacity:1; } 100% { left:100%; opacity:0; } }
@keyframes spDot { from { opacity:0; transform:scale(0); } }
@keyframes spRing { from { opacity:.9; transform:scale(1); } to { opacity:0; transform:scale(2.6); } }

@media (max-width:640px) {
  [data-testid="stMainBlockContainer"] { padding:5vh 16px 32px; }
  .sp-lock svg { height:46px; }
  .sp-sub { font-size:1rem; }
  .st-key-sp_actions > div { flex:1 1 100% !important; width:100% !important; }
  .st-key-sp_actions button { width:100%; }
}
@media (prefers-reduced-motion: reduce) {
  .sp *, .sp, .st-key-sp_actions, .st-key-sp_home, .sp-chart, .sp-chart *, .sp-cap { animation:none !important; }
}
</style>
"""


def render():
    """Draw the opening screen; the buttons record that the visitor has entered and switch page."""
    box = st.container(key="splash")
    box.markdown(CSS.replace("__SCAN__", f"{SCAN}").replace("__T0__", f"{T0}").replace("__CAP__", f"{T0 + SCAN:.1f}"),
                unsafe_allow_html=True)
    box.markdown(f"""
<div class="sp">
  <div class="sp-lock">
    <svg viewBox="0 0 177 40" role="img" aria-label="FundTruth">
      <rect class="sp-tile" x="0" y="0" width="40" height="40" rx="9" fill="#2A5F91" stroke="rgba(255,255,255,.22)"/>
      <path class="sp-rule" d="M10 28.5 H30" stroke="#FFFFFF" stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>
      <path class="sp-tick" pathLength="1" d="M12.5 17.5 L18 23 L28.5 11.5" fill="none" stroke="#FFFFFF"
            stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
      <path class="sp-word" d="{_wordmark()}" fill="#FFFFFF"/>
    </svg>
  </div>
  <div class="sp-h1" role="heading" aria-level="1"><span class="ln"><span>Funds make promises.</span></span><span class="ln"><span>Check them against the evidence.</span></span></div>
  <div class="sp-sub">FundTruth tests a fund's reported returns and its ESG promises against public data.
  Every flag comes with its evidence and the question to put to the manager.</div>
</div>""", unsafe_allow_html=True)

    target = None
    with box, st.container(key="sp_actions", horizontal=True, horizontal_alignment="center", gap="small"):
        if st.button("Open the watchlist", key="go_watchlist", icon=":material/list_alt:"):
            target = "watchlist"
        if st.button("Check your own fund", key="go_check", icon=":material/upload_file:"):
            target = "check"
        if st.button("How it works", key="go_method", icon=":material/menu_book:"):
            target = "method"
    with box, st.container(key="sp_home", horizontal=True, horizontal_alignment="center"):
        if st.button("Continue to the home page", key="go_home", type="tertiary"):
            target = "home"
    if target:
        st.session_state["entered"] = True
        if target == "home":
            st.rerun()
        ui.go(target)
    box.markdown(_chart(), unsafe_allow_html=True)
    # warm the shared caches while the visitor reads, so the next page opens at once
    ui.context()
    ui.headline_numbers()
