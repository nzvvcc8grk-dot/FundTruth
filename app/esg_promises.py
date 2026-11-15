"""FundTruth module 2, ESG promises: does an ESG fund do what its documents promise?

Module 1 asks "are the reported returns credible?"; this module asks
"is the fund's sustainability promise visible in what it holds, in how it behaves and in what
it costs?". It never uses ESG ratings (paid and inconsistent across providers): it compares
the fund's OWN written promises with verifiable public facts.

Three readings per fund, in the same format as FundTruth's readings (readings.Reading):
  1. Promise vs portfolio  : holdings against the promised exclusions (energy-sector weight,
                             names on the Norges Bank exclusion list).
  2. Promise vs behaviour  : does a "fossil-free" fund still move with oil and gas?
                             OLS of the fund's return over its parent index fund on the energy
                             sector's return over the same parent, Newey-West t-statistic.
                             A fund that holds the parent minus its energy stocks has slope
                             b = -w/(1-w), w = parent's energy weight; so -b/(1-b) is the
                             energy underweight implied by returns alone (no holdings needed).
  3. Label vs index        : how different is the ESG fund from its non-ESG parent, and what
                             does the label cost? R^2, tracking error, active share (when both
                             holdings files exist) and the fee gap per GBP 10,000 per year.
A status is a signal to ask the manager a question, never proof of greenwashing.

Promises come from esg_claims.json (extracted with an LLM from issuer pages).
CLAIM_PROMPT is the extraction prompt used; extract_claims() runs it when an API key is set.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

from readings import DESC, LOOK, NA, NOTHING, UNUSUAL, Reading

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_MONTHS = 36
NEAR_INDEX_R2, NEAR_INDEX_TE = 0.97, 0.02      # "near-index" label: provisional thresholds
LOW_ACTIVE_SHARE = 0.20                        # below this the portfolio is mostly the parent's
ENERGY_TOLERANCE = 0.02                        # energy weight tolerated under a full fossil promise
T_CRIT = 2.0
LEVEL_ORDER = {UNUSUAL: 0, LOOK: 1, NOTHING: 2, DESC: 3, NA: 4}
PROMISE_LABEL = {"full": "excludes oil, gas and coal companies (or owners of their reserves)",
                 "reduced": "promises lower carbon exposure than the market",
                 "partial": "excludes only thermal coal and/or oil sands",
                 "none": "makes no fossil-fuel promise"}

CLAIM_PROMPT = """You read a fund document (KID, factsheet or prospectus extract).
Return ONLY a JSON object with these keys:
  "name": fund name, "index": tracked index or null,
  "exclusions": list of excluded activities exactly as the document names them,
  "fossil_promise": one of "full" (excludes oil, gas and coal producers or owners of reserves),
                    "reduced" (promises lower carbon exposure), "partial" (only thermal coal and/or
                    oil sands), "none",
  "quotes": the sentences (verbatim, max 3) that support the exclusions and the fossil promise,
  "expense_ratio": decimal (0.0020 for 0.20%) or null.
Never infer a promise the text does not make. If unsure, use "none" and quote nothing.
DOCUMENT:
"""


# ---------------------------------------------------------------- inputs
def load_claims(path=None):
    path = path or os.path.join(HERE, "esg_claims.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def extract_claims(document_text, model="claude-sonnet-5-5", client=None):
    """Run CLAIM_PROMPT on a document with the Anthropic API (needs ANTHROPIC_API_KEY).
    The output must be checked by a person before it enters esg_claims.json."""
    if client is None:
        import anthropic  # optional dependency
        client = anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=1200,
                                 messages=[{"role": "user", "content": CLAIM_PROMPT + document_text[:60000]}])
    text = msg.content[0].text
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


_SUFFIX = re.compile(r"\b(incorporated|inc|corporation|corp|company|co|plc|ltd|limited|sa|ag|nv|se|"
                     r"holdings?|group|the|class [a-z]|cl [a-z]|asa|ab|oyj|spa|nyrt|bhd|tbk|pcl)\b")


def norm_name(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = _SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_exclusions(path):
    """Norges Bank exclusion list (OpenSanctions 'targets.simple.csv'): normalised name -> (name, reason)."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    reason_col = next((c for c in df.columns if c.lower() in ("sanctions", "notes", "program_ids")), None)
    out = {}
    for _, r in df.iterrows():
        names = [r.get("name", "")] + [a for a in r.get("aliases", "").split(";") if a]
        for n in names:
            k = norm_name(n)
            if len(k) >= 4:
                out[k] = (r.get("name", n), r.get(reason_col, "") if reason_col else "")
    return out


def load_holdings(ticker, holdings_dir):
    p = os.path.join(holdings_dir, f"{ticker.upper()}.csv")
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p).fillna({"sector": "", "ticker": ""})
    return h if len(h) else None


# Rule-based reading of a pasted fund description, used when no LLM is configured.
# The user always checks and edits the result before the analysis runs.
EXCLUSION_TERMS = {
    "fossil fuels": [r"fossil fuel"],
    "fossil fuel reserves": [r"fossil[- ]fuels? reserves", r"\b(coal|oil|gas) reserves", r"own(s|ing)? (fossil[- ]fuel )?reserves"],
    "oil and gas": [r"\boil\b(?! sands)", r"\bgas\b", r"petroleum"],
    "thermal coal": [r"thermal coal", r"\bcoal\b"], "oil sands": [r"oil sands", r"tar sands"],
    "tobacco": [r"tobacco"], "controversial weapons": [r"controversial weapons", r"cluster munitions",
                                                        r"landmines", r"chemical (and|&) biological"],
    "weapons and firearms": [r"\bweapons\b", r"firearms", r"small arms", r"military"],
    "nuclear weapons": [r"nuclear weapons"], "nuclear power": [r"nuclear power"],
    "UN Global Compact violators": [r"global compact"], "alcohol": [r"alcohol"], "gambling": [r"gambling"],
    "adult entertainment": [r"adult entertainment"], "cannabis": [r"cannabis"]}
_LOW_CARBON = [r"low(er)? carbon", r"carbon (intensity|emissions|footprint)", r"reduced fossil", r"paris[- ]aligned",
               r"climate transition", r"net zero"]


def keyword_claims(text):
    """Pre-fill a claim from pasted text: exclusions, fossil-fuel promise type, supporting quotes."""
    t = str(text)
    low = t.lower()
    found = [k for k, pats in EXCLUSION_TERMS.items() if any(re.search(p, low) for p in pats)]
    if any(k in found for k in ("fossil fuels", "oil and gas")):
        promise = "full"
    elif any(re.search(p, low) for p in _LOW_CARBON):
        promise = "reduced"
    elif any(k in found for k in ("thermal coal", "oil sands")):
        promise = "partial"
    else:
        promise = "none"
    pats = [p for v in EXCLUSION_TERMS.values() for p in v] + _LOW_CARBON + [r"exclu", r"sustainab", r"\besg\b"]
    sents = [x.strip() for x in re.split(r"(?<=[.;!?])\s+|\n+", t) if x.strip()]
    quotes = [x[:300] for x in sents if any(re.search(p, x.lower()) for p in pats)][:3]
    return dict(exclusions=found, fossil_promise=promise, quotes=quotes)


# ---------------------------------------------------------------- statistics
def _align(*series):
    df = pd.concat(series, axis=1, join="inner").dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


def near_index_stats(rf, rp):
    """R^2, tracking error and tracking difference of a fund against its parent (monthly)."""
    f, p = _align(rf, rp)
    act = f - p
    r2 = float(np.corrcoef(f, p)[0, 1] ** 2)
    return dict(months=len(f), first=str(f.index[0].date()), last=str(f.index[-1].date()),
                r2=r2, te=float(act.std(ddof=1) * np.sqrt(12)), td=float(act.mean() * 12))


def fossil_slope(rf, rp, re_):
    """Slope of (fund - parent) on (energy - parent), Newey-West (3 lags) standard error."""
    f, p, e = _align(rf, rp, re_)
    y, x = f - p, e - p
    m = sm.OLS(y.values, sm.add_constant(x.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    b, se = float(m.params[1]), float(m.bse[1])
    implied = lambda s: -s / (1 - s)
    return dict(months=len(y), b=b, se=se, t=b / se if se > 0 else np.nan,
                implied_underweight=implied(b), ci_low=implied(b + 1.96 * se), ci_high=implied(b - 1.96 * se))


def _has_tickers(h):
    t = h["ticker"].astype(str).str.upper().str.strip()
    return (t.str.len().between(1, 6) & ~t.isin(["-", "NAN", ""])).mean() > 0.8


def _key(h, by=None):
    """Match key: ticker when available (by='ticker'), otherwise the normalised company name."""
    by = by or ("ticker" if _has_tickers(h) else "name")
    if by == "ticker":
        t = h["ticker"].astype(str).str.upper().str.strip().str.replace(".", " ", regex=False)
        ok = t.str.len().between(1, 6) & ~t.isin(["-", "NAN", ""])
        return pd.Series(np.where(ok, t, h["name"].map(norm_name)), index=h.index)
    return h["name"].map(norm_name)


def _common(*hs):
    return "ticker" if all(_has_tickers(h) for h in hs) else "name"


def active_share(h1, h2):
    by = _common(h1, h2)
    a = pd.Series(h1["weight"].values, index=_key(h1, by).values).groupby(level=0).sum()
    b = pd.Series(h2["weight"].values, index=_key(h2, by).values).groupby(level=0).sum()
    a, b = a / a.sum(), b / b.sum()
    u = a.index.union(b.index)
    return float(0.5 * (a.reindex(u, fill_value=0) - b.reindex(u, fill_value=0)).abs().sum())


def energy_mask(h, energy_tickers=None):
    """Energy holdings: the issuer's sector label when the file has one, otherwise membership of the
    energy-sector ETF (XLE = S&P 500 Energy). None when neither is available."""
    sec = h["sector"].astype(str).str.strip().str.lower()
    if (sec.str.len() > 1).mean() > 0.5:
        return sec.str.startswith("energy")
    if energy_tickers:
        tick, names = energy_tickers if isinstance(energy_tickers, tuple) else (energy_tickers, set())
        return _key(h, "ticker").isin(tick) if _has_tickers(h) or not names else _key(h, "name").isin(names)
    return None


def energy_weight(h, energy_tickers=None):
    m = energy_mask(h, energy_tickers)
    return None if m is None else float(h.loc[m, "weight"].sum() / h["weight"].sum())


def removed_vs_parent(h, hp):
    """Parent holdings absent from the fund: what the ESG screen leaves out."""
    by = _common(h, hp)
    gone = hp[~_key(hp, by).isin(set(_key(h, by)))]
    return gone.sort_values("weight", ascending=False), float(gone["weight"].sum() / hp["weight"].sum())


# Norges Bank exclusion reasons -> category, and the words in a fund's own exclusions that cover it
_REASONS = [("tobacco", "tobacco"), ("cannabis", "cannabis"), ("coal", "coal"), ("nuclear weapons", "weapons"),
            ("cluster munitions", "weapons"), ("weapons", "weapons"), ("greenhouse", "climate"),
            ("environmental", "environment"), ("human rights", "human rights"), ("war or conflict", "human rights"),
            ("corruption", "corruption"), ("ethical", "ethics")]
_COVERS = {"tobacco": ["tobacco"], "cannabis": ["cannabis"], "coal": ["coal", "fossil", "oil, gas"],
           "weapons": ["weapon", "military", "arms"], "climate": ["fossil", "carbon", "coal"],
           "environment": ["global compact"], "human rights": ["global compact"],
           "corruption": ["global compact"], "ethics": ["global compact"]}


def reason_category(text):
    t = str(text).lower()
    return next((cat for k, cat in _REASONS if k in t), "other")


def promise_covers(category, exclusions):
    words = " | ".join(exclusions).lower()
    return any(w in words for w in _COVERS.get(category, []))


# ---------------------------------------------------------------- readings
HORIZON_YEARS, ASSUMED_RETURN = 20, 0.06     # illustration of the compounded fee gap
MIN_FEE_GAP = 0.0005                           # 0.05% = GBP 5 a year per GBP 10,000: below this the fee gap is immaterial


def reading_label_vs_index(c, ticker, rf, rp, h_fund=None, h_parent=None):
    st = near_index_stats(rf, rp)
    gap = c["expense_ratio"] - c["parent_expense_ratio"]
    as_ = active_share(h_fund, h_parent) if h_fund is not None and h_parent is not None else None
    near = st["r2"] >= NEAR_INDEX_R2 and st["te"] <= NEAR_INDEX_TE
    costly = gap >= MIN_FEE_GAP
    level = LOOK if costly and (near or (as_ is not None and as_ < LOW_ACTIVE_SHARE)) else NOTHING
    found = (f"Over {st['months']} months ({st['first'][:7]} to {st['last'][:7]}) {ticker} explains "
             f"{st['r2']:.1%} of its variation with {c['parent']} (R^2); tracking error against {c['parent']} {st['te']:.2%} a year; "
             f"it returned {st['td']:+.2%} a year versus {c['parent']} after fees.")
    if "not the same index" in str(c.get("parent_name", "")):
        found += (f" Note: {c['parent']} is the closest low-cost index fund, not the ESG fund's own parent index, so this "
                  "measures the cost of the label against the cheapest similar alternative.")
    if as_ is not None:
        found += f" Active share against {c['parent']}: {as_:.0%} (share of the portfolio that differs)."
    cost = (f"Fees: {c['expense_ratio']*100:.4g}% for both funds: the ESG label costs nothing extra." if abs(gap) < 1e-9 else
            f"Fees: {c['expense_ratio']*100:.4g}% against {c['parent_expense_ratio']*100:.4g}% for {c['parent']}, i.e. "
            f"£{abs(gap) * 10000:,.0f} a year {'more' if gap > 0 else 'less'} on £10,000"
            + ("" if gap <= 0 or costly else " (immaterial)") + ".")
    if costly:
        fv = lambda r: 10000 * (1 + r) ** HORIZON_YEARS
        lost = fv(ASSUMED_RETURN) - fv(ASSUMED_RETURN - gap)
        cost += (f" Over {HORIZON_YEARS} years, at an assumed {ASSUMED_RETURN:.0%} a year before the fee gap, "
                 f"£10,000 ends about £{lost:,.0f} lower in the ESG version for the same market return.")
    return Reading(
        key="label_vs_index", title="Label vs index: how different is the ESG version, and what does it cost?",
        level=level, tag="primary",
        measured=f"Monthly returns of {ticker} against its non-ESG parent {c['parent']} ({c['parent_name']}); fees from the issuers' pages.",
        found=found + " " + cost,
        uncertainty="R^2 and tracking error are estimated on monthly prices; active share is a snapshot of one day's holdings.",
        innocent=("Exclusion-based ESG funds are designed to stay close to their parent index: a near-index portfolio "
                  "is what the documents promise, not a breach. The question is whether the extra fee is worth the screens."),
        ask=(f"Which holdings does the ESG screen actually remove, what share of the {c['parent']} portfolio is that, "
             "and how is the extra fee justified?") if level == LOOK else "",
        not_validated="Thresholds (R^2 >= 0.97, tracking error <= 2%, active share < 20%, fee gap >= 0.05%) are provisional conventions, not calibrated.")


def reading_fossil_behaviour(c, ticker, rf, rp, re_, parent_energy=None, fund_energy=None):
    promise = c.get("fossil_promise", "none")
    s = fossil_slope(rf, rp, re_)
    u = s["implied_underweight"]
    found = (f"Over {s['months']} months, {ticker}'s return relative to {c['parent']} moves with the energy sector "
             f"({c['energy_proxy']}) with slope {s['b']:+.3f} (t = {s['t']:.1f}). That is the behaviour of a portfolio "
             f"holding {abs(u):.1%} {'less' if u >= 0 else 'more'} energy than {c['parent']} "
             f"(95% range: {s['ci_low']:+.1%} to {s['ci_high']:+.1%} underweight).")
    if parent_energy is not None:
        found += f" For scale: {c['parent']} holds {parent_energy:.1%} in energy stocks today."
        if fund_energy is not None:
            gap = parent_energy - fund_energy
            agree = s["ci_low"] <= gap <= s["ci_high"]
            found += (f" The holdings files give {fund_energy:.1%} for {ticker}, an underweight of {gap:.1%}: "
                      + ("inside the range estimated from returns, so the two methods are consistent (the returns give the average exposure over the whole period, the holdings one day)." if agree else
                         "outside the range estimated from returns (the portfolio may have changed over time)."))
    if promise in ("full", "reduced"):
        level = NOTHING if s["t"] < -T_CRIT else UNUSUAL if s["t"] > T_CRIT else LOOK
        tag = "primary"
    else:
        level, tag = DESC, "informational"
    return Reading(
        key="fossil_behaviour", title="Promise vs behaviour: does the fund still move with oil and gas?",
        level=level, tag=tag,
        measured=(f"Promise: the fund {PROMISE_LABEL[promise]}. Regression of ({ticker} - {c['parent']}) on "
                  f"({c['energy_proxy']} - {c['parent']}), monthly, Newey-West standard errors."),
        found=found,
        uncertainty="A slope near zero can also mean the test lacks power (short history, small energy weight in the parent).",
        innocent=("Energy services, equipment, refining or pipeline companies may own no reserves; sector-neutral designs "
                  "re-weight the remaining energy companies; a low-carbon index may cut emissions mainly outside energy."),
        ask=("Which energy-sector companies does the fund hold, and how does each satisfy the promise?"
             if level in (LOOK, UNUSUAL) else ""),
        reason=("" if tag == "primary" else f"the fund {PROMISE_LABEL[promise]}, so no oil-and-gas underweight is promised; shown for information."),
        not_validated="No labelled greenwashing cases; the 'implied underweight' assumes the fund is its parent minus energy stocks.")


def reading_promise_vs_holdings(c, ticker, h, exclusions, h_parent=None, energy_tickers=None):
    title = "Promise vs portfolio: does the fund hold what it says it excludes?"
    if h is None:
        return Reading(key="promise_vs_holdings", title=title, level=NA,
                       reason="no holdings file for this fund (the issuer's daily holdings download is needed).")
    promise = c.get("fossil_promise", "none")
    m = energy_mask(h, energy_tickers)
    # A promise about fossil-fuel RESERVES covers companies that own reserves (energy sector, miners),
    # not utilities that burn coal: Norway's "coal or coal-based energy" reason mixes the two.
    reserves_only = any("reserve" in str(e).lower() for e in c["exclusions"])
    hits = []
    for i, r in h.iterrows():
        k = norm_name(r["name"])
        if k in exclusions:
            cat = reason_category(exclusions[k][1])
            covered = promise_covers(cat, c["exclusions"])
            power = bool(covered and cat == "coal" and reserves_only and m is not None and not m.loc[i])
            hits.append(dict(name=r["name"], weight=float(r["weight"]), reason=exclusions[k][1], category=cat,
                             covered=covered and not power, power=power))
    inside = sorted([x for x in hits if x["covered"]], key=lambda x: -x["weight"])
    power = sorted([x for x in hits if x["power"]], key=lambda x: -x["weight"])
    outside = sorted([x for x in hits if not x["covered"] and not x["power"]], key=lambda x: -x["weight"])
    ew = None if m is None else float(h.loc[m, "weight"].sum() / h["weight"].sum())
    level = LOOK if inside or (promise == "full" and ew is not None and ew > ENERGY_TOLERANCE) else NOTHING
    found = f"{len(h)} holdings."
    over = promise == "full" and ew is not None and ew > ENERGY_TOLERANCE
    named = [x["name"] for x in inside[:6]]
    if inside:      # what raised the status comes first, so a shortened text (PDF report) still shows it
        found += (f" {len(inside)} holding(s), {sum(x['weight'] for x in inside):.2%} of the fund, are excluded by Norway's "
                  "sovereign fund for a reason the fund's own promise also covers: "
                  + "; ".join(f"{x['name']} ({x['weight']:.2%}, NBIM: {x['reason'].split(' - ')[-1].lower()})" for x in inside[:6]) + ".")
    if over and m.any():
        named += list(h.loc[m].nlargest(4, "weight")["name"])
    if h_parent is not None:
        gone, gw = removed_vs_parent(h, h_parent)
        found += (f" Compared with {c['parent']}, the fund leaves out {len(gone)} companies worth {gw:.1%} of "
                  f"{c['parent']}; the largest: " + ", ".join(f"{n} ({w:.2%})" for n, w in zip(gone['name'][:5], gone['weight'][:5])) + ".")
    if ew is not None:
        pe = energy_weight(h_parent, energy_tickers) if h_parent is not None else None
        found += f" Energy-sector stocks: {ew:.1%} of the fund" + (f" against {pe:.1%} of {c['parent']}." if pe is not None else ".")
        if m.any():
            top = h.loc[m].nlargest(4, "weight")
            found += " Largest: " + ", ".join(f"{n} ({w:.2%})" for n, w in zip(top["name"], top["weight"])) + "."
    if not exclusions:
        found += " Exclusion list not loaded."
    elif not inside:
        found += " No holding is excluded by Norway's sovereign fund for a reason the fund's promise covers."
    if power:
        found += (f" For information, {len(power)} holding(s) ({sum(x['weight'] for x in power):.2%}) are excluded by Norway "
                  "for coal-based power; the fund's promise covers owners of fossil-fuel reserves, which these utilities "
                  "are not, so they do not breach it: " + ", ".join(x["name"] for x in power[:6]) + ".")
    if outside:
        found += (f" For information, {len(outside)} other holding(s) ({sum(x['weight'] for x in outside):.2%}) are excluded "
                  "by Norway for reasons outside this fund's promise, e.g. "
                  + ", ".join(f"{x['name']} ({x['reason'].split(' - ')[-1].lower()})" for x in outside[:3]) + ".")
    return Reading(
        key="promise_vs_holdings", title=title, level=level, tag="primary",
        measured=(f"Promised exclusions: {', '.join(c['exclusions'])}. The issuer's holdings file (one day) is compared with "
                  f"{c['parent']}'s, with the energy sector (XLE constituents) and with the companies excluded by Norway's "
                  "sovereign wealth fund (Norges Bank list: coal, tobacco, weapons, conduct)."),
        found=found,
        uncertainty="One-day snapshot; names are matched after normalisation and may mis-match; Norway's criteria and thresholds differ from the fund's.",
        innocent=("Norway applies its own thresholds (for example coal-based power, not coal reserves) and conduct judgements; "
                  "a company Norway excludes can pass a fund's narrower screen; an energy-sector company may own no reserves; "
                  "differences with the parent can also come from index rebalancing dates."),
        ask=(f"For {', '.join(named)}: which of the fund's exclusion rules applies to each of them, and why is it still held?"
             if level == LOOK else ""),
        not_validated="Name matching is not audited; Norway's list is a public proxy for the fund's own exclusion data, which is not published.")


# ---------------------------------------------------------------- claim card
@dataclass
class ClaimCard:
    ticker: str
    claim: dict
    readings: list
    metrics: dict = field(default_factory=dict)

    @property
    def headline(self):
        lv = min((r.level for r in self.readings), key=lambda x: LEVEL_ORDER[x])
        return {UNUSUAL: "Promise not visible in the data: ask the manager",
                LOOK: "Worth a question to the manager",
                NOTHING: "Promises consistent with the data that apply"}.get(lv, "Not enough data to check the promises")

    @property
    def questions(self):
        return [r.ask for r in self.readings if r.ask]


def analyse_fund(ticker, returns, claims, holdings_dir=None, exclusions=None, holdings_override=None):
    c = claims["funds"][ticker]
    if ticker not in returns or c["parent"] not in returns:
        raise ValueError(f"{ticker}: returns for the fund or its parent {c['parent']} are missing")
    rf, rp = returns[ticker].dropna(), returns[c["parent"]].dropna()
    n = len(pd.concat([rf, rp], axis=1, join="inner").dropna())
    if n < MIN_MONTHS:
        raise ValueError(f"{ticker}: only {n} common months with {c['parent']} (< {MIN_MONTHS})")
    hdir = holdings_dir or os.path.join(HERE, "data", "holdings")
    h_f, h_p = load_holdings(ticker, hdir), load_holdings(c["parent"], hdir)
    if holdings_override and ticker in holdings_override:          # a user's own holdings file
        h_f = holdings_override[ticker]
    h_e = load_holdings(c["energy_proxy"], hdir)
    if h_e is not None:                                            # energy universe: tickers and names, cash lines removed
        h_e = h_e[~h_e["name"].astype(str).str.lower().str.contains("money market|cash")]
        etk = (set(_key(h_e, "ticker")), set(_key(h_e, "name")))
    else:
        etk = None
    pe = energy_weight(h_p, etk) if h_p is not None else None
    rs = [reading_promise_vs_holdings(c, ticker, h_f, exclusions or {}, h_p, etk)]
    if c["energy_proxy"] in returns:
        fe = energy_weight(h_f, etk) if h_f is not None else None
        rs.append(reading_fossil_behaviour(c, ticker, rf, rp, returns[c["energy_proxy"]].dropna(), pe, fe))
    else:
        rs.append(Reading(key="fossil_behaviour", title="Promise vs behaviour: does the fund still move with oil and gas?",
                          level=NA, reason=f"no returns for the energy proxy {c['energy_proxy']}."))
    rs.append(reading_label_vs_index(c, ticker, rf, rp, h_f, h_p))
    m = dict(near_index_stats(rf, rp))
    if c["energy_proxy"] in returns:
        m.update({f"fossil_{k}": v for k, v in fossil_slope(rf, rp, returns[c["energy_proxy"]].dropna()).items()})
    m.update(fee_gap_gbp_per_10k=(c["expense_ratio"] - c["parent_expense_ratio"]) * 10000,
             energy_weight=(energy_weight(h_f, etk) if h_f is not None else None) or np.nan,
             parent_energy_weight=pe if pe is not None else np.nan,
             removed_weight=removed_vs_parent(h_f, h_p)[1] if h_f is not None and h_p is not None else np.nan,
             active_share=active_share(h_f, h_p) if h_f is not None and h_p is not None else np.nan)
    return ClaimCard(ticker, c, rs, m)


def card_markdown(card):
    c = card.claim
    L = [f"## {card.ticker} - {c['name']}", f"**{card.headline}**", "",
         f"Index: {c['index']}. Compared with: {c['parent']} ({c['parent_name']}).", "",
         "**What the fund promises** (verbatim, source: " + c["source"] + ")"]
    L += [f"> {q}" for q in c["quotes"]] + [""]
    for r in card.readings:
        L.append(f"### {r.title} - {r.level}")
        if r.level == NA:
            L.append(f"Not applicable: {r.reason}")
            continue
        for lab, v in (("Measured", r.measured), ("Found", r.found), ("How sure", r.uncertainty),
                       ("Why it may be innocent", r.innocent), ("Ask the manager", r.ask)):
            if v:
                L.append(f"- *{lab}:* {v}")
        if r.reason:
            L.append(f"- *Note:* {r.reason}")
        L.append("")
    L.append("_A status is a reason to ask a question, never proof of greenwashing._")
    return "\n".join(L)


def run(data_dir=None, out_dir=None):
    data_dir = data_dir or os.path.join(HERE, "data")
    out_dir = out_dir or os.path.join(HERE, "results", "esg")
    os.makedirs(out_dir, exist_ok=True)
    claims = load_claims()
    R = pd.read_csv(os.path.join(data_dir, "esg_returns.csv"), index_col="date", parse_dates=True)
    excl = load_exclusions(os.path.join(data_dir, "nbim_exclusions.csv"))
    cards, rows = [], []
    for t in claims["funds"]:
        try:
            card = analyse_fund(t, R, claims, os.path.join(data_dir, "holdings"), excl)
        except ValueError as e:
            rows.append(dict(fund=t, error=str(e)))
            continue
        cards.append(card)
        rows.append(dict(fund=t, parent=card.claim["parent"], fossil_promise=card.claim["fossil_promise"],
                         headline=card.headline, **{f"status_{r.key}": r.level for r in card.readings}, **card.metrics))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "esg_summary.csv"), index=False)
    with open(os.path.join(out_dir, "claim_cards.md"), "w", encoding="utf-8") as fh:
        fh.write("# FundTruth: ESG claim cards\n\n" + "\n\n".join(card_markdown(c) for c in cards))
    return summary, cards


if __name__ == "__main__":
    s, _ = run()
    cols = [c for c in ["fund", "parent", "fossil_promise", "headline", "r2", "te", "td", "fossil_b", "fossil_t",
                        "fossil_implied_underweight", "fee_gap_gbp_per_10k", "active_share", "energy_weight", "error"] if c in s]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(s[cols].round(4).to_string(index=False))
