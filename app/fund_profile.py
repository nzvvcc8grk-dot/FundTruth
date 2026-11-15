"""Fund profile and context selection (factors, peers). NOT part of the score.

The integrity score engine (fundtruth.py) is frozen. This module only
describes the fund, chooses which factor set and which peers fit its context,
and produces visible warnings. Nothing here feeds the score.

Rules: values come from the metadata table when present, else from explicit
user input, else 'unknown' with a visible warning. Nothing is guessed silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REGIONS = ("US", "Europe", "Global", "EM", "Other")
ASSET_CLASSES = ("equity", "bond", "multi-asset", "real_assets", "other")
PRICING = ("exchange_traded", "nav_listed_holdings", "appraised_nav")
STYLES = ("active", "passive")
UNKNOWN = "unknown"
ENGINE_CATEGORIES = ("equity", "bond", "multi-asset")   # what the frozen engine scores
ENGINE_PROXY = {"real_assets": "multi-asset", "other": "multi-asset"}   # nearest engine category
MIN_PEERS = 20

# Kenneth R. French Data Library, monthly 3-factor files. Names of the US,
# Developed and Europe files verified on the library page on 2026-11-13.
# Global and Emerging files are NOT mapped: names/layout not verified.
FACTOR_LIBRARY = {
    "US": dict(label="US Fama/French 3 factors",
               file="F-F_Research_Data_Factors_CSV.zip"),
    "Europe": dict(label="Europe (European) Fama/French 3 factors",
                   file="Europe_3_Factors_CSV.zip"),
    "Global": dict(label="Developed markets Fama/French 3 factors (proxy for Global)",
                   file="Developed_3_Factors_CSV.zip"),
}
FACTOR_CURRENCY = "USD"      # per project specification; to verify on the library page
_ALIASES = {
    "region": {"us": "US", "usa": "US", "united states": "US", "europe": "Europe",
               "eu": "Europe", "global": "Global", "world": "Global", "em": "EM",
               "emerging": "EM", "emerging markets": "EM", "other": "Other"},
    "asset_class": {"equity": "equity", "equities": "equity", "bond": "bond",
                    "bonds": "bond", "fixed income": "bond", "multi-asset": "multi-asset",
                    "multi_asset": "multi-asset", "multiasset": "multi-asset",
                    "real_assets": "real_assets", "real assets": "real_assets",
                    "other": "other"},
    "pricing": {"exchange_traded": "exchange_traded", "etf": "exchange_traded",
                "market_priced": "exchange_traded",          # legacy alias
                "market": "exchange_traded",
                "nav_listed_holdings": "nav_listed_holdings", "nav": "nav_listed_holdings",
                "appraised_nav": "appraised_nav", "appraised": "appraised_nav"},
    "style": {"active": "active", "passive": "passive", "index": "passive"},
}


def _norm(field_name, value, warns):
    """Normalise one categorical value; unknown/unrecognised -> 'unknown' + warning."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return UNKNOWN
    key = str(value).strip().lower()
    out = _ALIASES[field_name].get(key)
    if out is None:
        warns.append(f"{field_name} '{value}' not recognised; treated as unknown.")
        return UNKNOWN
    return out


def _num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(x) else x


@dataclass
class FundProfile:
    fund_id: str
    region: str = UNKNOWN
    currency: str = UNKNOWN
    asset_class: str = UNKNOWN
    pricing: str = UNKNOWN
    style: str = UNKNOWN
    benchmark: str | None = None
    inception: str | None = None
    expense_ratio: float | None = None
    aum: float | None = None
    data_warnings: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)   # field -> 'meta' | 'user' | 'unknown'

    @property
    def scoring_supported(self):
        """The frozen engine knows equity, bond and multi-asset. Real assets and 'other'
        (property, private credit, ...) are scored through the multi-asset engine category
        with a visible note (see engine_category); an unknown asset class is not scored."""
        return self.asset_class in ENGINE_CATEGORIES or self.asset_class in ENGINE_PROXY

    @property
    def engine_category(self):
        return self.asset_class if self.asset_class in ENGINE_CATEGORIES else ENGINE_PROXY.get(self.asset_class)


def build_profile(fund_id, meta=None, user_input=None):
    """Build a FundProfile. Priority per field: metadata table, then explicit
    user input, else 'unknown' with a visible warning."""
    meta = {} if meta is None else dict(meta)
    user = {} if user_input is None else dict(user_input)
    warns, prov, vals = [], {}, {}
    for f in ("region", "currency", "asset_class", "pricing", "style"):
        src, raw = UNKNOWN, None
        for name, d in (("meta", meta), ("user", user)):
            v = d.get(f)
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip():
                src, raw = name, v
                break
        if f == "currency":
            val = str(raw).strip().upper() if raw is not None else UNKNOWN
        else:
            val = _norm(f, raw, warns) if raw is not None else UNKNOWN
        prov[f], vals[f] = (src if val != UNKNOWN else "unknown"), val
    for f in ("benchmark", "inception"):
        v = meta.get(f + ("_id" if f == "benchmark" else "")) or user.get(f)
        vals[f] = None if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "" else str(v)
    p = FundProfile(fund_id=str(fund_id), benchmark=vals["benchmark"], inception=vals["inception"],
                    expense_ratio=_num(meta.get("expense_ratio", user.get("expense_ratio"))),
                    aum=_num(meta.get("aum", user.get("aum"))),
                    region=vals["region"], currency=vals["currency"],
                    asset_class=vals["asset_class"], pricing=vals["pricing"],
                    style=vals["style"], provenance=prov)
    for f in ("region", "currency", "asset_class", "pricing", "style"):
        if getattr(p, f) == UNKNOWN:
            warns.append(f"{f} unknown: not guessed; results that depend on it are "
                         "shown as generic and reliability is reduced.")
    p.data_warnings = warns
    return p


def profile_from_legacy_meta(row, fund_id):
    """Profile for the 40 US-listed demo funds (data/funds_meta.csv has category,
    style, type). Region and currency are NOT fields of that table: they are set
    to US / USD because the tickers are US-listed (explicit note below); pricing
    is PROPOSED from the fund type (ETF -> exchange_traded, mutual fund ->
    nav_listed_holdings) and must be confirmed by the user."""
    meta = dict(asset_class=row.get("category"), style=row.get("style"),
                region="US", currency="USD")
    kind = str(row.get("type", "")).upper()
    if kind == "ETF":
        meta["pricing"] = "exchange_traded"
    elif kind == "MF":
        meta["pricing"] = "nav_listed_holdings"
    p = build_profile(fund_id, meta)
    p.data_warnings.insert(0, "Region/currency set to US/USD because the ticker is "
                              "US-listed (not a field of the source table); verify.")
    if kind in ("ETF", "MF"):
        p.data_warnings.append("Pricing type proposed from the fund type; please confirm.")
        p.provenance["pricing"] = "proposed"
    return p


_APPRAISED_WORDS = ("property", "real estate", "private", "hedge", "infrastructure", "unlisted")


def propose_pricing(fund_type=None, category_label=None):
    """Proposed pricing level (to be confirmed by the user), or None when it cannot
    be told: ETF -> exchange_traded; labels such as property/private/hedge ->
    appraised_nav; otherwise None (user must say)."""
    if str(fund_type or "").upper() == "ETF":
        return "exchange_traded"
    lab = str(category_label or "").lower()
    if any(w in lab for w in _APPRAISED_WORDS):
        return "appraised_nav"
    if str(fund_type or "").upper() in ("MF", "MUTUAL FUND", "UCITS"):
        return "nav_listed_holdings"
    return None


@dataclass
class FactorSet:
    key: str | None            # 'US' | 'Europe' | 'Global' | None
    label: str
    file: str | None
    currency: str
    warnings: list
    usable: bool


def factor_set_for(profile):
    """Which factor set fits the profile, with visible warnings. A missing set
    means the factor regression is skipped (never silently replaced)."""
    w = []
    r = profile.region
    spec = FACTOR_LIBRARY.get(r)
    if spec is None:
        reason = {"EM": "Emerging-markets file not mapped (layout not verified).",
                  "Other": "No factor set mapped for region 'Other'.",
                  UNKNOWN: "Region unknown."}.get(r, "No factor set mapped.")
        w.append(f"{reason} Factor regression skipped; no factor set substituted.")
        return FactorSet(None, "none", None, FACTOR_CURRENCY, w, False)
    if r == "Global":
        w.append("Global funds use Developed-markets factors as a proxy.")
    if profile.currency == UNKNOWN:
        w.append("Fund currency unknown: cannot check against the factor currency "
                 f"({FACTOR_CURRENCY}).")
    elif profile.currency != FACTOR_CURRENCY:
        w.append(f"Fund currency {profile.currency} differs from the factor currency "
                 f"{FACTOR_CURRENCY}: alpha and betas include currency effects and must "
                 "not be read as skill or style on their own.")
    if r == "US" and profile.currency not in (UNKNOWN, "USD"):
        w.append(f"Currency/region mismatch: region US but currency {profile.currency}.")
    if r == "Europe" and profile.currency == "USD":
        w.append("Region Europe with USD currency: check the share class (possible "
                 "USD class of a European fund).")
    if profile.asset_class == "bond":
        w.append("Bond fund: equity factors explain little of bond returns (weak "
                 "explanatory variables); read R2 and alpha with caution.")
    return FactorSet(r, spec["label"], spec["file"], FACTOR_CURRENCY, w, True)


@dataclass
class PeerSelection:
    peers: pd.DataFrame
    level: str                 # 'asset_class+region' | 'asset_class' | 'none'
    n: int
    note: str                  # to be DISPLAYED to the user
    engine_pooled: bool        # frozen engine pools across categories below MIN_REF


def select_peers(reference, profile, min_peers=MIN_PEERS, exclude=None):
    """Select peers by (asset_class, region); if fewer than `min_peers`, fall back
    to asset_class only and say so. The reference table needs 'category' (the
    asset class) and optionally 'region'; a missing 'region' column means all US
    (v1 reference: 40 US-listed funds). `exclude` drops the fund itself."""
    ref = reference if exclude is None else reference.drop(index=exclude, errors="ignore")
    if not profile.scoring_supported:
        return PeerSelection(ref.iloc[0:0], "none", 0,
                             f"Asset class '{profile.asset_class}' is not scored by the "
                             "integrity engine; diagnostics only.", False)
    ac = ref[ref["category"] == profile.asset_class]
    has_region = "region" in ref.columns
    reg = ac[ac["region"] == profile.region] if has_region else (
        ac if profile.region == "US" else ac.iloc[0:0])
    if len(reg) >= min_peers:
        note = f"Peer set: {profile.asset_class} funds in region {profile.region} (n={len(reg)})."
        return PeerSelection(reg, "asset_class+region", len(reg), note, False)
    note = (f"FALLBACK: only {len(reg)} {profile.asset_class} funds in region {profile.region} "
            f"(< {min_peers}); peers are all {profile.asset_class} funds regardless of region "
            f"(n={len(ac)}); regions are mixed.")
    pooled = len(ac) < min_peers
    if pooled:
        note += (f" Still fewer than {min_peers}: the frozen engine pools across asset classes.")
    return PeerSelection(ac, "asset_class", len(ac), note, pooled)
