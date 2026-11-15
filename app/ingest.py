"""Strict ingestion of a Bloomberg-style folder. Never corrects anything silently.

    python ingest.py bbg/ [--out results/bbg_run] [--anonymise] [--max-gaps 0]

(`python -m fundtruth.ingest` is not possible: the frozen engine file is named fundtruth.py,
so a package with that name cannot exist.)

Folder: returns.csv  meta.csv  [benchmarks.csv]  field_log.csv
  returns.csv    date, fund_id, value, kind (return | nav)
  meta.csv       required: fund_id, region, currency, asset_class
                 wanted: pricing, benchmark_id, expense_ratio, inception, aum, domicile,
                         is_dead, last_date; optional: category_label
  benchmarks.csv date, benchmark_id, value, kind
  field_log.csv  field, retrieval_date, universe

STOPS with IngestError (exit code 2) on file-level problems: missing files or columns,
duplicated rows, dates that are not calendar month-ends, wrong units (percentages),
unrecognised or empty required metadata (including an undeclared currency), fund_ids that are
not anonymised codes (F001...) unless --anonymise.
EXCLUDES a fund (each exclusion listed with its reason in data_quality_report.txt) when:
fewer than 36 months, more than --max-gaps missing months, repeated zero returns (a run of
3 or more exact zeros, or more than 10% zeros: a stale-price pattern, PROVISIONAL rule),
constant series.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fund_profile import ASSET_CLASSES, PRICING, REGIONS, _ALIASES
from validation import MIN_OBS, valida_dati

REQ = {"returns.csv": ["date", "fund_id", "value", "kind"],
       "meta.csv": ["fund_id", "region", "currency", "asset_class"],
       "benchmarks.csv": ["date", "benchmark_id", "value", "kind"],
       "field_log.csv": ["field", "retrieval_date", "universe"]}
ANON = re.compile(r"^F\d{3,}$")
ZERO_RUN, ZERO_SHARE = 3, 0.10            # PROVISIONAL stale-price rule
MAX_LIST = 10


class IngestError(Exception):
    """A problem that stops the run, with a readable message."""


@dataclass
class Ingested:
    returns: dict = field(default_factory=dict)        # fund_id -> clean monthly Series
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmarks: dict = field(default_factory=dict)
    field_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["fund_id", "reason"]))
    notes: list = field(default_factory=list)
    id_map: pd.DataFrame | None = None                  # PRIVATE: never written to shareable outputs
    checksums: dict = field(default_factory=dict)
    received: str = ""


def _fmt(items):
    items = [str(i) for i in items]
    return ", ".join(items[:MAX_LIST]) + (f" (+{len(items) - MAX_LIST} more)" if len(items) > MAX_LIST else "")


def _read_csv(folder, name, required=True):
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        if required:
            raise IngestError(f"{name}: file not found in {folder}/.")
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]
    miss = [c for c in REQ[name] if c not in df.columns]
    if miss:
        raise IngestError(f"{name}: required column(s) missing: {miss}. Found: {list(df.columns)}.")
    return df


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]


def _long_to_series(df, idcol, what):
    """Validate a long table and return {id: monthly return Series}, excluded list.
    Hard stops: duplicates, unparseable dates, non month-end dates, bad kind, non-numeric
    values, wrong units, non-positive NAV."""
    df = df.copy()
    df[idcol] = df[idcol].str.strip()
    if (df[idcol] == "").any():
        raise IngestError(f"{what}: empty {idcol} found.")
    bad = sorted(set(df["kind"].str.strip().str.lower()) - {"return", "nav"})
    if bad:
        raise IngestError(f"{what}: kind must be 'return' or 'nav'; found {bad}.")
    df["kind"] = df["kind"].str.strip().str.lower()
    try:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
    except (ValueError, TypeError) as e:
        raise IngestError(f"{what}: unparseable date ({e}).") from None
    val = pd.to_numeric(df["value"].str.strip().replace("", np.nan), errors="coerce")
    nonnum = val.isna() & (df["value"].str.strip() != "") & (df["value"].str.strip().str.lower() != "nan")
    if nonnum.any():
        raise IngestError(f"{what}: non-numeric value in row(s) {_fmt(list(df.index[nonnum] + 2))}.")
    df["value"] = val
    dup = df[df.duplicated([idcol, "date"], keep=False)]
    if len(dup):
        raise IngestError(f"{what}: duplicated (id, date) rows for {_fmt(sorted(dup[idcol].unique()))}. "
                          "Share classes need distinct fund_id; fix the file, nothing is removed silently.")
    notme = df[~df["date"].dt.is_month_end | (df["date"] != df["date"].dt.normalize())]
    if len(notme):
        raise IngestError(f"{what}: dates that are not calendar month-ends for {_fmt(sorted(notme[idcol].unique()))} "
                          f"(example {notme['date'].iloc[0].date()}). Monthly month-end data are required "
                          "(Bloomberg: Per=M, Days=A, Fill=P).")
    mixed = df.groupby(idcol)["kind"].nunique()
    if (mixed > 1).any():
        raise IngestError(f"{what}: both 'return' and 'nav' rows for {_fmt(list(mixed[mixed > 1].index))}.")
    series, excluded, unit_bad = {}, [], []
    for fid, g in df.groupby(idcol, sort=True):
        g = g.dropna(subset=["value"]).sort_values("date")
        s = g.set_index("date")["value"]
        gaps_nav = 0
        if g["kind"].iloc[0] == "nav":
            if (s <= 0).any():
                raise IngestError(f"{what}: non-positive NAV for {fid}.")
            full = pd.date_range(s.index[0], s.index[-1], freq="ME")
            s = s.reindex(full)
            gaps_nav = int(s.isna().sum())
            s = (s / s.shift(1) - 1.0).iloc[1:]
        s.index.name = None
        series[fid] = (s.rename(fid), gaps_nav)
        x = s.dropna().to_numpy(dtype=float)
        if len(x):
            try:
                valida_dati(s.rename(fid))
            except ValueError as e:
                msg = str(e)
                if "percent" in msg.lower() or "outside" in msg.lower():
                    unit_bad.append(fid)
                else:
                    raise IngestError(f"{what}: {msg}") from None
    if unit_bad:
        raise IngestError(f"{what}: wrong units for {_fmt(unit_bad)}: values look like percentages (or are "
                          "implausible). Returns must be monthly DECIMAL fractions (0.012 = 1.2%); "
                          "nothing was converted.")
    return series


def _norm_meta(meta):
    meta = meta.copy()
    meta["fund_id"] = meta["fund_id"].str.strip()
    dup = meta["fund_id"][meta["fund_id"].duplicated()].unique()
    if len(dup):
        raise IngestError(f"meta.csv: duplicated fund_id {_fmt(dup)}.")
    for c in ("region", "asset_class", "currency"):
        blank = meta.loc[meta[c].str.strip() == "", "fund_id"]
        if len(blank):
            extra = (" A currency must be declared for every fund (mixed currencies are allowed, "
                     "undeclared ones are not)." if c == "currency" else "")
            raise IngestError(f"meta.csv: empty {c} for {_fmt(blank)}.{extra}")
    for c, field_name in (("region", "region"), ("asset_class", "asset_class"), ("pricing", "pricing")):
        if c not in meta.columns:
            meta[c] = ""
            continue
        mapped = meta[c].map(lambda v: _ALIASES[field_name].get(v.strip().lower(), None) if v.strip() else "")
        bad = meta.loc[mapped.isna(), ["fund_id", c]]
        if len(bad):
            raise IngestError(f"meta.csv: unrecognised {c} value(s): "
                              f"{_fmt([f'{a}={b!r}' for a, b in bad.itertuples(index=False)])}.")
        meta[c] = mapped
    meta["currency"] = meta["currency"].str.strip().str.upper()
    for c in ("benchmark_id", "inception", "expense_ratio", "aum", "domicile", "is_dead", "last_date", "category_label"):
        if c not in meta.columns:
            meta[c] = ""
    er = pd.to_numeric(meta["expense_ratio"].replace("", np.nan), errors="coerce")
    if (er > 0.2).any():
        raise IngestError("meta.csv: expense_ratio above 0.2 for "
                          f"{_fmt(meta.loc[er > 0.2, 'fund_id'])}: looks like percent; use decimals (0.0075).")
    meta["expense_ratio"] = er
    meta["is_dead"] = meta["is_dead"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
    return meta.set_index("fund_id")


def ingest(folder, anonymise=False, max_gaps=0, min_months=MIN_OBS):
    ing = Ingested()
    ret = _read_csv(folder, "returns.csv")
    meta_raw = _read_csv(folder, "meta.csv")
    bm = _read_csv(folder, "benchmarks.csv", required=False)
    flog = _read_csv(folder, "field_log.csv")
    for n in ("returns.csv", "meta.csv", "benchmarks.csv", "field_log.csv"):
        if os.path.exists(os.path.join(folder, n)):
            ing.checksums[n] = _sha(os.path.join(folder, n))
    ing.field_log = flog[REQ["field_log.csv"]]
    ing.received = str(ing.field_log["retrieval_date"].max())
    meta = _norm_meta(meta_raw)
    ser = _long_to_series(ret, "fund_id", "returns.csv")
    nometa = sorted(set(ser) - set(meta.index))
    if nometa:
        raise IngestError(f"returns.csv: fund_id without a meta.csv row: {_fmt(nometa)}.")
    ids = sorted(ser)
    nonanon = [i for i in ids if not ANON.match(i)]
    if nonanon:
        if not anonymise:
            raise IngestError(f"fund_id not anonymised for {_fmt(nonanon)}: use codes F001, F002, ... or run "
                              "with --anonymise (the name mapping is then kept in private/, never in outputs).")
        mp = {old: f"F{i + 1:03d}" for i, old in enumerate(ids)}
        ing.id_map = pd.DataFrame({"original_id": list(mp), "code": list(mp.values())})
        ser = {mp[k]: (v[0].rename(mp[k]), v[1]) for k, v in ser.items()}
        meta = meta.rename(index=mp)
    excl = []
    for fid, (s, gaps_nav) in ser.items():
        x = s.dropna()
        full = pd.date_range(x.index[0], x.index[-1], freq="ME") if len(x) else []
        gaps = max(len(full) - len(x), 0) if len(x) else 0
        zeros = (x == 0).to_numpy()
        run = best = 0
        for z in zeros:
            run = run + 1 if z else 0
            best = max(best, run)
        reason = None
        if len(x) < min_months:
            reason = f"short history: {len(x)} months (< {min_months})"
        elif gaps > max_gaps:
            reason = f"{gaps} missing month(s) inside the span (allowed: {max_gaps}); nothing was interpolated"
        elif best >= ZERO_RUN or zeros.mean() > ZERO_SHARE:
            reason = (f"repeated zero returns: longest run {best}, {zeros.mean():.0%} zeros "
                      f"(rule: run >= {ZERO_RUN} or share > {ZERO_SHARE:.0%}; stale-price pattern, provisional)")
        elif float(x.std(ddof=0)) < 1e-12:
            reason = "constant series"
        if reason:
            excl.append(dict(fund_id=fid, reason=reason))
        else:
            ing.returns[fid] = x.rename(fid)
    ing.excluded = pd.DataFrame(excl, columns=["fund_id", "reason"])
    ing.meta = meta.loc[[i for i in ing.returns]]
    if bm is not None:
        bs = _long_to_series(bm, "benchmark_id", "benchmarks.csv")
        ing.benchmarks = {k: v[0].dropna() for k, v in bs.items()}
    used = set(ing.meta["benchmark_id"].replace("", np.nan).dropna())
    missing_b = sorted(used - set(ing.benchmarks))
    if missing_b:
        ing.notes.append(f"benchmark_id declared in meta but without series: {_fmt(missing_b)}; "
                         "those funds are analysed without benchmark.")
    ing.notes.append(f"{int(ing.meta['is_dead'].sum())} dead/merged fund(s) kept in the universe."
                     if ing.meta["is_dead"].any() else
                     "No dead/merged fund in the universe: survivorship bias is likely (only funds that "
                     "still exist are observed).")
    cur = sorted(ing.meta["currency"].unique())
    if len(cur) > 1:
        ing.notes.append(f"Mixed fund currencies, all declared: {cur}. Factors are USD-based: warnings apply.")
    return ing


def quality_report(ing, extra=None):
    """Text of data_quality_report.txt: anonymised ids only, every exclusion with its reason."""
    L = ["FundTruth data-quality report",
         f"Data received/retrieved (field_log): {ing.received}",
         f"Funds in universe after checks: {len(ing.returns)}; excluded: {len(ing.excluded)}",
         "Input file fingerprints (sha256, first 16 hex): " + ", ".join(f"{k}={v}" for k, v in ing.checksums.items()),
         "", "Exclusions (one by one, nothing corrected silently):"]
    L += [f"  {r.fund_id}: {r.reason}" for r in ing.excluded.itertuples()] or ["  none"]
    L += ["", "Notes:"] + [f"  {n}" for n in ing.notes]
    if len(ing.meta):
        L += ["", "Universe by asset class x region (after exclusions):"]
        g = ing.meta.groupby(["asset_class", "region"]).size()
        L += [f"  {a} x {r}: {n}" for (a, r), n in g.items()]
        L += ["", "Pricing types: " + ", ".join(f"{k or 'unknown'}={v}" for k, v in ing.meta["pricing"].value_counts().items())]
    L += ["", "Sourcing (field_log.csv):"] + [f"  {r.field}; retrieved {r.retrieval_date}; {r.universe}"
                                              for r in ing.field_log.itertuples()]
    L += list(extra or [])
    L += ["", "A flag is a signal to investigate, never proof. Cite as 'Bloomberg Terminal, retrieved <date>'."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="results/bbg_run")
    ap.add_argument("--anonymise", action="store_true")
    ap.add_argument("--max-gaps", type=int, default=0)
    ap.add_argument("--private-dir", default="private")
    a = ap.parse_args(argv)
    try:
        ing = ingest(a.folder, a.anonymise, a.max_gaps)
    except IngestError as e:
        print(f"INGEST STOPPED: {e}", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "data_quality_report.txt"), "w").write(quality_report(ing))
    if ing.id_map is not None:
        os.makedirs(a.private_dir, exist_ok=True)
        ing.id_map.to_csv(os.path.join(a.private_dir, "id_map.csv"), index=False)
        print(f"id mapping written to {a.private_dir}/id_map.csv (keep it local, never share it)")
    print(f"ingest OK: {len(ing.returns)} funds, {len(ing.excluded)} excluded; report in {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
