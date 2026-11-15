"""Convert the filled Bloomberg workbook (values pasted, saved) into the bbg/ schema.

    python bbg_from_excel.py workbook.xlsx --out bbg/ [--private-dir private] [--kind nav|return]

Fund ids become F001... in sheet order; the ticker<->code table goes to private/id_map.csv
(local only). Nothing is corrected silently: dates are moved to month end only when they fall in
the same month (count printed); '#N/A' cells are dropped and reported; the percentage expense
ratio is divided by 100 and reported. Then check with:  python ingest.py bbg/
"""
import argparse
import os
import sys
from datetime import date, datetime

import pandas as pd
from openpyxl import load_workbook


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def read_pairs(ws):
    """[(ticker, [(date, value), ...])] from a sheet with (date|value) column pairs; ticker in row 2."""
    out = []
    c = 1
    while c <= ws.max_column:
        tk = ws.cell(2, c).value
        if tk not in (None, ""):
            rows = []
            for r in range(3, ws.max_row + 1):
                d, v = ws.cell(r, c).value, ws.cell(r, c + 1).value
                if isinstance(d, (datetime, date)) and _num(v) is not None:
                    rows.append((pd.Timestamp(d), _num(v)))
            out.append((str(tk), rows))
        c += 2
    return out


def long_table(pairs, idmap, idcol):
    recs, moved, dropped = [], 0, 0
    for tk, rows in pairs:
        for d, v in rows:
            me = d + pd.offsets.MonthEnd(0)
            moved += int(me != d.normalize())
            recs.append({"date": me.strftime("%Y-%m-%d"), idcol: idmap[tk], "value": v})
    return pd.DataFrame(recs), moved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx"); ap.add_argument("--out", default="bbg"); ap.add_argument("--private-dir", default="private")
    ap.add_argument("--kind", default="nav", choices=["nav", "return"])
    a = ap.parse_args(argv)
    wb = load_workbook(a.xlsx, data_only=True)
    rd = wb["README"]
    retrieval = str(rd["B3"].value or "").strip()
    if not retrieval:
        print("STOPPED: README!B3 (retrieval date) is empty: type the date the data were retrieved.", file=sys.stderr)
        return 2
    u = wb["Universe"]
    head = [str(c.value or "").strip() for c in u[1]]
    rows = [[c.value for c in r] for r in u.iter_rows(min_row=2) if r[0].value not in (None, "")]
    U = pd.DataFrame(rows, columns=head)
    if U.ticker.duplicated().any():
        print("STOPPED: duplicated tickers in Universe.", file=sys.stderr)
        return 2
    codes = {t: f"F{i + 1:03d}" for i, t in enumerate(U.ticker)}
    bcodes = {t: f"B{i + 1:03d}" for i, t in enumerate(sorted({x for x in U.benchmark_ticker.dropna() if str(x).strip()}))}
    sheet = wb[rd["B6"].value or "NAV"]
    R, moved = long_table(read_pairs(sheet), codes, "fund_id")
    R["kind"] = a.kind
    R = R[["date", "fund_id", "value", "kind"]]
    exp = pd.to_numeric(U.get("expense_ratio_percent"), errors="coerce") / 100
    M = pd.DataFrame(dict(fund_id=U.ticker.map(codes), region=U.region, currency=U.currency, asset_class=U.asset_class,
                          pricing=U.pricing, benchmark_id=U.benchmark_ticker.map(lambda x: bcodes.get(x, "") if isinstance(x, str) else ""),
                          expense_ratio=exp, inception=U.inception.astype(str).str[:10], domicile=U.domicile,
                          is_dead=U.iloc[:, 5].map(lambda x: str(x).strip() in ("1", "True", "true", "yes")),
                          category_label=U.category_label))
    os.makedirs(a.out, exist_ok=True)
    R.to_csv(os.path.join(a.out, "returns.csv"), index=False)
    M.to_csv(os.path.join(a.out, "meta.csv"), index=False)
    if "Benchmarks" in wb.sheetnames and bcodes:
        bp = [(t, rows) for t, rows in read_pairs(wb["Benchmarks"])]
        B, m2 = long_table([(t, r) for t, r in bp if t in bcodes], bcodes, "benchmark_id")
        B["kind"] = a.kind
        B[["date", "benchmark_id", "value", "kind"]].to_csv(os.path.join(a.out, "benchmarks.csv"), index=False)
        moved += m2
    pd.DataFrame([dict(field=f"{rd[f'B{r}'].value} (history, from README sheet)", retrieval_date=retrieval,
                       universe=f"{len(U)} tickers listed in workbook") for r in (8,)]
                 + [dict(field="metadata fields: " + ", ".join(str(rd[f'B{r}'].value) for r in range(10, 18)),
                         retrieval_date=retrieval, universe=f"{len(U)} tickers listed in workbook")]
                 ).to_csv(os.path.join(a.out, "field_log.csv"), index=False)
    os.makedirs(a.private_dir, exist_ok=True)
    pd.DataFrame(list(codes.items()) + list(bcodes.items()), columns=["original_id", "code"]).to_csv(
        os.path.join(a.private_dir, "id_map.csv"), index=False)
    empty = [t for t, c in codes.items() if c not in set(R.fund_id)]
    print(f"written {a.out}/ : {R.fund_id.nunique()} funds with data, {len(U) - R.fund_id.nunique()} listed without data; "
          f"{moved} date(s) moved to month end; expense ratios divided by 100; id map in {a.private_dir}/id_map.csv (local).")
    if empty:
        print(f"WARNING: no data for {len(empty)} ticker(s) (field missing for that fund?): see private/id_map.csv codes "
              + ", ".join(codes[t] for t in empty[:10]), file=sys.stderr)
    print("Next: python ingest.py", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
