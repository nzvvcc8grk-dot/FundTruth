"""Create bloomberg_template.xlsx: ready-to-paste BDP/BDH formulas for the Bloomberg Excel add-in.

    python make_bloomberg_template.py [--slots 150]

NOT TESTED against a Bloomberg Terminal (none available where this was written). Every field
mnemonic below is a CANDIDATE: confirm each with FLDS on the Terminal and edit the cells in the
'README' sheet; all formulas read the mnemonics from there, so one edit fixes the whole workbook.
Formulas are written as plain text; they calculate only in Excel with the Bloomberg add-in.
"""
import argparse

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as L

CAND = "candidate, to confirm with FLDS"
FIELDS = [  # label, cell, default mnemonic, what it is
    ("NAV field (history)", "B8", "FUND_NET_ASSET_VAL", "monthly NAV per share"),
    ("Total-return index field (history, alternative)", "B9", "TOT_RETURN_INDEX_GROSS_DVDS", "total return index; use if NAV ignores distributions"),
    ("Currency", "B10", "CRNCY", "currency of the share class"),
    ("Expense ratio", "B11", "FUND_EXPENSE_RATIO", "reported as PERCENT (0.75 = 0.75%); converter divides by 100"),
    ("Inception date", "B12", "FUND_INCEPT_DT", "inception"),
    ("Category label", "B13", "FUND_ASSET_CLASS_FOCUS", "free-text style/category (used only for proposals)"),
    ("Geographic focus", "B14", "FUND_GEO_FOCUS", "helps to set region"),
    ("Domicile", "B15", "FUND_DOMICILE", "domicile"),
    ("Fund type", "B16", "FUND_TYP", "open-end / ETF / closed-end"),
    ("Active/dead status", "B17", "FUND_ACTIVE_STATUS", "to detect dead or merged funds (field existence unknown)"),
]


def build(path, slots):
    wb = Workbook()
    r = wb.active
    r.title = "README"
    bold = Font(bold=True)
    r["A1"] = "FundTruth Bloomberg extraction workbook (UNTESTED; all mnemonics are candidates)"; r["A1"].font = bold
    r["A3"] = "Retrieval date (type it, yyyy-mm-dd)"; r["B3"] = ""
    r["A4"] = "Start date"; r["B4"] = "2015-12-31"
    r["A5"] = "End date"; r["B5"] = "2026-10-31"
    r["A6"] = "Series sheet to convert"; r["B6"] = "NAV"
    r["A7"] = "Field (edit after FLDS)"; r["B7"] = "Mnemonic"; r["C7"] = "Meaning"; r["D7"] = "Status"
    for lab, cell, mn, meaning in FIELDS:
        row = int(cell[1:])
        r[f"A{row}"], r[f"B{row}"], r[f"C{row}"], r[f"D{row}"] = lab, mn, meaning, CAND
        r[f"B{row}"].fill = PatternFill("solid", fgColor="FFF2CC")
    r["A19"] = ("Steps: 1) list tickers in Universe!A (e.g. 'XXXXXX US Equity' as shown by the Terminal), fill region, asset_class, "
                "pricing, benchmark ticker. 2) Let the formulas calculate. 3) Select all, copy, paste special VALUES (the file must not "
                "depend on the add-in). 4) Save. 5) python bbg_from_excel.py this.xlsx --out bbg/  then  python ingest.py bbg/")
    r["A20"] = "Allowed values: region = US|Europe|Global|EM|Other; asset_class = equity|bond|multi-asset|real_assets|other; pricing = exchange_traded|nav_listed_holdings|appraised_nav."
    r["A21"] = "Never commit this workbook or the bbg/ folder (Bloomberg licence). Share only the aggregated results of run_all.py."
    r.column_dimensions["A"].width = 48; r.column_dimensions["B"].width = 34; r.column_dimensions["C"].width = 60

    u = wb.create_sheet("Universe")
    heads = ["ticker", "region", "asset_class", "pricing", "benchmark_ticker", "is_dead (type 1 if dead/merged)",
             "currency", "expense_ratio_percent", "inception", "category_label", "geo_focus", "domicile", "fund_type", "status"]
    for j, h in enumerate(heads, 1):
        u.cell(1, j, h).font = bold
    bdp = {7: "B10", 8: "B11", 9: "B12", 10: "B13", 11: "B14", 12: "B15", 13: "B16", 14: "B17"}
    for i in range(2, slots + 2):
        for col, cell in bdp.items():
            u.cell(i, col, f'=IF($A{i}="","",BDP($A{i},README!${cell[0]}${cell[1:]}))')
    for sheet, fieldcell, src in (("NAV", "B8", "Universe"), ("TRI", "B9", "Universe"), ("Benchmarks", "B9", "Universe")):
        ws = wb.create_sheet(sheet)
        ws["A1"] = "Each fund: two columns (date | value). Row 1 of the date column = ticker. BDH monthly, calendar days, NO fill (gaps stay visible)."
        for k in range(slots):
            c = 2 * k + 1
            col = "E" if sheet == "Benchmarks" else "A"
            ws.cell(2, c, f'=IF({src}!${col}{k + 2}="","",{src}!${col}{k + 2})')
            ws.cell(3, c, f'=IF({L(c)}2="","",BDH({L(c)}2,README!${fieldcell[0]}${fieldcell[1:]},README!$B$4,README!$B$5,"Per=M","Days=A"))')
    wb.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=150)
    ap.add_argument("--out", default="bloomberg_template.xlsx")
    a = ap.parse_args()
    build(a.out, a.slots)
    print("written", a.out)
