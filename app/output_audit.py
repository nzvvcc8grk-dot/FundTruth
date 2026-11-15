"""Confidentiality audit of a results folder: flags any file that still looks like raw series,
non-anonymised fund IDs, or a name mapping. Used by run_all.py and runnable alone:
    python output_audit.py results/bbg_run [--allow-public-ids]
Exit code 0 = clean, 4 = issues found (listed)."""
from __future__ import annotations

import os
import re
import sys

import pandas as pd

ANON = re.compile(r"^F\d{3,}$")
RAW_NAMES = re.compile(r"^(date|dates|value|values|nav|price|prices|close|return|returns|ret|month|datetime|timestamp)$", re.I)
DATEISH = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
FORBIDDEN_FILES = re.compile(r"(id_map|returns|benchmarks|factors_|snapshot|meta\.csv|field_log)", re.I)
ID_COLS = ("fund_id", "fund", "Fund", "ticker")


def audit(folder, allow_public_ids=False, private_ids=()):
    issues = []
    for root, _, files in os.walk(folder):
        for n in files:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, folder)
            if FORBIDDEN_FILES.search(n) and n not in ("data_quality_report.txt",):
                issues.append(f"{rel}: file name suggests raw/input/private data")
            if n.endswith(".csv"):
                try:
                    d = pd.read_csv(p)
                except Exception as e:
                    issues.append(f"{rel}: unreadable ({e})")
                    continue
                bad = [c for c in d.columns if RAW_NAMES.match(str(c).strip())]
                if bad:
                    issues.append(f"{rel}: column(s) {bad} look like raw series fields")
                if sum(bool(DATEISH.match(str(c))) for c in d.columns) >= 3:
                    issues.append(f"{rel}: wide table with date-like headers (raw series)")
                for c in d.columns:
                    if (d[c].dtype == object or pd.api.types.is_string_dtype(d[c])) and len(d) >= 3 and d[c].astype(str).str.match(DATEISH).mean() > .5:
                        issues.append(f"{rel}: column '{c}' holds dates")
                for c in ID_COLS:
                    if c in d.columns:
                        vals = d[c].astype(str)
                        if not (allow_public_ids or rel.startswith('yahoo')) and (~vals.str.match(ANON)).any():
                            ex = vals[~vals.str.match(ANON)].iloc[0]
                            issues.append(f"{rel}: non-anonymous fund id in '{c}' (e.g. {ex})")
                        if vals.duplicated().sum() > 0 and len(d) > 0 and "level" not in d.columns and "scenario" not in d.columns \
                           and "statistic" not in d.columns and "test" not in d.columns and "reading" not in d.columns:
                            issues.append(f"{rel}: repeated fund ids ({int(vals.duplicated().sum())}): possible long series")
            if n.endswith((".txt", ".json", ".csv")) and private_ids:
                txt = open(p, errors="ignore").read()
                hit = [i for i in private_ids if len(str(i)) >= 4 and str(i) in txt]
                if hit:
                    issues.append(f"{rel}: contains original fund identifier(s), e.g. {hit[0]}")
    return issues


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    allow = "--allow-public-ids" in a
    folder = [x for x in a if not x.startswith("--")][0]
    ids = []
    if os.path.exists("private/id_map.csv"):
        ids = pd.read_csv("private/id_map.csv")["original_id"].astype(str).tolist()
    iss = audit(folder, allow, ids)
    print("AUDIT: clean, no raw series or non-anonymous IDs found." if not iss else
          "AUDIT ISSUES:\n  " + "\n  ".join(iss))
    return 4 if iss else 0


if __name__ == "__main__":
    sys.exit(main())
