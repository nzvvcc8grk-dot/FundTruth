"""Execute the notebook and export an A4 PDF via LaTeX; require exactly 5 pages.

  python build_report.py && python export_report.py

Patches applied to the generated .tex (cosmetic, reproducible): A4, 10 pt,
2 cm margins, no date, and removal of the soul/ulem packages ONLY if the local
LaTeX installation lacks them (they are only used for struck-out text).
"""
import re
import shutil
import subprocess
import sys

NB, TEX = "FundTruth_report.ipynb", "FundTruth_report.tex"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(f"failed: {' '.join(cmd)}")
    return r.stdout


run(["jupyter", "nbconvert", "--to", "notebook", "--execute", NB, "--output", NB])
run(["jupyter", "nbconvert", "--to", "latex", "--no-input", NB])
t = open(TEX, encoding="utf-8").read()
for pkg in ("soul", "ulem"):
    if subprocess.run(["kpsewhich", f"{pkg}.sty"], capture_output=True,
                      text=True).stdout.strip() == "":
        t = re.sub(r"^\s*\\usepackage(\[[^\]]*\])?\{%s\}.*$" % pkg, "", t, flags=re.M)
t = t.replace(r"\documentclass[11pt]{article}", r"\documentclass[10pt,a4paper]{article}")
t = re.sub(r"\\geometry\{verbose[^}]*\}",
           r"\\geometry{a4paper,tmargin=1.8cm,bmargin=1.8cm,lmargin=2cm,rmargin=2cm}", t)
t = t.replace(r"\maketitle", r"\setlength{\droptitle}{-2.4cm}\date{}\maketitle", 1)
open(TEX, "w", encoding="utf-8").write(t)
for _ in range(2):
    subprocess.run(["xelatex", "-interaction=nonstopmode", TEX],
                   capture_output=True, text=True)
info = run(["pdfinfo", "FundTruth_report.pdf"])
pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
print("pages:", pages)
if pages != 5:
    sys.exit("PAGE COUNT IS NOT 5: adjust the text before submitting.")
