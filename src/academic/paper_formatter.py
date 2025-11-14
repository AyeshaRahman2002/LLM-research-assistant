# src/academic/paper_formatter.py
"""
Minimal Markdown -> 'IEEE/ACM-like' formatting helper (dependency-light).
- Wraps your MD in a templated front matter.
- Optionally calls pandoc if available to render PDF.

Usage:
  python -m src.academic.paper_formatter --md reports/final_report.md --style ieee --out reports/final_report_formatted.md
  python -m src.academic.paper_formatter --md reports/final_report.md --style acm --pdf

If --pdf is passed and pandoc is on PATH, emits a PDF next to the MD.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

IEEE_HDR = """---
title: "{title}"
author:
  - name: "Research Assistant"
date: "{date}"
documentclass: article
geometry: margin=1in
fontsize: 11pt
---

# Abstract
"""

ACM_HDR = """---
title: "{title}"
author:
  - name: "Research Assistant"
date: "{date}"
documentclass: acmart
geometry: margin=1in
fontsize: 10pt
---

# Abstract
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--style", choices=["ieee", "acm"], default="ieee")
    ap.add_argument("--out", default="")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    src = Path(args.md)
    body = src.read_text()
    hdr = IEEE_HDR if args.style == "ieee" else ACM_HDR
    from datetime import date

    out_md = (
        Path(args.out) if args.out else src.with_name(src.stem + f"_{args.style}.md")
    )
    out_md.write_text(
        hdr.format(
            title=src.stem.replace("_", " ").title(), date=date.today().isoformat()
        )
        + "\n"
        + body
    )

    print(f"DONE wrote {out_md}")
    if args.pdf:
        if shutil.which("pandoc"):
            out_pdf = out_md.with_suffix(".pdf")
            subprocess.run(["pandoc", str(out_md), "-o", str(out_pdf)], check=False)
            print(f"DONE (best effort) PDF -> {out_pdf}")
        else:
            print("[WARN] pandoc not found; skipped PDF render")


if __name__ == "__main__":
    main()
