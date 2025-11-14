# src/literature/cite_link_injection.py
import json
from pathlib import Path

ITEMS = Path("data/literature_items.json")
SUMS = Path("results/summaries.json")
OUTJ = Path("results/summaries_cited.json")
INMD = Path("reports/literature_summary.md")
OUTMD = Path("reports/literature_summary_cited.md")

if __name__ == "__main__":
    if not (ITEMS.exists() and SUMS.exists()):
        raise SystemExit("Need items and summaries.")
    items = json.loads(ITEMS.read_text())
    sums = json.loads(SUMS.read_text())
    link_by_title = {
        it["title"].strip(): (
            it.get("url") or (it.get("doi") and f"https://doi.org/{it['doi']}") or ""
        )
        for it in items
    }
    for s in sums:
        s["link"] = link_by_title.get(s["title"].strip(), "")
    OUTJ.write_text(json.dumps(sums, indent=2), encoding="utf-8")
    print(f"DONE wrote -> {OUTJ}")
    if INMD.exists():
        text = INMD.read_text()
        for t, url in link_by_title.items():
            if url:
                text = text.replace(f"### {t}", f"### [{t}]({url})")
        OUTMD.write_text(text, encoding="utf-8")
        print(f"DONE linkified -> {OUTMD}")
