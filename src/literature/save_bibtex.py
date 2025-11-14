# src/literature/save_bibtex.py
import json
import re
from pathlib import Path

IN = Path("data/literature_items.json")
OUT = Path("reports/refs.bib")


def _slug(t):
    return (re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_") or "ref")[:40]


if __name__ == "__main__":
    if not IN.exists():
        raise SystemExit("Run normalize_papers first.")
    items = json.loads(IN.read_text())
    lines = []
    for it in items:
        key = _slug(it["title"])
        url = it.get("url") or ""
        doi = it.get("doi") or ""
        lines.append("@misc{" + key + ",")
        lines.append(f"  title={{ {it['title']} }},")
        if doi:
            lines.append(f"  doi={{ {doi} }},")
        if url:
            lines.append(f"  url={{ {url} }},")
        lines.append("  note={LLM-Research-Assistant AutoBib}")
        lines.append("}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE BibTeX -> {OUT}")
