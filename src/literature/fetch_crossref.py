# src/literature/fetch_crossref.py
import argparse
import json
import time
from pathlib import Path

import requests

IN = Path("data/literature_items.json")
OUT = Path("data/literature_items_enriched.json")

CR = "https://api.crossref.org/works"


def lookup_title(title: str, pause=0.2):
    params = {"query.title": title, "rows": 1, "select": "DOI,URL,title,issued"}
    r = requests.get(CR, params=params, timeout=20)
    time.sleep(pause)
    if r.status_code != 200:
        return {}
    items = r.json().get("message", {}).get("items", [])
    return items[0] if items else {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default=str(IN))
    ap.add_argument("--outfile", default=str(OUT))
    a = ap.parse_args()

    items = json.loads(Path(a.infile).read_text(encoding="utf-8"))
    out = []
    for it in items:
        if it.get("doi"):
            out.append(it)
            continue
        hit = lookup_title(it["title"])
        it["doi"] = hit.get("DOI", it.get("doi"))
        it["url"] = hit.get("URL", it.get("url"))
        out.append(it)

    Path(a.outfile).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"DONE crossref-enriched -> {a.outfile} (n={len(out)})")
