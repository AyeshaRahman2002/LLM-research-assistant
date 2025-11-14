# src/literature/normalize_papers.py
import argparse
import json
import re
from pathlib import Path

ARXIV_XML = Path("data/literature_raw.xml")
S2_JSON = Path("data/semanticscholar_raw.json")
OUT_JSON = Path("data/literature_items.json")


def _parse_arxiv(xml_text: str):
    ids = re.findall(r"<id>(.*?)</id>", xml_text, flags=re.DOTALL)[1:]
    titles = re.findall(r"<title>(.*?)</title>", xml_text, flags=re.DOTALL)[1:]
    sums = re.findall(r"<summary>(.*?)</summary>", xml_text, flags=re.DOTALL)
    items = []
    for i, t in enumerate(titles):
        aid = (ids[i] if i < len(ids) else "").strip()
        items.append(
            {
                "id": aid.split("/")[-1],
                "title": re.sub(r"\s+", " ", t).strip(),
                "summary": re.sub(
                    r"\s+", " ", (sums[i] if i < len(sums) else "")
                ).strip(),
                "source": "arxiv",
                "url": aid,
                "doi": None,
            }
        )
    return items


def _parse_s2(raw: dict):
    out = []
    for d in raw.get("data", []):
        ext = d.get("externalIds", {}) or {}
        out.append(
            {
                "id": d.get("paperId")
                or ext.get("ArXiv")
                or ext.get("DOI")
                or d.get("url"),
                "title": (d.get("title") or "").strip(),
                "summary": (d.get("abstract") or "").strip(),
                "source": "semanticscholar",
                "url": d.get("url"),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
            }
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv", default=str(ARXIV_XML))
    ap.add_argument("--s2", default=str(S2_JSON))
    ap.add_argument("--out", default=str(OUT_JSON))
    a = ap.parse_args()

    items = []
    if Path(a.arxiv).exists():
        items += _parse_arxiv(Path(a.arxiv).read_text(encoding="utf-8"))
    if Path(a.s2).exists():
        items += _parse_s2(json.loads(Path(a.s2).read_text(encoding="utf-8")))

    seen = set()
    uniq = []
    for it in items:
        k = it["title"].casefold()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(it)

    Path(a.out).write_text(json.dumps(uniq, indent=2), encoding="utf-8")
    print(f"DONE normalized -> {a.out} (n={len(uniq)})")
