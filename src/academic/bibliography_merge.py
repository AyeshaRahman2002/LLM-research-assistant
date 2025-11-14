# src/academic/bibliography_merge.py
"""
Tiny BibTeX merger & de-duplicator by citation key.
- Merges multiple .bib files into reports/merged_refs.bib
- Drops duplicates by entry key

Usage:
  python -m src.academic.bibliography_merge --inputs reports/refs.bib other.bib --out reports/merged_refs.bib
"""
import argparse
import re
from pathlib import Path

ENTRY_RE = re.compile(r"@\w+\s*\{\s*([^,]+)\s*,", re.IGNORECASE)


def split_entries(text: str):
    # naive split by top-level '@' entries
    parts, buf, depth = [], [], 0
    for ch in text:
        buf.append(ch)
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if depth == 0 and "".join(buf).strip().startswith("@") and ch == "}":
            parts.append("".join(buf).strip())
            buf = []
    if buf and "@" in "".join(buf):
        parts.append("".join(buf).strip())
    return parts


def key_of(entry: str) -> str:
    m = ENTRY_RE.search(entry)
    return m.group(1).strip() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/merged_refs.bib")
    args = ap.parse_args()

    seen, merged = set(), []
    for p in args.inputs:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        for e in split_entries(txt):
            k = key_of(e)
            if k and k not in seen:
                seen.add(k)
                merged.append(e)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n\n".join(merged))
    print(f"DONE merged {len(merged)} entries -> {args.out}")


if __name__ == "__main__":
    main()
