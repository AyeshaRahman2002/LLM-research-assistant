# src/literature/write_summary_md.py
import json
from pathlib import Path
from textwrap import shorten

SUM_PATH = Path("results/summaries.json")
PNG_PATH = Path("results/literature_clusters.png")
OUT_MD = Path("reports/literature_summary.md")


def main():
    if not SUM_PATH.exists():
        raise SystemExit("Missing results/summaries.json. Run summarize_papers first.")
    data = json.loads(SUM_PATH.read_text())

    lines = []
    lines.append("# Literature Review — Auto Summary\n")
    if PNG_PATH.exists():
        lines.append(f"![clusters]({PNG_PATH.as_posix()})\n")
    lines.append("## Papers & Generated Summaries\n")
    for i, d in enumerate(data, 1):
        title = d["title"].strip()
        src = shorten(d["summary"].strip(), width=260, placeholder="…")
        gen = d["generated"].strip()
        lines.append(f"### {i}. {title}\n")
        lines.append(f"**Abstract (truncated):** {src}\n\n")
        lines.append(f"**Generated summary:** {gen}\n\n---\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE wrote report -> {OUT_MD}")


if __name__ == "__main__":
    main()
