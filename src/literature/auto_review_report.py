# src/literature/auto_review_report.py
import json
from pathlib import Path

SUMS = Path("results/summaries.json")
CLUST = Path("results/literature_clusters.json")
OUT = Path("reports/auto_review.md")

if __name__ == "__main__":
    if not (SUMS.exists() and CLUST.exists()):
        raise SystemExit("Need results/summaries.json and literature_clusters.json")
    sums = json.loads(SUMS.read_text())
    labels = json.loads(CLUST.read_text())["labels"]
    lines = ["# Automated Review\n"]
    by_c = {}
    for s, y in zip(sums, labels[: len(sums)]):
        by_c.setdefault(y, []).append(s)
    for c in sorted(by_c):
        lines.append(f"\n## Cluster {c}\n")
        for i, s in enumerate(by_c[c], 1):
            lines.append(f"**{i}. {s['title']}** — {s['generated']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"DONE review -> {OUT}")
