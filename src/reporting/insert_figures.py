# src/reporting/insert_figures.py
from pathlib import Path

IN_MD = Path("reports/final_report.md")
OUT_MD = Path("reports/final_report.figures.md")

FIGS = [
    ("results/literature_clusters.png", "Literature clusters"),
    ("results/visualizations/correlation_heatmap.png", "Correlation heatmap"),
    ("plots/tradeoff_curves.png", "Accuracy–Latency trade-off"),
    ("results/literature_umap.png", "UMAP projection"),
]

if __name__ == "__main__":
    base = IN_MD.read_text(encoding="utf-8") if IN_MD.exists() else "# Final Report\n"
    lines = [base, "\n## Figures\n"]
    for p, cap in FIGS:
        if Path(p).exists():
            lines.append(f"\n![{cap}]({p})\n")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"DONE augmented report -> {OUT_MD}")
