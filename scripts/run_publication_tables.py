# scripts/run_publication_tables.py
"""
Aggregate efficiency CSVs into a LaTeX table for papers.
"""
from pathlib import Path

import pandas as pd


def main():
    metrics = Path("results/efficiency_metrics.csv")
    if not metrics.exists():
        print("no efficiency_metrics.csv found")
        return
    df = pd.read_csv(metrics)
    table = df.to_latex(index=False, float_format="%.3f")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/publication_table.tex").write_text(table)
    print("DONE wrote reports/publication_table.tex")


if __name__ == "__main__":
    main()
