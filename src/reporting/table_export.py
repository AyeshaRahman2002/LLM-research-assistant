# src/reporting/table_export.py
import json
from pathlib import Path

import pandas as pd

OUT_XLSX = Path("reports/summary_tables.xlsx")

CANDIDATES = [
    ("results/efficiency_metrics.csv", "efficiency_metrics"),
    ("results/benchmark_table.csv", "benchmark_table"),
    ("results/rag_eval.csv", "rag_eval_snippets"),
    ("results/rag_eval_metrics.json", "rag_metrics_json"),
    ("results/tokens_per_sec.json", "tokens_per_sec_json"),
    ("results/batch_ablation.json", "batch_ablation_json"),
    ("results/summary_metrics.json", "summary_metrics_json"),
]


def as_df(path: Path):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    # json -> flat dataframe when possible
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.json_normalize(obj)


if __name__ == "__main__":
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    # Filter only existing files
    existing = [(p, name) for p, name in CANDIDATES if Path(p).exists()]
    if not existing:
        print("[warn] No input files found; skipping Excel export.")
        raise SystemExit(0)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
        wrote = False
        for p, name in existing:
            pth = Path(p)
            try:
                df = as_df(pth)
                df.to_excel(xw, sheet_name=name[:31], index=False)
                wrote = True
            except Exception:
                pd.DataFrame({"content": [pth.read_text()]}).to_excel(
                    xw, sheet_name=name[:31], index=False
                )
                wrote = True
        if not wrote:
            # ensure at least one visible sheet
            pd.DataFrame({"message": ["No data available"]}).to_excel(
                xw, sheet_name="empty", index=False
            )

    print(f"DONE tables workbook -> {OUT_XLSX}")
