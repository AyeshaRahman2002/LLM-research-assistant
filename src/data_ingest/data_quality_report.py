# src/data_ingest/data_quality_report.py
"""
Auto data-quality report (profiling-lite):
- Missingness per column
- Type inference summary
- Basic numeric outlier counts (z>3)
- Category distribution heads
- Writes JSON + Markdown summary

Usage:
  python -m src.data_ingest.data_quality_report --csv data/survey.csv \
     --out-json results/data_quality.json --out-md reports/data_quality_report.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def zscore_outliers(s: pd.Series, thresh=3.0) -> int:
    s = pd.to_numeric(s, errors="coerce")
    s = s.dropna()
    if len(s) < 3:
        return 0
    z = (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) > 0 else 1.0)
    return int((np.abs(z) > thresh).sum())


def profile(df: pd.DataFrame) -> dict:
    prof = {"rows": int(len(df)), "columns": {}}
    for c in df.columns:
        col = df[c]
        info = {
            "dtype": str(col.dtype),
            "non_null": int(col.notna().sum()),
            "nulls": int(col.isna().sum()),
            "null_pct": float(col.isna().mean() * 100),
        }
        if pd.api.types.is_numeric_dtype(col):
            info["min"] = float(pd.to_numeric(col, errors="coerce").min(skipna=True))
            info["max"] = float(pd.to_numeric(col, errors="coerce").max(skipna=True))
            info["mean"] = float(pd.to_numeric(col, errors="coerce").mean(skipna=True))
            info["outliers_z3"] = zscore_outliers(col)
        else:
            vc = col.astype("string").value_counts(dropna=True).head(10)
            info["top_categories"] = vc.to_dict()
        prof["columns"][c] = info
    return prof


def to_markdown(summary: dict, out_csv_path: str = None) -> str:
    lines = []
    lines.append("# Data Quality Report\n")
    lines.append(f"Rows: **{summary['rows']}**\n")
    lines.append("## Columns\n")
    for c, info in summary["columns"].items():
        lines.append(f"### {c}")
        lines.append(f"- dtype: `{info['dtype']}`")
        lines.append(
            f"- non-null: {info['non_null']} / nulls: {info['nulls']} ({info['null_pct']:.2f}%)"
        )
        if "outliers_z3" in info:
            lines.append(
                f"- min/mean/max: {info.get('min')} / {info.get('mean'):.3f} / {info.get('max')}"
            )
            lines.append(f"- outliers (|z|>3): **{info['outliers_z3']}**")
        else:
            cats = info.get("top_categories", {})
            if cats:
                lines.append("- top categories:")
                for k, v in cats.items():
                    lines.append(f"  - {k}: {v}")
        lines.append("")
    if out_csv_path:
        lines.append(f"\n> Cleaned CSV: `{out_csv_path}`")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", default="results/data_quality.json")
    ap.add_argument("--out-md", default="reports/data_quality_report.md")
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    summary = profile(df)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary)
    Path(args.out_md).write_text(md)
    print(json.dumps({"json": args.out_json, "markdown": args.out_md}))


if __name__ == "__main__":
    main()
