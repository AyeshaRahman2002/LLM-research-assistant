# src/data_ingest/csv_validators.py
"""
CSV schema & dtype validation with human-friendly reports.

Usage:
  python -m src.data_ingest.csv_validators --csv data/survey.csv \
      --schema-json configs/survey_schema.json \
      --out results/validation_report.json

Schema JSON example:
{
  "required_columns": ["id","age","gender","q1","q2","submitted_at"],
  "dtypes": {"id":"string","age":"int","gender":"category","submitted_at":"datetime"},
  "ranges": {"age":{"min":18,"max":100}},
  "categories": {"gender":["male","female","other"]},
  "unique": ["id"]
}
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PD_DTYPE_MAP = {
    "string": "string",
    "int": "Int64",
    "float": "Float64",
    "bool": "boolean",
    "datetime": "datetime64[ns]",
    "category": "category",
}


def coerce_dtype(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if dtype == "category":
        return series.astype("string").astype("category")
    if dtype == "bool":
        return (
            series.astype("string")
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "t": True,
                    "1": True,
                    "yes": True,
                    "false": False,
                    "f": False,
                    "0": False,
                    "no": False,
                }
            )
            .astype("boolean")
        )
    if dtype == "int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if dtype == "float":
        return pd.to_numeric(series, errors="coerce").astype("Float64")
    if dtype == "string":
        return series.astype("string")
    return series


def validate(df: pd.DataFrame, schema: dict) -> dict:
    report = {"errors": [], "warnings": [], "summary": {}}

    # required columns
    req = set(schema.get("required_columns", []))
    missing = [c for c in req if c not in df.columns]
    if missing:
        report["errors"].append({"type": "missing_columns", "columns": missing})

    # dtypes
    dtypes = schema.get("dtypes", {})
    coerced_cols = []
    for col, dt in dtypes.items():
        if col not in df.columns:
            continue
        before = str(df[col].dtype)
        df[col] = coerce_dtype(df[col], dt)
        after = str(df[col].dtype)
        if before != after:
            coerced_cols.append({"column": col, "from": before, "to": after})
    if coerced_cols:
        report["warnings"].append({"type": "coerced_dtypes", "details": coerced_cols})

    # ranges
    for col, r in schema.get("ranges", {}).items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        too_low = s < r.get("min", float("-inf"))
        too_high = s > r.get("max", float("inf"))
        bad = int((too_low | too_high).sum())
        if bad:
            report["errors"].append({"type": "range", "column": col, "violations": bad})

    # categories
    for col, cats in schema.get("categories", {}).items():
        if col not in df.columns:
            continue
        values = set(df[col].dropna().astype("string").str.lower().unique().tolist())
        allowed = set([str(x).lower() for x in cats])
        diff = sorted(list(values - allowed))
        if diff:
            report["warnings"].append(
                {"type": "unknown_categories", "column": col, "unknown": diff}
            )

    # unique constraints
    for col in schema.get("unique", []):
        if col in df.columns:
            dup = int(df[col].duplicated(keep=False).sum())
            if dup:
                report["errors"].append(
                    {"type": "non_unique", "column": col, "duplicates": dup}
                )

    # completion
    report["summary"] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--schema-json", required=True)
    ap.add_argument("--out", default="results/validation_report.json")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=True)
    schema = json.loads(Path(args.schema_json).read_text())
    rep = validate(df, schema)
    Path(args.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
