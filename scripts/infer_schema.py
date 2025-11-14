# scripts/infer_schema.py
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import pandas as pd


def infer_schema(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    schema = {
        "required_columns": list(df.columns),
        "dtypes": {},
        "ranges": {},
        "categories": {},
        "unique": [],
    }
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_integer_dtype(s):
            schema["dtypes"][c] = "int"
            schema["ranges"][c] = {
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
            }
        elif pd.api.types.is_float_dtype(s):
            schema["dtypes"][c] = "float"
            schema["ranges"][c] = {
                "min": float(s.min(skipna=True)),
                "max": float(s.max(skipna=True)),
            }
        elif pd.api.types.is_bool_dtype(s):
            schema["dtypes"][c] = "bool"
            schema["categories"][c] = [True, False]
        elif pd.api.types.is_datetime64_any_dtype(s):
            schema["dtypes"][c] = "datetime"
        else:
            schema["dtypes"][c] = "string"
            # If few unique values, treat as categorical
            nunique = s.nunique(dropna=True)
            if 2 <= nunique <= 20:
                schema["categories"][c] = sorted(
                    [str(x) for x in s.dropna().unique().tolist()][:50]
                )
        # Heuristic: column ending with _id should be unique
        if str(c).lower().endswith("_id"):
            schema["unique"].append(c)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(schema, indent=2))
    print(f"DONE Wrote inferred schema -> {out_path}")


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "configs/survey_schema.json"
    infer_schema(csv, out)
