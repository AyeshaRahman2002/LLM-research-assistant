# src/surveys/survey_clean.py
"""
Survey cleaning utilities:
- MCQ normalization (strip/space/lower)
- Likert scoring (1..5 or custom map)
- One-hot encode MCQ
- Output cleaned CSV + codebook JSON

Usage:
  python -m src.surveys.survey_clean \
    --csv data/survey.csv \
    --likert-cols q1,q2,q3 \
    --mcq-cols tools_used,interests \
    --likert-map "strongly disagree:1,disagree:2,neutral:3,agree:4,strongly agree:5" \
    --out-csv results/survey_clean.csv \
    --out-meta results/survey_codebook.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_map(spec: str):
    m = {}
    for kv in spec.split(","):
        if not kv.strip():
            continue
        k, v = kv.split(":")
        m[k.strip().lower()] = float(v)
    return m


def normalize_token(x):
    if pd.isna(x):
        return None
    return str(x).strip().lower()


def score_likert(df, cols, mapping):
    scored = {}
    for c in cols:
        if c not in df.columns:
            continue
        scored[c + "_score"] = df[c].map(
            lambda x: mapping.get(normalize_token(x), None)
        )
    return pd.DataFrame(scored)


def one_hot_mcq(df, cols, sep=";"):
    out = {}
    codebook = {}
    for c in cols:
        if c not in df.columns:
            continue
        tokens = (
            df[c]
            .fillna("")
            .astype(str)
            .map(lambda s: [t.strip().lower() for t in s.split(sep) if t.strip()])
        )
        vocab = sorted({t for lst in tokens for t in lst})
        codebook[c] = vocab
        for v in vocab:
            out[f"{c}__{v}"] = tokens.map(lambda lst: 1 if v in lst else 0)
    return pd.DataFrame(out), codebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--likert-cols", default="")
    ap.add_argument("--mcq-cols", default="")
    ap.add_argument(
        "--likert-map",
        default="strongly disagree:1,disagree:2,neutral:3,agree:4,strongly agree:5",
    )
    ap.add_argument("--out-csv", default="results/survey_clean.csv")
    ap.add_argument("--out-meta", default="results/survey_codebook.json")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(args.csv)

    likert_cols = [c for c in args.likert_cols.split(",") if c]
    mcq_cols = [c for c in args.mcq_cols.split(",") if c]
    mapping = parse_map(args.likert_map)

    df_scored = score_likert(df, likert_cols, mapping)
    df_mcq, codebook = one_hot_mcq(df, mcq_cols)

    out = df.copy()
    if not df_scored.empty:
        out = pd.concat([out, df_scored], axis=1)
    if not df_mcq.empty:
        out = pd.concat([out, df_mcq], axis=1)

    out.to_csv(args.out_csv, index=False)
    meta = {
        "likert_map": mapping,
        "mcq_codebook": codebook,
        "rows": len(out),
        "columns": list(out.columns),
    }
    Path(args.out_meta).write_text(json.dumps(meta, indent=2))
    print(json.dumps({"cleaned_csv": args.out_csv, "meta": args.out_meta}))


if __name__ == "__main__":
    main()
