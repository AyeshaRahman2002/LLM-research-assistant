# src/data_analysis/anomaly_detect.py
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest

OUT = Path("results/anomalies.csv")


def z_flags(df, thresh=3.0):
    num = df.select_dtypes(include="number")
    Z = (num - num.mean()) / num.std(ddof=0)
    flags = Z.abs() > thresh
    rows = flags.any(axis=1)
    out = df.loc[rows].copy()
    out["zscore_flags"] = flags.loc[rows].apply(
        lambda r: ",".join([c for c, v in r.items() if v]), axis=1
    )
    return out


def iforest(df):
    num = df.select_dtypes(include="number").dropna()
    if len(num) < 20:
        return pd.DataFrame()
    clf = IsolationForest(n_estimators=100, random_state=42, contamination="auto")
    y = clf.fit_predict(num)
    out = df.loc[num.index].copy()
    out["iforest_flag"] = (y == -1).astype(int)
    return out[out["iforest_flag"] == 1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    z = z_flags(df)
    i = iforest(df)
    merged = pd.concat([z, i[~i.index.isin(z.index)]], axis=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT, index=False)
    print(f"DONE anomalies -> {OUT} (rows={len(merged)})")
