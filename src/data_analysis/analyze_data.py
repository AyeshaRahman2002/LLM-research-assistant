# src/data_analysis/analyze_data.py
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path("results")
VIS_DIR = RESULTS_DIR / "visualizations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


def _infer_roles(df: pd.DataFrame, target: Optional[str]):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    y = df[target] if (target and target in df.columns) else None
    return num_cols, cat_cols, y


def descriptive_stats(df: pd.DataFrame):
    desc = df.describe(include="all").transpose()
    desc.to_csv(RESULTS_DIR / "descriptive_stats.csv")
    return desc


def missing_report(df: pd.DataFrame):
    miss = df.isna().sum().sort_values(ascending=False).rename("missing")
    out = pd.concat([miss, (miss / len(df)).rename("missing_pct")], axis=1)
    out.to_csv(RESULTS_DIR / "missing_values.csv")
    return out


def export_standardized(df: pd.DataFrame, num_cols):
    if len(num_cols) < 1:
        return None
    z = df.copy()
    zc = z[num_cols].astype(float)
    sc = StandardScaler().fit(zc)
    z[num_cols] = sc.transform(zc)
    p = RESULTS_DIR / "standardized.csv"
    z.to_csv(p, index=False)
    return p


def correlations(df: pd.DataFrame, num_cols):
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(4)
        corr.to_csv(RESULTS_DIR / "correlations.csv")
        return corr
    return pd.DataFrame()


def ttest_binary(df: pd.DataFrame, num_cols, y):
    out = []
    if y is None:
        return pd.DataFrame()
    if y.nunique() == 2:
        g1, g2 = list(y.dropna().unique())
        for c in num_cols:
            try:
                a = df.loc[y == g1, c].dropna()
                b = df.loc[y == g2, c].dropna()
                if len(a) > 3 and len(b) > 3:
                    t, p = stats.ttest_ind(a, b, equal_var=False)
                    out.append({"feature": c, "t": float(t), "p_value": float(p)})
            except Exception:
                pass
    res = pd.DataFrame(out).sort_values("p_value") if out else pd.DataFrame()
    if not res.empty:
        res.to_csv(RESULTS_DIR / "ttest_results.csv", index=False)
    return res


def anova_oneway(df: pd.DataFrame, num_cols, y):
    """One-way ANOVA when target has >2 groups."""
    if y is None or y.nunique() < 3:
        return pd.DataFrame()
    out = []
    for c in num_cols:
        groups = [df.loc[y == g, c].dropna().values for g in y.dropna().unique()]
        groups = [g for g in groups if len(g) > 2]
        if len(groups) >= 3:
            F, p = stats.f_oneway(*groups)
            out.append({"feature": c, "F": float(F), "p_value": float(p)})
    res = pd.DataFrame(out).sort_values("p_value") if out else pd.DataFrame()
    if not res.empty:
        res.to_csv(RESULTS_DIR / "anova_results.csv", index=False)
    return res


def simple_regression(df: pd.DataFrame, num_cols, y) -> Dict[str, Any]:
    if y is None or not pd.api.types.is_numeric_dtype(y):
        return {}
    X = df[num_cols].dropna()
    y2 = y.loc[X.index]
    if len(X) < 10:
        return {}
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X.values)
    lr = LinearRegression().fit(Xz, y2.values)
    coefs = dict(zip(num_cols, lr.coef_.tolist()))
    summary = {
        "target": y.name,
        "n": int(len(X)),
        "r2": float(lr.score(Xz, y2.values)),
        "coef": coefs,
    }
    (RESULTS_DIR / "regression_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def unsupervised_kmeans(df: pd.DataFrame, num_cols, k=3):
    if len(num_cols) < 2:
        return {}
    X = df[num_cols].dropna()
    if len(X) < k:
        return {}
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X.values)
    labels = pd.Series(km.labels_, index=X.index, name="kmeans_label")
    labels.to_csv(RESULTS_DIR / "kmeans_labels.csv")
    return {"k": int(k), "n": int(len(X)), "inertia": float(km.inertia_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--kmeans", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    num_cols, cat_cols, y = _infer_roles(df, args.target)

    desc = descriptive_stats(df)
    miss = missing_report(df)
    std_path = export_standardized(df, num_cols)
    corr = correlations(df, num_cols)
    ttest = ttest_binary(df, num_cols, y)
    anov = anova_oneway(df, num_cols, y)
    reg = simple_regression(df, num_cols, y)
    km = {} if y is not None else unsupervised_kmeans(df, num_cols, k=args.kmeans)

    summary = {
        "input_csv": args.csv,
        "rows": int(len(df)),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "outputs": {
            "descriptive_stats.csv": (RESULTS_DIR / "descriptive_stats.csv").as_posix(),
            "missing_values.csv": (RESULTS_DIR / "missing_values.csv").as_posix(),
            "standardized.csv": std_path.as_posix() if std_path else None,
            "correlations.csv": (
                (RESULTS_DIR / "correlations.csv").as_posix()
                if not corr.empty
                else None
            ),
            "ttest_results.csv": (
                (RESULTS_DIR / "ttest_results.csv").as_posix()
                if not ttest.empty
                else None
            ),
            "anova_results.csv": (
                (RESULTS_DIR / "anova_results.csv").as_posix()
                if not anov.empty
                else None
            ),
            "regression_summary.json": (
                (RESULTS_DIR / "regression_summary.json").as_posix() if reg else None
            ),
            "kmeans_labels.csv": (
                (RESULTS_DIR / "kmeans_labels.csv").as_posix() if km else None
            ),
        },
        "models": {"regression": reg, "kmeans": km},
    }
    (RESULTS_DIR / "data_analysis_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("DONE analysis complete -> results/data_analysis_summary.json")


if __name__ == "__main__":
    main()
