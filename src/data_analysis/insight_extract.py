# src/data_analysis/insight_extract.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results")
OUT = RES / "insights.json"

if __name__ == "__main__":
    insights = {"strong_correlations": [], "significant_features": []}

    # Top absolute correlations (upper triangle only)
    cp = RES / "correlations.csv"
    if cp.exists():
        C = pd.read_csv(cp, index_col=0).abs()
        mask = np.triu(np.ones(C.shape), k=1).astype(bool)
        tri_vals = C.where(mask).stack().sort_values(ascending=False)
        for (a, b), v in tri_vals.head(10).items():
            if v >= 0.5:
                insights["strong_correlations"].append(
                    {"a": a, "b": b, "abs_corr": float(v)}
                )

    # Significant features from t-test / anova
    for pth in [RES / "ttest_results.csv", RES / "anova_results.csv"]:
        if pth.exists():
            df = pd.read_csv(pth).sort_values("p_value")
            insights["significant_features"] += df.head(10).to_dict(orient="records")

    OUT.write_text(json.dumps(insights, indent=2), encoding="utf-8")
    print(f"DONE insights -> {OUT}")
