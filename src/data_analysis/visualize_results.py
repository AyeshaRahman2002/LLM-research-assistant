# src/data_analysis/visualize_results.py
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

RESULTS_DIR = Path("results")
VIS_DIR = RESULTS_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def histograms(df: pd.DataFrame, limit=6):
    num_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    for c in num_cols[:limit]:
        s = df[c].dropna()
        if s.empty:
            continue
        plt.figure(figsize=(6, 4))
        s.plot(kind="hist", bins=30)
        plt.title(f"Histogram — {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"hist_{c}.png", dpi=140)
        plt.close()


def correlation_heatmap(df: pd.DataFrame):
    num_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if len(num_cols) < 2:
        return
    C = df[num_cols].corr().values
    plt.figure(figsize=(7, 6))
    im = plt.imshow(C, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def scatter_pairs(df: pd.DataFrame, limit=4):
    num_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ][:limit]
    if len(num_cols) < 2:
        return
    n = len(num_cols)
    size = 3.0
    fig, axes = plt.subplots(n, n, figsize=(size * n, size * n))
    for i, xi in enumerate(num_cols):
        for j, yj in enumerate(num_cols):
            ax = axes[i, j]
            if i == j:
                s = df[xi].dropna()
                if not s.empty:
                    ax.hist(s, bins=25)
            else:
                tmp = df[[xi, yj]].dropna()
                if not tmp.empty:
                    ax.scatter(tmp[xi], tmp[yj], s=6, alpha=0.7)
            if i == n - 1:
                ax.set_xlabel(yj)
            if j == 0:
                ax.set_ylabel(xi)
    fig.suptitle("Scatter Matrix (first few numeric columns)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(VIS_DIR / "scatter_matrix.png", dpi=140)
    plt.close(fig)


def boxplots_by_group(df: pd.DataFrame, target: str, limit=6):
    if target not in df.columns:
        return
    if df[target].nunique() < 2:
        return
    num_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    for c in num_cols[:limit]:
        plt.figure(figsize=(6, 4))
        groups = [g[1][c].dropna().values for g in df.groupby(target)]
        labels = [str(g) for g, _ in df.groupby(target)]
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.title(f"{c} by {target}")
        plt.tight_layout()
        plt.savefig(VIS_DIR / f"box_{c}_by_{target}.png", dpi=150)
        plt.close()


def pairwise_kde(df: pd.DataFrame, limit=3):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    num_cols = num_cols[:limit]
    if len(num_cols) < 2:
        return

    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            xi, yj = num_cols[i], num_cols[j]

            # 1) Align by dropping NaNs jointly so x and y are SAME LENGTH
            sub = df[[xi, yj]].dropna()
            if len(sub) < 40:  # need enough samples
                continue

            x = sub[xi].values
            y = sub[yj].values

            # 2) Guard against constant columns (KDE fails if zero variance)
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue

            # 3) Percentile trimming for stable grids
            xmin, xmax = np.percentile(x, [1, 99])
            ymin, ymax = np.percentile(y, [1, 99])
            if (
                not np.isfinite([xmin, xmax, ymin, ymax]).all()
                or xmin == xmax
                or ymin == ymax
            ):
                continue

            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])

            try:
                kde = gaussian_kde(values)
                f = np.reshape(kde(positions).T, xx.shape)
            except Exception:
                # KDE can still fail on weird distributions; just skip gracefully
                continue

            plt.figure(figsize=(5.5, 4.5))
            plt.imshow(np.rot90(f), extent=[xmin, xmax, ymin, ymax], aspect="auto")
            plt.scatter(x, y, s=3, alpha=0.4)
            plt.xlabel(xi)
            plt.ylabel(yj)
            plt.title("Pairwise KDE")
            plt.tight_layout()
            plt.savefig(VIS_DIR / f"kde_{xi}_{yj}.png", dpi=150)
            plt.close()


def parallel_coordinates(df: pd.DataFrame, limit=6):
    from pandas.plotting import parallel_coordinates as _pc

    num_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    cols = num_cols[:limit]
    if len(cols) < 3:
        return
    tmp = df[cols].copy().dropna().copy()
    tmp["__bucket__"] = pd.qcut(
        tmp[cols[0]], q=min(4, max(2, len(tmp) // 10)), duplicates="drop"
    ).astype(str)
    plt.figure(figsize=(8, 4.5))
    _pc(tmp, "__bucket__", colormap=None)
    plt.title("Parallel Coordinates (bucketed by first numeric col)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "parallel_coordinates.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    histograms(df)
    correlation_heatmap(df)
    scatter_pairs(df)
    if args.target:
        boxplots_by_group(df, args.target)
    pairwise_kde(df)
    parallel_coordinates(df)
    print(f"DONE plots saved -> {VIS_DIR}/")


if __name__ == "__main__":
    main()
