# src/literature/plot_clusters.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PCA_PATH = Path("plots/literature_pca.npy")
LAB_PATH = Path("results/literature_clusters.json")
ITEMS_PATH = Path("data/literature_items.json")
OUT_PNG = Path("results/literature_clusters.png")


def main():
    if not PCA_PATH.exists() or not LAB_PATH.exists():
        raise SystemExit("Run embed_cluster first to create PCA and labels.")
    X = np.load(PCA_PATH)  # [N,2]
    labels = json.loads(LAB_PATH.read_text())["labels"]
    items = json.loads(ITEMS_PATH.read_text()) if ITEMS_PATH.exists() else None

    plt.figure(figsize=(8, 6))
    for k in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == k]
        plt.scatter(X[idx, 0], X[idx, 1], s=40, alpha=0.8, label=f"Cluster {k}")
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.title("Literature Clusters (MiniLM embeddings + PCA)")
    plt.legend(loc="best", fontsize=9, frameon=False)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print(f"DONE saved cluster plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
