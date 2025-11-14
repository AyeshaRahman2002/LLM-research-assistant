# src/llm_opt/plot_tradeoffs.py
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IN = Path("results/benchmark_table.csv")
OUT = Path("plots/tradeoff_curves.png")


def main():
    df = pd.read_csv(IN)
    plt.figure(figsize=(6, 4))
    for _, r in df.iterrows():
        plt.scatter(r["latency_s"], r["rougeL"])
        plt.text(r["latency_s"], r["rougeL"], r["model_dir"].split("/")[-1], fontsize=8)
    plt.xlabel("Latency (s) ↓")
    plt.ylabel("ROUGE-L ↑")
    plt.title("Accuracy vs. Latency Trade-off")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160)
    print(f"DONE plot -> {OUT}")


if __name__ == "__main__":
    main()
