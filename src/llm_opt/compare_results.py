# src/llm_opt/compare_results.py
import csv
import json
from pathlib import Path

EFF = Path("results/efficiency_metrics.csv")
SUM = Path("results/summary_metrics.json")
OUT = Path("results/benchmark_table.csv")


def main():
    if not (EFF.exists() and SUM.exists()):
        raise SystemExit(
            "Need results/efficiency_metrics.csv and results/summary_metrics.json."
        )
    eff = list(csv.DictReader(EFF.open()))
    summ = json.loads(SUM.read_text())
    for r in eff:
        r["rougeL"] = summ.get("rougeL", "")
        r["bertscore_f1"] = summ.get("bertscore_f1", "")
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eff[0].keys())
        w.writeheader()
        w.writerows(eff)
    print(f"DONE table -> {OUT}")


if __name__ == "__main__":
    main()
