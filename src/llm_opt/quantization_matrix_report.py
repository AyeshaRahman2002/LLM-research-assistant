# src/llm_opt/quantization_matrix_report.py
"""
Visualize / dump quantization errors (mock report).
"""
import json
from pathlib import Path

import numpy as np


def main():
    W = np.random.randn(256, 256)
    Wq = np.round(W, 1)
    err = float(np.mean((W - Wq) ** 2))
    Path("results").mkdir(exist_ok=True)
    report = {"mean_squared_quant_error": err, "shape": list(W.shape)}
    Path("results/quantization_matrix_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(report)


if __name__ == "__main__":
    main()
