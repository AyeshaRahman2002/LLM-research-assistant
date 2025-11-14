# tests/test_analyze_data.py
import json
import os
import sys
from pathlib import Path

import pandas as pd


def test_analyze_data_end_to_end(tmp_path: Path, monkeypatch):
    # tiny CSV with numeric + binary target
    csv = tmp_path / "toy.csv"
    df = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": [10, 9, 8, 7, 6, 5],
            "group": ["A", "A", "B", "B", "B", "A"],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )
    df.to_csv(csv, index=False)

    # Act: run the CLI main (fast; no network)
    os.chdir(tmp_path)
    from src.data_analysis import analyze_data as m

    # Simulate CLI: --csv ... --target target
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    sys.argv = ["analyze_data", "--csv", str(csv), "--target", "target"]
    m.main()

    # Assert: summary json exists and references expected artifacts
    out = tmp_path / "results" / "data_analysis_summary.json"
    assert out.exists(), "results/data_analysis_summary.json should be created"
    payload = json.loads(out.read_text())
    assert payload["rows"] == 6
    assert "descriptive_stats.csv" in payload["outputs"]
    # optional regressions might be skipped on tiny data. just ensure at least one output exists
    produced = [k for k, v in payload["outputs"].items() if v]
    assert produced, "At least one analysis artifact should be produced"

    # Also ensure missing values report exists
    assert (tmp_path / "results" / "missing_values.csv").exists()
