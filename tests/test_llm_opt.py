# tests/test_llm_opt.py
import json
import os
from pathlib import Path


def test_system_resource_report(tmp_path):
    # Run the system resource probe; it has no GPU/network hard requirements
    os.chdir(tmp_path)
    from src.llm_opt import system_resource_report as m

    m.main()
    p = Path("results/system_resources.json")
    assert p.exists(), "system_resources.json should be created"
    data = json.loads(p.read_text())
    # must include CPU count and 'gpu' key
    assert "cpu_count" in data
    assert "gpu" in data
