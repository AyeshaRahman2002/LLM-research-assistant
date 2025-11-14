# tests/test_flash_attn.py
import json
import subprocess
from pathlib import Path


def test_flash_attention_check_runs():
    subprocess.run(["python3", "-m", "src.llm_opt.flash_attn_check"], check=True)
    p = Path("results/flash_attn_check.json")
    assert p.exists()
    data = json.loads(p.read_text())
    assert "elapsed_sec" in data
