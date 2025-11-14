# tests/test_efficiency.py
import json
import subprocess
from pathlib import Path


def test_kv_cache_and_flash_attn(tmp_path):
    subprocess.run(["python3", "-m", "src.llm_opt.kv_cache_bench"], check=True)
    subprocess.run(["python3", "-m", "src.llm_opt.flash_attn_check"], check=True)
    assert Path("results/kv_cache_bench.json").exists()
    assert Path("results/flash_attn_check.json").exists()
    kv = json.loads(Path("results/kv_cache_bench.json").read_text())
    assert "speedup_ratio" in kv


def test_quantization_report_and_triton():
    subprocess.run(
        ["python3", "-m", "src.llm_opt.quantization_matrix_report"], check=True
    )
    subprocess.run(["python3", "-m", "src.llm_opt.triton_export"], check=True)
    assert Path("results/quantization_matrix_report.json").exists()
    assert Path("results/tiny_gpt2_traced.pt").exists()
