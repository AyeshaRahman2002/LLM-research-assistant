# scripts/env_check.py
#!/usr/bin/env python3
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _ok(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def nvidia_smi():
    p = shutil.which("nvidia-smi")
    if not p:
        return None
    try:
        out = subprocess.check_output(
            [
                p,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "nvidia-smi present but query failed"


def main():
    rep = {}
    # python & os
    rep["python"] = sys.version.split()[0]
    rep["platform"] = sys.platform
    # torch
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
        rep["cuda_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        )
    except Exception as e:
        rep["torch_error"] = repr(e)
    # libs
    for m in [
        "transformers",
        "faiss",
        "faiss_cpu",
        "bitsandbytes",
        "peft",
        "pandas",
        "scikit_learn",
        "matplotlib",
    ]:
        rep[m] = _ok(m)
    rep["nvidia_smi"] = nvidia_smi()
    Path("results").mkdir(exist_ok=True, parents=True)
    out = Path("results/env_report.json")
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"DONE wrote -> {out}")


if __name__ == "__main__":
    main()
