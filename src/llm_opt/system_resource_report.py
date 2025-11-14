# src/llm_opt/system_resource_report.py
import json
import os
import shutil
import subprocess
from pathlib import Path

OUT = Path("results/system_resources.json")


def gpu_info():
    p = shutil.which("nvidia-smi")
    if not p:
        return {}
    try:
        out = subprocess.check_output([p, "-L"]).decode()
        out2 = subprocess.check_output(
            [
                p,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ).decode()
        return {"list": out.strip(), "util": out2.strip()}
    except Exception:
        return {"list": "n/a", "util": "n/a"}


def main():
    info = {"cpu_count": os.cpu_count(), "gpu": gpu_info()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_GB"] = round(vm.total / (1024**3), 2)
        info["ram_available_GB"] = round(vm.available / (1024**3), 2)
        info["load_avg"] = psutil.getloadavg() if hasattr(psutil, "getloadavg") else []
    except Exception:
        info["psutil"] = "not available"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2))
    print(f"DONE system report -> {OUT}")


if __name__ == "__main__":
    main()
