# src/llm_opt/distributed_eval.py
"""
Simulate multi-GPU evaluation (even on CPU) — logs device info.
"""
import json
from pathlib import Path

import torch


def main():
    world = torch.cuda.device_count() if torch.cuda.is_available() else 1
    devices = (
        [f"cuda:{i}" for i in range(world)] if torch.cuda.is_available() else ["cpu"]
    )
    out = {"world_size": world, "devices": devices}
    Path("results").mkdir(exist_ok=True)
    Path("results/distributed_eval.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
