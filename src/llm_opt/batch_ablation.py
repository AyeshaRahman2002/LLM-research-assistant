# src/llm_opt/batch_ablation.py
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "experiments/quantized/fp32"
OUT = Path("results/batch_ablation.json")

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    res = []
    prompts = ["Summarize attention."] * 256
    with torch.no_grad():
        for b in [1, 2, 4, 8, 16, 32]:
            enc = tok(
                prompts[:b], return_tensors="pt", padding=True, truncation=True
            ).to(dev)
            t0 = time.time()
            _ = mdl.generate(**enc, max_new_tokens=64)
            dt = time.time() - t0
            res.append(
                {
                    "batch_size": b,
                    "latency_s": round(dt, 3),
                    "throughput_req_per_s": round(b / max(dt, 1e-6), 2),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"DONE batch ablation -> {OUT}")
