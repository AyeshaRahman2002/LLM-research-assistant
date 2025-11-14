# src/llm_opt/eval_tokens_per_sec.py
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "experiments/quantized/fp32"
OUT = Path("results/tokens_per_sec.json")

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    res = []
    with torch.no_grad():
        for L in [32, 128, 512, 1024]:
            prompt = "x " * L
            enc = tok([prompt], return_tensors="pt").to(dev)
            t0 = time.time()
            ids = mdl.generate(**enc, max_new_tokens=200)
            dt = time.time() - t0
            res.append(
                {
                    "prompt_tokens": L,
                    "gen_tokens": int(ids.shape[-1]),
                    "latency_s": round(dt, 3),
                    "tokens_per_sec": round(ids.shape[-1] / dt, 2),
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"DONE toks/sec -> {OUT}")
