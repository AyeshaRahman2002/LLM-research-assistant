# src/llm_opt/int8_vs_fp16_eval.py
import csv
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CANDS = [
    ("experiments/quantized/fp16", "fp16"),
    ("experiments/quantized/int8-bnb", "int8-bnb"),
]
OUT = Path("results/int8_vs_fp16.csv")

PROMPT = "Explain quantization-aware training in 3 sentences."


def bench(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    with torch.no_grad():
        enc = tok([PROMPT], return_tensors="pt").to(dev)
        t0 = time.time()
        _ = mdl.generate(**enc, max_new_tokens=160)
        dt = time.time() - t0
    return dt


if __name__ == "__main__":
    rows = []
    for d, tag in CANDS:
        if Path(d).exists():
            lat = bench(d)
            rows.append({"model": tag, "latency_s": round(lat, 3)})
            print(f"DONE {tag}: {lat:.3f}s")
        else:
            print(f"[skip] {tag} not found")
    if rows:
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"DONE wrote -> {OUT}")
