# src/llm_opt/evaluate_efficiency.py
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = ["experiments/quantized/fp32"]

PROMPTS = [
    "Summarize the key ideas of attention mechanisms in 3 sentences.",
    "Explain parameter-efficient fine-tuning in 2 sentences.",
]

OUT_CSV = Path("results/efficiency_metrics.csv")


def vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    total, reserved = torch.cuda.get_device_properties(
        0
    ).total_memory, torch.cuda.memory_reserved(0)
    return reserved / (1024**2)


def evaluate_dir(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    latencies = []
    with torch.no_grad():
        for p in PROMPTS:
            enc = tok([p], return_tensors="pt").to(device)
            t0 = time.time()
            out = mdl.generate(**enc, max_new_tokens=128, num_beams=4)
            dt = time.time() - t0
            _ = tok.decode(out[0], skip_special_tokens=True)
            latencies.append(dt)
    return sum(latencies) / len(latencies), vram_mb()


def main():
    rows = []
    for m in MODELS:
        if not Path(m).exists():
            print(f"[skip] {m} not found")
            continue
        lat, vram = evaluate_dir(m)
        rows.append(
            {"model_dir": m, "latency_s": round(lat, 3), "vram_MB": round(vram, 1)}
        )
        print(f"DONE {m} lat={lat:.3f}s vram={vram:.1f}MB")

    # write CSV
    if rows:
        import csv

        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"DONE metrics -> {OUT_CSV}")


if __name__ == "__main__":
    main()
