# src/llm_opt/profiler_trace.py
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "experiments/quantized/fp32"
OUT = Path("results/torch_profile.json")

if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained(MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    enc = tok(["Explain LoRA in one paragraph."], return_tensors="pt").to(dev)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
        + ([torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            _ = mdl.generate(**enc, max_new_tokens=128)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(prof.key_averages().table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace(str(OUT.with_suffix(".trace.json")))
    print(f"DONE profiler trace -> {OUT} & {OUT.with_suffix('.trace.json')}")
