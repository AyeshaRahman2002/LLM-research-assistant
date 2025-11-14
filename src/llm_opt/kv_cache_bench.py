# src/llm_opt/kv_cache_bench.py
"""
Benchmark: KV-cache on/off latency for decoder models.
Writes -> results/kv_cache_bench.json
"""
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    m = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

    prompt = "Large language models are transforming"
    inputs = tok(prompt, return_tensors="pt")

    t0 = time.time()
    _ = m.generate(**inputs, max_new_tokens=32, use_cache=True)
    t1 = time.time()
    _ = m.generate(**inputs, max_new_tokens=32, use_cache=False)
    t2 = time.time()

    result = {
        "model": model_name,
        "time_with_cache": round(t1 - t0, 3),
        "time_no_cache": round(t2 - t1, 3),
        "speedup_ratio": round((t2 - t1) / (t1 - t0 + 1e-9), 2),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/kv_cache_bench.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
