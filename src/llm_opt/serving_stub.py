# src/llm_opt/serving_stub.py
"""
Stub: serve model via CLI or local API (mock).
Logs model load + sample inference.
"""
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


def serve(model_name="sshleifer/tiny-gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    t0 = time.time()
    out = model.generate(**tok("hello", return_tensors="pt"), max_new_tokens=5)
    text = tok.decode(out[0])
    elapsed = round(time.time() - t0, 3)
    log = {"model": model_name, "elapsed": elapsed, "output": text}
    Path("results").mkdir(exist_ok=True)
    Path("results/serving_stub.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    serve()
