# src/llm_opt/quantize_models.py
import os
from pathlib import Path

import torch
from src.utils.config import load_config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
OUT = Path("experiments/quantized")
OUT.mkdir(parents=True, exist_ok=True)


def _bnb_ok():
    try:

        return True
    except Exception:
        return False


def load_model(name: str, quant: str):
    has_cuda = torch.cuda.is_available()
    bnb = _bnb_ok()
    print(f"[info] quant={quant} cuda={has_cuda} bnb={bnb}")
    tok = AutoTokenizer.from_pretrained(name)

    if quant in ("int8", "int4") and has_cuda and bnb:
        from transformers import BitsAndBytesConfig

        cfg = BitsAndBytesConfig(
            load_in_8bit=(quant == "int8"),
            load_in_4bit=(quant == "int4"),
            bnb_4bit_compute_dtype=torch.float16,
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            name, quantization_config=cfg, device_map="auto"
        )
        return tok, mdl, f"{quant}-bnb"

    if quant in ("int8", "int4"):
        print("[warn] falling back to CPU dynamic quantization.")
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to("cpu").eval()
        mdl = torch.quantization.quantize_dynamic(
            mdl, {torch.nn.Linear}, dtype=torch.qint8
        )
        return tok, mdl, f"{quant}-cpu"

    dtype = torch.float16 if has_cuda else torch.float32
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=dtype)
    if has_cuda:
        mdl = mdl.to("cuda")
    return tok, mdl, ("fp16" if has_cuda else "fp32")


if __name__ == "__main__":
    base = load_config()["summarization"]["models"][0]
    has_cuda = torch.cuda.is_available()
    for q in [None, "int8", "int4"]:
        if q in ("int8", "int4") and not has_cuda:
            print(f"[skip] {q} quantization on CPU-only host")
            continue
        tok, mdl, tag = load_model(base, q or "fp")
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(OUT / tag)
        mdl.save_pretrained(OUT / tag)
        print(f"DONE saved {tag} -> {OUT/tag}")
