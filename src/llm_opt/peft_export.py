# src/llm_opt/peft_export.py
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CKPT = Path("experiments/peft_flan_t5/checkpoint-best")
BASE = "google/flan-t5-small"
OUT = Path("experiments/peft_flan_t5/merged")

if __name__ == "__main__":
    if not CKPT.exists():
        raise SystemExit("Checkpoint not found.")
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    model = PeftModel.from_pretrained(base, CKPT).merge_and_unload()
    OUT.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(OUT)
    model.save_pretrained(OUT)
    print(f"DONE merged -> {OUT}")
