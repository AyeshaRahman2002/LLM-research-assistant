# src/data_analysis/llm_commentary.py
# Generates a short narrative using your summarization model
import os
from pathlib import Path

import pandas as pd
import torch
from src.utils.config import load_config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RESULTS_DIR = Path("results")
OUT_MD = RESULTS_DIR / "data_analysis_commentary.md"

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


def load_snippets():
    parts = []
    s = RESULTS_DIR / "data_analysis_summary.json"
    if s.exists():
        parts.append(s.read_text())
    for f in [
        "descriptive_stats.csv",
        "correlations.csv",
        "ttest_results.csv",
        "regression_summary.json",
    ]:
        p = RESULTS_DIR / f
        if p.exists():
            parts.append(
                f"\n==== {f} (head) ====\n"
                + pd.read_csv(p).head(10).to_csv(index=False)
                if f.endswith(".csv")
                else "\n==== regression_summary.json ====\n" + p.read_text()
            )
    return "\n".join(parts)[:4000]  # keep prompt small


@torch.no_grad()
def generate(text, model_name: str, max_new_tokens=220):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    enc = tok(
        [text], padding=True, truncation=True, max_length=1024, return_tensors="pt"
    ).to(device)
    out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=4)
    return tok.decode(out_ids[0], skip_special_tokens=True)


def main():
    cfg = load_config()
    model_name = cfg["summarization"]["models"][0]
    prompt = (
        "You are a data analyst. Read the snippets and write 1–2 concise paragraphs "
        "summarizing key patterns (not raw stats), correlations worth noting, and any cautions "
        "about data quality or sample size. Be precise and measured.\n\n"
        + load_snippets()
    )
    commentary = generate(prompt, model_name)
    OUT_MD.write_text("# Data Analysis Commentary\n\n" + commentary, encoding="utf-8")
    print(f"DONE commentary -> {OUT_MD}")


if __name__ == "__main__":
    main()
