# src/literature/cluster_summaries.py
import json
from pathlib import Path

import torch
from src.utils.config import load_config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ITEMS = Path("data/literature_items.json")
LABS = Path("results/literature_clusters.json")
OUT = Path("reports/cluster_summaries.md")


@torch.no_grad()
def _gen(model, prompt):
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    enc = tok([prompt], return_tensors="pt", truncation=True, max_length=1024).to(dev)
    ids = mdl.generate(**enc, max_new_tokens=220, num_beams=4)
    return tok.decode(ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    if not (ITEMS.exists() and LABS.exists()):
        raise SystemExit("Need items & cluster labels.")
    items = json.loads(ITEMS.read_text())
    labels = json.loads(LABS.read_text())["labels"]
    cfg = load_config()
    model = cfg["summarization"]["models"][0]
    buckets = {}
    for it, y in zip(items, labels):
        buckets.setdefault(y, []).append(it)
    lines = ["# Cluster Summaries\n"]
    for k, docs in sorted(buckets.items()):
        ctx = "\n".join(f"- {d['title']}: {d['summary']}" for d in docs[:12])
        prompt = (
            "Summarize the common themes across these papers (6–8 sentences). "
            "Be cautious and avoid claims not supported by the items.\n" + ctx
        )
        lines.append(f"## Cluster {k}\n{_gen(model,prompt)}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"DONE cluster summaries -> {OUT}")
