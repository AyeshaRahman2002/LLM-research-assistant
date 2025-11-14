# src/literature/summarize_papers.py
# Pure PyTorch summarization (no TF/JAX, no pipeline)

import os

# Hard-disable TF & JAX BEFORE importing transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# reduce OpenMP noise on mac
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import torch
from src.utils.config import ensure_dirs, load_config
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_items():
    p = Path("data/literature_items.json")
    if not p.exists():
        raise SystemExit(
            "Missing data/literature_items.json. Run embed_cluster.py first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


@torch.no_grad()
def summarize_batch_torch(
    texts, model_name: str, max_new_tokens: int, batch_size: int = 4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        ).to(device)
        gen_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )
        outs = tok.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend(outs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["app", "research"],
        default="app",
        help="Generic profiles (no institution naming).",
    )
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    items = load_items()

    texts = [f"{it['title']}. {it['summary']}" for it in items[:20]]

    model_name = cfg["summarization"]["models"][0]
    t0 = time.time()
    summaries = summarize_batch_torch(
        texts, model_name=model_name, max_new_tokens=cfg["summarization"]["max_tokens"]
    )
    dt = time.time() - t0

    out = []
    for it, sm in zip(items[: len(summaries)], summaries):
        out.append({"title": it["title"], "summary": it["summary"], "generated": sm})

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/summaries.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    Path("results/metrics.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "model": model_name,
                "count": len(out),
                "latency_s": round(dt, 3),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"DONE summarized {len(out)} docs in {dt:.2f}s -> results/summaries.json")
