# src/rag/rag_qa.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from src.utils.config import load_config
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

RAG_DIR = Path("results/rag")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path("results/rag_qa.json")

try:
    import faiss

    FAISS_OK = (RAG_DIR / "index.faiss").exists()
except Exception:
    FAISS_OK = False

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")


@torch.no_grad()
def _encode(texts):
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    embs = []
    for i in range(0, len(texts), 16):
        enc = tok(
            texts[i : i + 16], padding=True, truncation=True, return_tensors="pt"
        ).to(dev)
        out = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mean = torch.nn.functional.normalize(mean, p=2, dim=1)
        embs.append(mean.cpu().numpy())
    return np.vstack(embs).astype("float32")


def _search(qv, doc_vecs, k=5):
    if FAISS_OK:
        index = faiss.read_index(str(RAG_DIR / "index.faiss"))
        D, I = index.search(qv, k)
        return I[0], D[0]
    sims = doc_vecs @ qv[0]
    topk = int(min(k, len(sims)))
    idx = np.argpartition(-sims, topk - 1)[:topk]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    items = json.loads((RAG_DIR / "items.json").read_text())
    vecs = np.load(RAG_DIR / "doc_vecs.npy")
    qv = _encode([args.question])
    idxs, scores = _search(qv, vecs, k=args.k)

    ctx = "\n\n".join([f"- {items[i]['title']}: {items[i]['summary']}" for i in idxs])
    cfg = load_config()
    gen = cfg["summarization"]["models"][0]
    tok = AutoTokenizer.from_pretrained(gen)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(gen)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()

    prompt = (
        "Use only the context to answer the question in 5–8 sentences. "
        "Avoid fabricating unsupported claims.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {args.question}\n\nAnswer:"
    )
    with torch.no_grad():
        enc = tok([prompt], return_tensors="pt", truncation=True, max_length=1024).to(
            dev
        )
        out = mdl.generate(**enc, max_new_tokens=220, num_beams=4, early_stopping=True)
        answer = tok.decode(out[0], skip_special_tokens=True)

    payload = {
        "question": args.question,
        "answer": answer,
        "citations": [
            {"title": items[i]["title"], "score": float(s)}
            for i, s in zip(idxs, scores)
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE RAG QA -> {args.out}")


if __name__ == "__main__":
    main()
