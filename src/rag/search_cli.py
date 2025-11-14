# src/rag/search_cli.py
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_DIR = Path("results/rag")

try:

    FAISS_OK = (RAG_DIR / "index.faiss").exists()
except Exception:
    FAISS_OK = False


@torch.no_grad()
def _encode(q: str) -> np.ndarray:
    """Encode a single query into an L2-normalized vector."""
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    enc = tok([q], return_tensors="pt", truncation=True, padding=True).to(dev)
    out = mdl(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1)
    mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
    mean = torch.nn.functional.normalize(mean, p=2, dim=1)
    return mean.cpu().numpy().astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    items = json.loads((RAG_DIR / "items.json").read_text())
    vecs = np.load(RAG_DIR / "doc_vecs.npy")
    qv = _encode(args.query)

    if FAISS_OK:
        import faiss

        index = faiss.read_index(str(RAG_DIR / "index.faiss"))
        D, I = index.search(qv, args.k)
        idxs = I[0]
        scores = D[0]
    else:
        sims = vecs @ qv[0]
        topk = int(min(args.k, len(sims)))
        idx = np.argpartition(-sims, topk - 1)[:topk]
        idxs = idx[np.argsort(-sims[idx])]
        scores = sims[idxs]

    print(f"\nTop-{args.k} results for: {args.query}\n")
    for i, s in zip(idxs, scores):
        print(f"({s:.3f}) {items[i]['title']}")
    print()


if __name__ == "__main__":
    main()
