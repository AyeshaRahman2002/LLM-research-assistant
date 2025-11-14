# src/rag/rag_eval_metrics.py
import json
import math
import re
from pathlib import Path

import numpy as np

RAG_DIR = Path("results/rag")
OUT = Path("results/rag_eval_metrics.json")

_word = re.compile(r"[A-Za-z0-9]+")


def tokenize(s: str):
    return set(w.lower() for w in _word.findall(s or ""))


def ndcg(rel, k):
    rel = rel[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rel))
    ideal = sorted(rel, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


if __name__ == "__main__":
    items = json.loads((RAG_DIR / "items.json").read_text())
    vecs = np.load(RAG_DIR / "doc_vecs.npy")
    # Load queries used by quick_eval (fallback to two defaults)
    qcsv = Path("results/rag_eval.csv")
    queries = []
    if qcsv.exists():
        import csv

        with qcsv.open() as f:
            r = csv.DictReader(f)
            for row in r:
                queries.append(row["query"])
    if not queries:
        queries = [
            "What are current methods to speed up LLM inference?",
            "How do diffusion LLMs differ from autoregressive ones?",
        ]

    # Build lightweight search (NumPy cosine)
    def search(qv, k=5):
        sims = vecs @ qv[0]
        topk = int(min(k, len(sims)))
        idx = np.argpartition(-sims, topk - 1)[:topk]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]

    # Encode on the fly with same trick as build_index
    import torch
    from transformers import AutoModel, AutoTokenizer

    EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()

    def enc1(q):
        with torch.no_grad():
            e = tok([q], return_tensors="pt", truncation=True, padding=True).to(dev)
            out = mdl(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1)
            mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            mean = torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy()
            return mean.astype("float32")

    results = []
    for q in queries:
        qv = enc1(q)
        idxs, sims = search(qv, k=5)
        # Relevance via token overlap between query and doc (title+summary)
        qtok = tokenize(q)
        rel = []
        for i in idxs:
            t = items[i]["title"] + " " + items[i]["summary"]
            rel.append(len(qtok.intersection(tokenize(t))))  # integer gain
        ndcg5 = ndcg(rel, 5)

        # Diversity: average pairwise cosine distance among top-k
        V = vecs[idxs]
        S = V @ V.T  # cosine since normalized
        n = len(V)
        if n > 1:
            tri = []
            for a in range(n):
                for b in range(a + 1, n):
                    tri.append(1.0 - float(S[a, b]))
            diversity = sum(tri) / len(tri)
        else:
            diversity = 0.0

        results.append(
            {"query": q, "nDCG@5": round(ndcg5, 4), "diversity": round(diversity, 4)}
        )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"DONE RAG metrics -> {OUT}")
