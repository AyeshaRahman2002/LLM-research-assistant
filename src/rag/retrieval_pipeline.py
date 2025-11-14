# src/rag/retrieval_pipeline.py
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from src.utils.config import load_config
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

# Try FAISS, but fall back to NumPy if it isn’t available or crashes
TRY_FAISS = True
FAISS_OK = False
if TRY_FAISS:
    try:
        import faiss

        FAISS_OK = True
    except Exception:
        FAISS_OK = False

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ITEMS = Path("data/literature_items.json")
IDX_DIR = Path("results/rag")
IDX_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = Path("results/rag_eval.csv")


# Embedding
@torch.no_grad()
def _encode(texts):
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    embs = []
    bs = 16
    for i in range(0, len(texts), bs):
        enc = tok(
            texts[i : i + bs], padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        out = mdl(**enc).last_hidden_state  # [B,T,H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B,T,1]
        mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        mean = torch.nn.functional.normalize(mean, p=2, dim=1)
        embs.append(mean.cpu().numpy())
    return np.vstack(embs).astype("float32")


# Index (FAISS or NumPy)
class Retriever:
    def __init__(self, docs, doc_vecs):
        self.docs = docs
        self.doc_vecs = doc_vecs
        self.use_faiss = False
        if FAISS_OK:
            try:
                index = faiss.IndexFlatIP(doc_vecs.shape[1])
                index.add(doc_vecs)
                self.index = index
                self.use_faiss = True
            except Exception:
                self.use_faiss = False

    def search(self, qvec, k=5):
        if self.use_faiss:
            D, I = self.index.search(qvec.astype("float32"), k)
            return I[0], D[0]
        # NumPy cosine similarity fallback
        sims = self.doc_vecs @ qvec[0]
        topk = int(min(k, len(sims)))
        idx = np.argpartition(-sims, topk - 1)[:topk]
        idx = idx[np.argsort(-sims[idx])]
        return idx, sims[idx]


# RAG Summarize
def rag_summarize(query, retriever, gen_model_name, k=5):
    qv = _encode([query])
    idxs, _ = retriever.search(qv, k=k)
    ctx = "\n\n".join(
        [
            f"- {retriever.docs[i]['title']}: {retriever.docs[i]['summary']}"
            for i in idxs
        ]
    )

    tok = AutoTokenizer.from_pretrained(gen_model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    prompt = (
        "Use only the following context to write a concise synthesis (6–8 sentences). "
        "Avoid claims not supported by the context.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    enc = tok([prompt], return_tensors="pt", truncation=True, max_length=1024).to(
        device
    )
    out = mdl.generate(**enc, max_new_tokens=220, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True), [
        retriever.docs[i]["title"] for i in idxs
    ]


# Driver
def quick_eval():
    if not ITEMS.exists():
        raise SystemExit(
            "Run literature pipeline first to create data/literature_items.json"
        )

    items = json.loads(ITEMS.read_text())
    texts = [f"{it['title']} — {it['summary']}" for it in items]
    doc_vecs = _encode(texts)
    retr = Retriever(items, doc_vecs)

    cfg = load_config()
    gen_name = cfg["summarization"]["models"][0]

    queries = [
        "What are current methods to speed up LLM inference?",
        "How do diffusion LLMs differ from autoregressive ones?",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "num_ctx", "snippet"])
        for q in queries:
            ans, titles = rag_summarize(q, retr, gen_name, k=5)
            w.writerow([q, len(titles), ans[:220].replace("\n", " ") + "…"])

    print(
        f"DONE RAG eval -> {OUT_CSV}  (backend: {'FAISS' if retr.use_faiss else 'NumPy'})"
    )


if __name__ == "__main__":
    quick_eval()
