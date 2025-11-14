# src/rag/build_index.py
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ITEMS = Path("data/literature_items.json")
OUT_DIR = Path("results/rag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_OK = False
try:
    import faiss

    FAISS_OK = True
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


if __name__ == "__main__":
    if not ITEMS.exists():
        raise SystemExit(
            "Run literature pipeline first (data/literature_items.json missing)."
        )
    items = json.loads(ITEMS.read_text())
    texts = [f"{it['title']} — {it['summary']}" for it in items]
    vecs = _encode(texts)
    np.save(OUT_DIR / "doc_vecs.npy", vecs)
    (OUT_DIR / "items.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    if FAISS_OK:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        faiss.write_index(index, str(OUT_DIR / "index.faiss"))
        backend = "FAISS"
    else:
        backend = "NumPy"

    (OUT_DIR / "meta.json").write_text(
        json.dumps({"backend": backend, "n_docs": len(items)}, indent=2)
    )
    print(f"DONE built index -> {OUT_DIR}  (backend: {backend}, n={len(items)})")
