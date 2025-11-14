# src/rag/hybrid_reranker.py
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .build_index import _encode  # re-use encoder

ITEMS = Path("data/literature_items.json")
OUT = Path("results/rag/hybrid_search_demo.json")

if __name__ == "__main__":
    items = json.loads(ITEMS.read_text())
    texts = [f"{it['title']} {it['summary']}" for it in items]

    # embeddings
    E = _encode(texts)  # (N, D) L2-normalized
    # tfidf (cosine)
    tf = TfidfVectorizer(stop_words="english").fit(texts)
    M = tf.transform(texts)  # sparse

    def search(q, alpha=0.6, k=5):
        qv = _encode([q])  # (1,D)
        emb = E @ qv[0]  # cosine
        tsv = tf.transform([q])
        tfc = (M @ tsv.T).toarray().ravel()
        score = alpha * emb + (1 - alpha) * tfc / (np.linalg.norm(tsv.data) + 1e-9)
        idx = np.argsort(-score)[:k]
        return [{"title": items[i]["title"], "score": float(score[i])} for i in idx]

    demo = {
        "query": "Methods to speed up LLM inference",
        "results": search("Methods to speed up LLM inference", alpha=0.6, k=5),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(demo, indent=2))
    print(f"DONE hybrid search demo -> {OUT}")
