# src/literature/umap_projection.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ITEMS = Path("data/literature_items.json")
OUT = np
OUT_NPY = Path("plots/literature_umap.npy")
OUT_PNG = Path("results/literature_umap.png")


def _encode(texts):
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    mdl = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    with torch.no_grad():
        embs = []
        for i in range(0, len(texts), 16):
            enc = tok(
                texts[i : i + 16], padding=True, truncation=True, return_tensors="pt"
            ).to(dev)
            out = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(torch.nn.functional.normalize(mean, p=2, dim=1).cpu().numpy())
    return np.vstack(embs)


if __name__ == "__main__":
    if not ITEMS.exists():
        raise SystemExit("Missing data/literature_items.json")
    X = _encode(
        [f"{it['title']} — {it['summary']}" for it in json.loads(ITEMS.read_text())]
    )
    try:
        import umap
    except Exception:
        print("[skip] umap-learn not installed; skipping.")
        raise SystemExit(0)
    U = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, U)
    plt.figure(figsize=(8, 6))
    plt.scatter(U[:, 0], U[:, 1], s=35, alpha=0.85)
    plt.title("UMAP Projection (MiniLM)")
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=160)
    print(f"DONE UMAP -> {OUT_NPY}, {OUT_PNG}")
