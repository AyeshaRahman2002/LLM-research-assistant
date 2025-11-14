# src/literature/embed_cluster.py
# Pure PyTorch + Transformers (no sentence-transformers, no TF/JAX)

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_JAX"] = "1"

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # works with HF AutoModel


def parse_arxiv_xml(xml_text: str):
    titles = re.findall(r"<title>(.*?)</title>", xml_text, flags=re.DOTALL)
    summaries = re.findall(r"<summary>(.*?)</summary>", xml_text, flags=re.DOTALL)
    titles = titles[1:] if titles else []  # drop feed title
    items = []
    for i, t in enumerate(titles):
        s = summaries[i] if i < len(summaries) else ""
        items.append(
            {
                "title": re.sub(r"\s+", " ", t).strip(),
                "summary": re.sub(r"\s+", " ", s).strip(),
            }
        )
    return items


@torch.no_grad()
def encode_texts(texts):
    """
    Mean-pool last hidden states (standard ST approach) with attention mask.
    Returns L2-normalized embeddings [N, D].
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embs = []
    bs = 16
    for i in range(0, len(texts), bs):
        batch = texts[i : i + bs]
        enc = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model(**enc)
        last_hidden = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        masked = last_hidden * mask
        summed = masked.sum(dim=1)  # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1)  # [B, 1]
        mean_pooled = summed / lengths
        # L2-normalize
        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        embs.append(mean_pooled.cpu())
    return torch.cat(embs, dim=0).numpy()


def main():
    raw_path = Path("data/literature_raw.xml")
    if not raw_path.exists():
        raise SystemExit("Missing data/literature_raw.xml. Run fetch_papers.py first.")

    xml_text = raw_path.read_text(encoding="utf-8")
    items = parse_arxiv_xml(xml_text)
    if not items:
        raise SystemExit("No items parsed from arXiv feed. Try a different query.")

    texts = [f"{it['title']} — {it['summary']}" for it in items]
    emb = encode_texts(texts)  # np.ndarray [N, D]

    k = min(6, len(items))
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(emb)
    labels = km.labels_.tolist()

    pca2 = PCA(n_components=2, random_state=42).fit_transform(emb)

    Path("data/literature_items.json").write_text(
        json.dumps(items, indent=2), encoding="utf-8"
    )
    Path("results/literature_clusters.json").write_text(
        json.dumps({"labels": labels}, indent=2), encoding="utf-8"
    )
    np.save("plots/literature_pca.npy", pca2)
    print(f"DONE embedded {len(items)} docs and clustered into {k} groups")


if __name__ == "__main__":
    main()
