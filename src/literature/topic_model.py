# src/literature/topic_model.py
import json
import re
from pathlib import Path

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

IN = Path("data/literature_items.json")
OUT_JSON = Path("results/lda_topics.json")


def clean(s):
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())


if __name__ == "__main__":
    items = json.loads(IN.read_text())
    texts = [clean(it["title"] + " " + it["summary"]) for it in items]
    vec = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    X = vec.fit_transform(texts)
    if X.shape[0] < 5 or X.shape[1] < 10:
        OUT_JSON.write_text(json.dumps({"topics": []}, indent=2))
        print("[skip] not enough data")
        exit(0)
    lda = LatentDirichletAllocation(
        n_components=min(6, X.shape[0] // 2 or 1), random_state=42
    )
    lda.fit(X)
    words = np.array(vec.get_feature_names_out())
    topics = []
    for k, comp in enumerate(lda.components_):
        idx = np.argsort(-comp)[:12]
        topics.append({"topic": int(k), "top_terms": words[idx].tolist()})
    OUT_JSON.write_text(json.dumps({"topics": topics}, indent=2))
    print(f"DONE LDA topics -> {OUT_JSON}")
