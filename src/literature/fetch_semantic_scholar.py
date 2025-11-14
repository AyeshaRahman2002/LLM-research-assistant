# src/literature/fetch_semantic_scholar.py
import argparse
import json
import os
import time
from pathlib import Path

import requests

API = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch(query: str, limit: int = 50):
    headers = {}
    if k := os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = k
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,url,externalIds,publicationDate,venue,authors",
    }
    r = requests.get(API, params=params, headers=headers, timeout=30)
    if r.status_code == 429:
        time.sleep(2)
        r = requests.get(API, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="LLM efficiency")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="data/semanticscholar_raw.json")
    a = ap.parse_args()
    data = fetch(a.query, a.limit)
    Path(a.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"DONE Semantic Scholar -> {a.out} (n={len(data.get('data', []))})")
