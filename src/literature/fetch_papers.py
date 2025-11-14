# src/literature/fetch_papers.py
import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from src.utils.config import ensure_dirs, load_config


def fetch_arxiv(
    query: str, max_results: int = 50, retries: int = 5, backoff: int = 10
) -> str:
    """
    Fetch papers from the arXiv API with simple retry logic for rate limits (HTTP 429).
    Retries up to `retries` times, doubling the wait each time.
    """
    base = os.getenv("ARXIV_BASE_URL", "https://export.arxiv.org/api/query")
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{base}?{urlencode(params)}"

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 429:
                wait = backoff * attempt
                print(f"[warn] arXiv rate-limited (HTTP 429), retrying in {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            wait = backoff * attempt
            print(
                f"[warn] fetch attempt {attempt}/{retries} failed: {e}; retrying in {wait}s..."
            )
            time.sleep(wait)

    raise SystemExit(f"[error] Failed to fetch arXiv data after {retries} retries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="LLM efficiency")
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--out", type=str, default="data/literature_raw.xml")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)

    xml_text = fetch_arxiv(args.query, args.max)
    Path(args.out).write_text(xml_text, encoding="utf-8")
    print(f"DONE saved arXiv feed -> {args.out}")
