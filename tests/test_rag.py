# tests/test_rag.py
from src.rag.rag_eval_metrics import ndcg, tokenize


def test_tokenize_basic():
    s = "Hello, world! tokens: 1, 2, 3."
    toks = tokenize(s)
    assert {"hello", "world", "tokens", "1", "2", "3"} <= toks


def test_ndcg_monotonicity():
    # Perfect ranking vs. reversed
    rel = [3, 2, 1, 0]
    assert ndcg(rel, 3) > ndcg(list(reversed(rel)), 3)
