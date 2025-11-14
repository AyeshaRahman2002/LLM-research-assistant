# src/rag/multiquery_expansion.py
import json
import re
from pathlib import Path

OUT = Path("results/rag_multiquery.json")

SEEDS = [
    ("speed up", ["accelerate", "optimize", "reduce latency"]),
    ("throughput", ["tokens per second", "toks/sec"]),
    ("quantization", ["int8", "int4", "bnb"]),
]


def expand(q: str):
    qs = {q}
    s = q.lower()
    for k, alts in SEEDS:
        if k in s:
            for a in alts:
                qs.add(re.sub(k, a, s))
    # generic rewrites
    qs.add(q + " practical techniques")
    qs.add(q + " trade-offs and limitations")
    return sorted(qs)


if __name__ == "__main__":
    base = "Methods to speed up LLM inference"
    out = {"base": base, "expansions": expand(base)}
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"DONE multiquery expansions -> {OUT}")
