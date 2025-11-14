# src/rag/factuality_check.py
import argparse
import json
from pathlib import Path

from rouge_score import rouge_scorer

RAG_QA = Path("results/rag_qa.json")
RAG_DIR = Path("results/rag")
OUT = Path("results/rag_factuality.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(RAG_QA))
    ap.add_argument(
        "--threshold", type=float, default=0.2, help="ROUGE-L F1 threshold for warning"
    )
    args = ap.parse_args()

    qa = json.loads(Path(args.qa).read_text())
    items = json.loads((RAG_DIR / "items.json").read_text())
    cited_titles = [c["title"] for c in qa.get("citations", [])]
    ctx = "\n".join(d["summary"] for d in items if d["title"] in cited_titles) or ""

    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = (
        rs.score(ctx, qa["answer"])["rougeL"].fmeasure
        if ctx and qa.get("answer")
        else 0.0
    )
    payload = {
        "question": qa.get("question"),
        "rougeL_f1_ctx_vs_answer": round(float(score), 4),
        "threshold": args.threshold,
        "faithful": bool(score >= args.threshold),
        "citations": qa.get("citations", []),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"DONE factuality check -> {OUT}  (faithful={payload['faithful']})")


if __name__ == "__main__":
    main()
