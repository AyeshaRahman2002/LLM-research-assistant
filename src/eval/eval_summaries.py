# src/eval/eval_summaries.py
import json
from pathlib import Path

from bert_score import score as bertscore
from rouge_score import rouge_scorer


def main():
    p = Path("results/summaries.json")
    if not p.exists():
        raise SystemExit(
            "Missing results/summaries.json. Run summarize_papers.py first."
        )
    data = json.loads(p.read_text(encoding="utf-8"))

    refs = [d["summary"] for d in data]
    hyps = [d["generated"] for d in data]

    # ROUGE-L
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_vals = [rs.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)]
    rougeL = sum(rouge_vals) / len(rouge_vals) if rouge_vals else 0.0

    # BERTScore
    P, R, F = bertscore(hyps, refs, lang="en", verbose=True)
    bert_f1 = float(F.mean())

    summary = {"rougeL": round(rougeL, 4), "bertscore_f1": round(bert_f1, 4)}
    Path("results/summary_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"DONE metrics -> results/summary_metrics.json\n{summary}")


if __name__ == "__main__":
    main()
