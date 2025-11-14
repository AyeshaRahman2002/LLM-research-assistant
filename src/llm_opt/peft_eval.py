# src/llm_opt/peft_eval.py
import json
from pathlib import Path

import torch
from bert_score import score as bertscore
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CKPT = Path("experiments/peft_flan_t5/checkpoint-best")
DATA = Path("results/summaries.json")
OUT = Path("results/peft_eval.json")


@torch.no_grad()
def _gen(dir_, texts):
    tok = AutoTokenizer.from_pretrained(dir_)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(dir_)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev).eval()
    outs = []
    for t in texts:
        enc = tok([t], return_tensors="pt", truncation=True, max_length=1024).to(dev)
        ids = mdl.generate(**enc, max_new_tokens=128, num_beams=4)
        outs.append(tok.decode(ids[0], skip_special_tokens=True))
    return outs


if __name__ == "__main__":
    if not (CKPT.exists() and DATA.exists()):
        raise SystemExit("Need checkpoint and summaries.")
    data = json.loads(DATA.read_text())
    texts = [f"{d['title']}. {d['summary']}" for d in data]
    refs = [d["generated"] for d in data]
    hyps = _gen(str(CKPT), texts[: len(refs)])
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = sum(rs.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)) / len(
        refs
    )
    _, _, F = bertscore(hyps, refs, lang="en", verbose=False)
    OUT.write_text(
        json.dumps({"rougeL": float(rouge), "bertscore_f1": float(F.mean())}, indent=2)
    )
    print(f"DONE PEFT eval -> {OUT}")
