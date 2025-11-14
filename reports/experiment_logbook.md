# Experiment Logbook

> Keep a chronological, human-readable record of experiments.
> Each entry: goal → config → results → interpretation → next steps.

---

## 2025-11-12 — Example
**Goal:** Quick PEFT sanity
**Config:** `train_peft_run.py` (default), `batch=4`, `max_steps=100`
**Results:** ROUGE-L ↑, tokens/sec ~X
**Interpretation:** Stable; try larger rank
**Next:** Run `evaluate_efficiency.py` and `int8_vs_fp16_eval.py`
