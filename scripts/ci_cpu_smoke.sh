# scripts/ci_cpu_smoke.sh
#!/usr/bin/env bash
set -euo pipefail

echo "==> CI CPU smoke run (no GPU, small inputs)"
export TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_JAX=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_THREADING_LAYER=SEQUENTIAL

# quick setup
mkdir -p data results reports plots experiments

# literature (tiny)
python3 -m src.literature.fetch_papers --query "LLM efficiency" --max 6
python3 -m src.literature.normalize_papers
python3 -m src.literature.embed_cluster
python3 -m src.literature.summarize_papers --mode app
python3 -m src.eval.eval_summaries
python3 -m src.literature.write_summary_md

# rag build + one QA (CPU path)
python3 -m src.rag.build_index
python3 -m src.rag.rag_qa --question "How to speed up LLM inference?" --k 3 || true
python3 -m src.rag.rag_eval_metrics || true

# reports
python3 -m src.reporting.report_generator
python3 -m src.reporting.table_export || true

echo "DONE CPU smoke passed."
