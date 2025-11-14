# scripts/run_all.sh
#!/usr/bin/env bash
# One-button pipeline for the whole project (uses .venv python explicitly)
set -euo pipefail

# ---- Always use the project venv python ----
PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[error] .venv not found. Run: make bootstrap"
  exit 1
fi
export PYTHONNOUSERSITE=1               # ignore user-site (keeps conda out)
source .venv/bin/activate               # ensure venv on PATH for tools like streamlit

export KMP_DUPLICATE_LIB_OK=TRUE
export MKL_THREADING_LAYER=SEQUENTIAL

# ---- Config you can edit if you want ----
# defaults
CSV=""; TARGET=""; UI=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --no-ui) UI=false; shift ;;
    *) echo "[warn] unknown arg: $1"; shift ;;
  esac
done

QUERY="LLM efficiency"
ARXIV_MAX=30
S2_LIMIT=30

# ---- Helpers ----
run() { echo ">>> $*"; "$@"; }
try() { echo ">>> (try) $*"; "$@" || echo "[warn] step failed: $* (continuing)"; }

# Calm backends
export TRANSFORMERS_NO_TF=1 TRANSFORMERS_NO_JAX=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "==> Ensuring base folders"
mkdir -p data results reports plots experiments data/uploads

# -------- 1) DATA ANALYSIS (optional) --------
if [[ -n "${CSV}" && -f "${CSV}" ]]; then
  echo "==> Data analysis on ${CSV}"
  if [[ -n "${TARGET}" ]]; then
    run "$PY" -m src.data_analysis.analyze_data --csv "${CSV}" --target "${TARGET}"
    try "$PY" -m src.data_analysis.visualize_results --csv "${CSV}" --target "${TARGET}"
  else
    run "$PY" -m src.data_analysis.analyze_data --csv "${CSV}"
    try "$PY" -m src.data_analysis.visualize_results --csv "${CSV}"
  fi
  try "$PY" -m src.data_analysis.insight_extract
  try "$PY" -m src.data_analysis.anomaly_detect --csv "${CSV}"
else
  echo "[skip] data analysis (no CSV configured)"
fi

# -------- 2) LITERATURE --------
echo "==> Literature fetching"
run "$PY" -m src.literature.fetch_papers --query "${QUERY}" --max "${ARXIV_MAX}"

if [[ -n "${SEMANTIC_SCHOLAR_API_KEY:-}" ]]; then
  try "$PY" -m src.literature.fetch_semantic_scholar --query "${QUERY}" --limit "${S2_LIMIT}"
else
  echo "[skip] Semantic Scholar (SEMANTIC_SCHOLAR_API_KEY not set)"
fi

run "$PY" -m src.literature.normalize_papers
run "$PY" -m src.literature.embed_cluster
run "$PY" -m src.literature.plot_clusters
try "$PY" -m src.literature.umap_projection
run "$PY" -m src.literature.summarize_papers --mode app
run "$PY" -m src.eval.eval_summaries
run "$PY" -m src.literature.write_summary_md
try "$PY" -m src.literature.cluster_summaries
try "$PY" -m src.literature.cite_link_injection
try "$PY" -m src.literature.save_bibtex

# -------- 3) RAG quick eval --------
try "$PY" -m src.rag.retrieval_pipeline
# Build reusable index + run CLI/QA/metrics
try "$PY" -m src.rag.build_index
try "$PY" -m src.rag.search_cli --query "LLM inference speedups" --k 5
try "$PY" -m src.rag.rag_qa --question "What are practical techniques to accelerate LLM decoding on commodity GPUs?" --k 5
try "$PY" -m src.rag.rag_eval_metrics
try "$PY" -m src.rag.factuality_check

# -------- 4) LLM OPT (quant + efficiency + plots) --------
echo "==> Quantization & efficiency"
try "$PY" -m src.llm_opt.quantize_models
try "$PY" -m src.llm_opt.evaluate_efficiency
try "$PY" -m src.llm_opt.compare_results
try "$PY" -m src.llm_opt.plot_tradeoffs
try "$PY" -m src.llm_opt.eval_tokens_per_sec
try "$PY" -m src.llm_opt.profiler_trace
try "$PY" -m src.llm_opt.batch_ablation

echo "==> PEFT training / eval / export"
try "$PY" -m src.llm_opt.train_peft_run
try "$PY" -m src.llm_opt.peft_eval
try "$PY" -m src.llm_opt.peft_export

# -------- 5) Final report --------
try "$PY" -m src.reporting.report_generator
try "$PY" -m src.reporting.insert_figures
try "$PY" -m src.reporting.table_export

# -------- 6) Launch UI --------
echo "==> Launching Streamlit UI at http://localhost:8501"
$UI && exec streamlit run ui/app.py --server.port 8501 --server.headless true
