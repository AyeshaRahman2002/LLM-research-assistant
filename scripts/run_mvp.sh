# scripts/run_mvp.sh
#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting MVP pipeline..."

# -------- 0) Ensure base folders --------
mkdir -p data results reports plots experiments data/uploads

# -------- 1) Literature Review --------
echo "==> Literature pipeline"
python3 -m src.literature.fetch_papers --query "LLM efficiency" --max 20
python3 -m src.literature.embed_cluster
python3 -m src.literature.summarize_papers --mode app
python3 -m src.eval.eval_summaries
python3 -m src.literature.plot_clusters
python3 -m src.literature.write_summary_md
echo "DONE Literature -> reports/literature_summary.md"

# -------- 2) Data Analysis (optional) --------
if [[ -f "data/sample.csv" ]]; then
  echo "==> Data analysis on data/sample.csv"
  python3 -m src.data_analysis.analyze_data --csv data/sample.csv
  python3 -m src.data_analysis.visualize_results --csv data/sample.csv
  python3 -m src.data_analysis.llm_commentary
  echo "DONE Data -> results/data_analysis_summary.json"
else
  echo "[skip] No data/sample.csv found (put a CSV there to enable)."
fi

# -------- 3) RAG quick eval (optional; requires literature items) --------
if [[ -f "data/literature_items.json" ]]; then
  echo "==> RAG quick eval"
  # Calm OpenMP / MKL on macOS to avoid FAISS/iomp/omp clashes
  export MKL_THREADING_LAYER=SEQUENTIAL
  export OMP_NUM_THREADS=1
  python3 -m src.rag.retrieval_pipeline || echo "[warn] RAG step failed; continuing."
  echo "DONE RAG -> results/rag_eval.csv (if step succeeded)"
else
  echo "[skip] No data/literature_items.json (run literature step first)."
fi

# -------- 4) LLM optimization (quant + eval + plot) --------
echo "==> LLM optimization (quantization & efficiency)"
python3 -m src.llm_opt.quantize_models || echo "[warn] quantize_models failed (bitsandbytes/GPU?). Continuing."
python3 -m src.llm_opt.evaluate_efficiency || echo "[warn] evaluate_efficiency failed. Continuing."
python3 -m src.llm_opt.compare_results || echo "[warn] compare_results failed. Continuing."
python3 -m src.llm_opt.plot_tradeoffs || echo "[warn] plot_tradeoffs failed. Continuing."

# -------- 5) Final report (MD + PDF) --------
echo "==> Final report"
python3 -m src.reporting.report_generator

echo "---------------------------------------------------------------"
echo "[DONE] MVP pipeline complete!"
echo "Open:"
echo " - Literature summary  -> reports/literature_summary.md"
echo " - Cluster plot        -> results/literature_clusters.png"
echo " - Data summary        -> results/data_analysis_summary.json (if run)"
echo " - RAG eval            -> results/rag_eval.csv (if run)"
echo " - Benchmarks          -> results/benchmark_table.csv (if run)"
echo " - Trade-off plot      -> plots/tradeoff_curves.png (if run)"
echo " - Final PDF           -> reports/final_report.pdf"
echo "---------------------------------------------------------------"
