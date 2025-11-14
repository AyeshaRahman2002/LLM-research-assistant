# scripts/run_rag.sh
#!/usr/bin/env bash
set -euo pipefail
echo "==> RAG pipeline"
if [[ ! -f "data/literature_items.json" ]]; then
  echo "Run the literature pipeline first (data/literature_items.json missing)."
  exit 1
fi
python3 -m src.rag.retrieval_pipeline
echo "DONE RAG eval -> results/rag_eval.csv"
