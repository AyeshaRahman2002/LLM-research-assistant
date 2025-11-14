# scripts/run_llm_opt.sh
#!/usr/bin/env bash
set -euo pipefail
echo "==> LLM optimization suite"

mkdir -p experiments/quantized

# 1) Quantize (fp16 folder is a straight save of base model)
python3 -m src.llm_opt.quantize_models

# 2) Efficiency metrics (latency/VRAM)
python3 -m src.llm_opt.evaluate_efficiency

# 3) Merge with summary metrics (ROUGE/BERTScore) & plot trade-offs
python3 -m src.llm_opt.compare_results
python3 -m src.llm_opt.plot_tradeoffs

echo "DONE Results:"
echo " - results/efficiency_metrics.csv"
echo " - results/benchmark_table.csv"
echo " - plots/tradeoff_curves.png"
