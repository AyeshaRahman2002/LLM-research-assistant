# scripts/run_efficiency_suite.sh
#!/usr/bin/env bash
set -euo pipefail
echo "==> Running LLM efficiency suite"
mkdir -p results

python3 -m src.llm_opt.quantize_models
python3 -m src.llm_opt.evaluate_efficiency
python3 -m src.llm_opt.compare_results
python3 -m src.llm_opt.plot_tradeoffs
python3 -m src.llm_opt.eval_tokens_per_sec
python3 -m src.llm_opt.batch_ablation
python3 -m src.llm_opt.profiler_trace
python3 -m src.llm_opt.system_resource_report
python3 -m src.llm_opt.kv_cache_bench
python3 -m src.llm_opt.flash_attn_check
python3 -m src.llm_opt.quantization_matrix_report
python3 -m src.llm_opt.triton_export
python3 -m src.llm_opt.distributed_eval
echo "DONE Efficiency suite completed."
