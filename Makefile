# LLM Research Assistant - Makefile
# Usage:
#   make help                 # discover targets
#   make bootstrap            # create .venv, install deps, sanity checks
#   make mvp                  # run minimal end-to-end pipeline
#   make literature           # run literature-only pipeline
#   make rag                  # build index + run a sample QA
#   make peft-train           # quick LoRA training
#   make llmopt               # run efficiency/quant dashboards
#   make report               # generate final report + tables
#   make ui                   # launch Streamlit
#   make test                 # run pytest suite (green already!)
#   make ci-smoke             # CPU-only smoke
#   make clean                # clean artifacts (keeps dirs)
#   make clean-deep           # really clean (removes venv, caches)
#
# set CSV=path/to.csv to override the default CSV for data steps.
# e.g., make all CSV=data/sample.csv

# Default goal
.DEFAULT_GOAL := help

# Configuration
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eo pipefail -c

# Try using repo venv if it exists, otherwise fallback to system python.
VENV_PY := ./.venv/bin/python
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)

STREAMLIT := $(if $(wildcard ./.venv/bin/streamlit),./.venv/bin/streamlit,streamlit)

# Inputs / Outputs
CSV ?= data/sample.csv
TARGET ?= target_column
SURVEY_CSV ?= $(CSV)
LIKERT_MAP ?= "strongly disagree:1,disagree:2,neutral:3,agree:4,strongly agree:5,1:1,2:2,3:3,4:4,5:5"
MCQ_COLS ?= "gender,education,preferred_model,uses_ai_tools"

# Paths
RESULTS_DIR := results
REPORTS_DIR := reports
PLOTS_DIR := plots
DATA_DIR := data
UPLOADS_DIR := $(DATA_DIR)/uploads
CONFIGS_DIR := configs

# Common flags
QUIET_WARN := || echo "[warn] continuing despite failure"
SKIP_WARN  := || echo "[skip] optional step unavailable"

# Phony targets
.PHONY: help bootstrap mvp ui clean clean-deep literature rag build-index rag-qa \
        peft-train peft-eval peft-export llmopt efficiency kv-bench flash-check \
        triton-export quant-report report report-appendix tables publication-tables \
        test test-reports ci-smoke cpu-docker env-check schema validate dq \
        survey-clean survey-clean-with-map run-all all print-vars

# Help
help:
	@echo ""
	@echo "LLM Research Assistant - Make targets"
	@echo "-------------------------------------"
	@awk 'BEGIN {FS = ":.*?#"} /^[a-zA-Z0-9_.-]+:.*?#/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# Setup & Env
bootstrap: # create venv, install deps, sanity checks
	./scripts/bootstrap.sh

env-check: # dump environment (CUDA/torch/faiss/bitsandbytes) -> results/env_report.json
	$(PY) -m src.env_check

# Quick E2E / UI
mvp: # run a minimal, CPU-friendly end-to-end pipeline
	bash scripts/run_mvp.sh

ui: # launch Streamlit on 8501
	$(STREAMLIT) run ui/app.py --server.port=8501 --server.headless true

# Housekeeping
clean: # remove artifacts but keep directories in place
	rm -rf __pycache__ src/**/__pycache__ */__pycache__
	rm -rf $(RESULTS_DIR)/* $(PLOTS_DIR)/* $(REPORTS_DIR)/* $(UPLOADS_DIR) experiments/*
	mkdir -p $(RESULTS_DIR) $(PLOTS_DIR) $(REPORTS_DIR) $(UPLOADS_DIR) experiments
	@echo "[OK] Cleaned build directories."

clean-deep: # nuke caches, venv, build outputs - CAUTION
	rm -rf __pycache__ src/**/__pycache__ */__pycache__ .pytest_cache .mypy_cache
	rm -rf $(RESULTS_DIR) $(PLOTS_DIR) $(REPORTS_DIR) $(UPLOADS_DIR) experiments
	rm -rf .venv
	@echo "[OK] Deep cleaned (including .venv)."

# Phase 1 - Literature
literature: # end-to-end literature pipeline (robust to rate limits)
	@echo "==> Running Literature Pipeline"
	-$(PY) -m src.literature.fetch_papers --query "LLM efficiency" --max 30 $(QUIET_WARN)
	$(PY) -m src.literature.normalize_papers
	$(PY) -m src.literature.embed_cluster
	$(PY) -m src.literature.plot_clusters
	-$(PY) -m src.literature.umap_projection $(SKIP_WARN)
	$(PY) -m src.literature.summarize_papers --mode app
	$(PY) -m src.eval.eval_summaries
	$(PY) -m src.literature.write_summary_md
	$(PY) -m src.literature.cluster_summaries
	$(PY) -m src.literature.cite_link_injection
	-$(PY) -m src.literature.auto_review_report $(QUIET_WARN)
	@echo "[OK] Literature pipeline complete."

# Phase 2 - Data / Survey helpers
schema: # infer schema JSON from CSV -> configs/survey_schema.json
	@mkdir -p $(CONFIGS_DIR)
	./scripts/infer_schema.py $(CSV) $(CONFIGS_DIR)/survey_schema.json

validate: schema # validate CSV against schema -> results/validation_report.json
	$(PY) -m src.data_ingest.csv_validators --csv $(CSV) --schema-json $(CONFIGS_DIR)/survey_schema.json --out $(RESULTS_DIR)/validation_report.json

dq: # data quality profile -> JSON + Markdown
	$(PY) -m src.data_ingest.data_quality_report --csv $(CSV) --out-json $(RESULTS_DIR)/data_quality_report.json --out-md $(REPORTS_DIR)/data_quality_notes.md

survey-clean: # clean survey with default inline Likert mapping
	@mkdir -p $(UPLOADS_DIR) $(RESULTS_DIR)
	$(PY) -m src.surveys.survey_clean \
		--csv $(SURVEY_CSV) \
		--likert-cols satisfaction_score \
		--mcq-cols "$(MCQ_COLS)" \
		--likert-map $(LIKERT_MAP) \
		--out-csv $(UPLOADS_DIR)/survey_cleaned.csv \
		--out-meta $(RESULTS_DIR)/survey_clean_log.json

survey-clean-with-map: # clean survey with external map file (set MAP=path.json)
	@if [ -z "$(MAP)" ]; then echo "Usage: make survey-clean-with-map MAP=configs/likert_map.json"; exit 1; fi
	@mkdir -p $(UPLOADS_DIR) $(RESULTS_DIR)
	$(PY) -m src.surveys.survey_clean \
		--csv $(SURVEY_CSV) \
		--likert-cols satisfaction_score \
		--mcq-cols "$(MCQ_COLS)" \
		--likert-map $(MAP) \
		--out-csv $(UPLOADS_DIR)/survey_cleaned.csv \
		--out-meta $(RESULTS_DIR)/survey_clean_log.json

# Phase 3 - PEFT / LLM Opt
peft-train: # quick LoRA training
	$(PY) -m src.llm_opt.train_peft_run

peft-eval: # evaluate PEFT checkpoint
	$(PY) -m src.llm_opt.peft_eval

peft-export: # merge LoRA into base model folder
	$(PY) -m src.llm_opt.peft_export

kv-bench: # kv-cache on/off micro-bench (CPU-safe)
	$(PY) -m src.llm_opt.kv_cache_bench

flash-check: # check FlashAttention availability and shapes
	$(PY) -m src.llm_opt.flash_attn_check

efficiency: # run efficiency suite incrementally
	$(PY) -m src.llm_opt.quantize_models
	$(PY) -m src.llm_opt.evaluate_efficiency
	$(PY) -m src.llm_opt.compare_results
	$(PY) -m src.llm_opt.plot_tradeoffs
	$(PY) -m src.llm_opt.eval_tokens_per_sec
	$(PY) -m src.llm_opt.batch_ablation
	$(PY) -m src.llm_opt.profiler_trace
	-$(PY) -m src.llm_opt.int8_vs_fp16_eval $(QUIET_WARN)
	-$(PY) -m src.llm_opt.system_resource_report $(QUIET_WARN)

triton-export: # export tiny GPT2 TorchScript (for Triton-like serving)
	$(PY) -m src.llm_opt.triton_export

quant-report: # quantization error report -> results/quantization_matrix_report.json
	$(PY) -m src.llm_opt.quantization_matrix_report

llmopt: efficiency triton-export quant-report # one-shot bundle target

# Phase 4 - RAG
build-index: # build FAISS/NumPy index from literature items
	$(PY) -m src.rag.build_index

rag-qa: build-index # run a sample RAG QA and factuality check
	$(PY) -m src.rag.rag_qa --question "How to speed up LLM inference?" --k 3 $(QUIET_WARN)
	-$(PY) -m src.rag.rag_eval_metrics $(QUIET_WARN)
	-$(PY) -m src.rag.factuality_check --qa results/rag_qa.json $(QUIET_WARN)
	@echo "[OK] RAG QA complete."

rag: build-index rag-qa # shorthand combo

# Phase 5 - Reporting
report: # build final MD/PDF and export tables
	$(PY) -m src.reporting.report_generator $(QUIET_WARN)
	$(PY) -m src.reporting.table_export $(QUIET_WARN)
	@echo "[OK] Report generated: $(REPORTS_DIR)/final_report.pdf"

report-appendix: # generate appendix MD if your script supports it
	@echo "[info] If you add an appendix builder, wire it here."

tables: # lightweight pub tables (dummy -> CSV/TeX)
	./scripts/run_publication_tables.py

publication-tables: tables # alias

# Phase 6 - Docker
cpu-docker:
	docker build -f Dockerfile.cpu -t llm-research-assistant:cpu .

# Phase 7 - Tests & CI
test:
	./scripts/run_pytests.sh

test-reports: # validate report generation in CI
	./scripts/test_reports.sh

ci-smoke: # CPU-only smoke for CI
	./scripts/ci_cpu_smoke.sh

# Orchestration
run-all:
	@echo "==> Full pipeline: literature -> RAG -> efficiency -> report (CSV=$(CSV))"
	$(MAKE) literature
	$(MAKE) rag
	$(MAKE) llmopt
	$(MAKE) report

all: run-all # default full run

# Debug helpers
print-vars:
	@echo "PY=$(PY)"
	@echo "STREAMLIT=$(STREAMLIT)"
	@echo "CSV=$(CSV)"
	@echo "SURVEY_CSV=$(SURVEY_CSV)"
	@echo "LIKERT_MAP=$(LIKERT_MAP)"
	@echo "MCQ_COLS=$(MCQ_COLS)"
