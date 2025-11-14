# Automated Literature Review and LLM Efficiency Evaluation Platform

The **LLM Research Assistant** is an integrated research automation system that streamlines the full lifecycle of academic and computational research.
It combines **literature retrieval**, **data analysis**, **survey processing**, **retrieval-augmented generation (RAG)**, **LLM optimization**, and **automated reporting** into one unified, extensible pipeline.

## Purpose

Modern research workflows often require manual coordination between tools for literature review, data analysis, model evaluation, and reporting.
This application automates those processes, allowing researchers to focus on insight generation rather than repetitive technical steps.

The system is designed to:

* Reduce time spent on data preparation and analysis
* Automatically synthesize and summarize research literature
* Clean and validate survey or experimental data
* Optimize and benchmark LLMs for efficiency
* Produce publication-ready reports with minimal manual editing
* Provide a visual dashboard for interactive exploration

## Aim

To develop a **modular, end-to-end research automation framework** that integrates LLM-based tools, analytics, and visualization into a reproducible, transparent, and efficient research assistant.

## Objectives

1. **Automate literature review** - fetch, embed, cluster, and summarize academic papers.
2. **Perform data ingestion and validation** - enforce schema, detect anomalies, and ensure data quality.
3. **Conduct statistical and model-based data analysis** - produce descriptive statistics, correlations, and predictive insights.
4. **Support LLM fine-tuning and efficiency optimization** - train lightweight adapters (PEFT), benchmark, and export optimized models.
5. **Enable retrieval-augmented reasoning (RAG)** - integrate document search and factual Q&A.
6. **Generate structured reports** - automatically compile findings into Markdown and PDF outputs.
7. **Provide a unified user interface** - through a Streamlit dashboard for interactive operation.
8. **Ensure extensibility** - allowing additional modules, pipelines, or domain-specific tasks to be added easily.

## System Overview

The application is organized into **eight structured phases**, each representing a logical component of the research workflow.

### **Phase 0 - Core Setup**

Establishes the environment, directory structure, dependencies, and baseline configuration.

**Components:**

* `requirements.txt`, `config.yaml`, `Dockerfile.cpu`
* Setup scripts (`bootstrap.sh`, `env_check.py`)
* Makefile automation for all phases
* Initial sanity tests and CI smoke checks

### **Phase 1 - Literature Automation**

Fetches, embeds, clusters, and summarizes research papers related to a topic.

**Key Modules:**

* `fetch_papers.py`, `fetch_semantic_scholar.py`, `fetch_crossref.py`
* `normalize_papers.py`, `embed_cluster.py`, `plot_clusters.py`
* `summarize_papers.py`, `eval_summaries.py`, `auto_review_report.py`

**Output:**
Structured literature summaries, topic clusters, and visualizations for review generation.

### **Phase 2 - Data Analysis Automation**

Processes datasets, computes statistical metrics, and generates visual insights.

**Key Modules:**

* `analyze_data.py` - descriptive stats, correlation, regression
* `visualize_results.py` - histograms, scatter plots, heatmaps
* `insight_extract.py`, `llm_commentary.py`, `anomaly_detect.py`
* `trend_forecast.py` - basic forecasting and trend detection

**Output:**
Interactive analytics, statistical summaries, and anomaly/trend detection reports.

### **Phase 3 - LLM Optimization & PEFT**

Implements model fine-tuning, quantization, and performance evaluation for large language models.

**Key Modules:**

* `train_peft.py`, `peft_eval.py`, `peft_export.py`
* `quantize_models.py`, `evaluate_efficiency.py`, `compare_results.py`
* `plot_tradeoffs.py`, `batch_ablation.py`, `profiler_trace.py`
* `system_resource_report.py`

**Output:**
Performance metrics, quantization results, and exportable fine-tuned LLMs.

### **Phase 4 - Retrieval-Augmented Generation (RAG)**

Builds and evaluates a retrieval-enhanced reasoning pipeline for contextual Q&A and document search.

**Key Modules:**

* `build_index.py`, `retrieval_pipeline.py`
* `rag_qa.py`, `rag_eval_metrics.py`
* `hybrid_reranker.py`, `multiquery_expansion.py`, `factuality_check.py`

**Output:**
RAG index files, QA logs, and factuality reports.

### **Phase 5 - Reporting & Documentation**

Generates structured reports, tables, and publication-ready summaries.

**Key Modules:**

* `report_generator.py`, `insert_figures.py`, `table_export.py`
* `reports/final_report.md` / `final_report.pdf`
* `reports/data_quality_notes.md` / `llm_efficiency_notes.md`

**Output:**
Complete Markdown + PDF report consolidating all pipeline outputs.

### **Phase 6 - Interactive Dashboard (UI)**

Streamlit-based interface for controlling all modules visually.

**Tabs Include:**

* **Survey Data** - Clean and validate survey responses
* **LLM Optimization** - Train, evaluate, and compare models
* **RAG Pipeline** - Build index and answer contextual questions
* **Efficiency Dashboard** - View tokens/sec and latency comparisons
* **Reports** - Export final PDF summaries

**Command:**

```bash
# Option 1 — Direct Streamlit launch
streamlit run ui/app.py

# Option 2 — via Makefile
make ui
```

Once it starts, open the URL printed in your terminal (typically http://localhost:8501) to explore the dashboard.

### **Phase 7 - Testing & Continuous Integration**

Automated tests for each major subsystem using Pytest and shell-based CI.

**Includes:**

* Unit and integration tests for data, LLM, and RAG components
* `ci_cpu_smoke.sh` for lightweight continuous testing
* `run_pytests.sh` for full test suite execution

**Command:**

```bash
make test
```

### **Phase 8 - Extension Modules**

Optional modular extensions for domain-specific research workflows.

**Included Extensions:**

* `csv_validators.py`, `data_quality_report.py`
* `survey_clean.py` - automatic Likert/MCQ cleaning
* `scheduler_stub.py` - experiment orchestration stub
* `write_abstracts.py`, `paper_formatter.py`, `bibliography_merge.py`
* Templates: `student_project_template.md`, `experiment_logbook.md`

These extensions demonstrate how the assistant can be adapted to different research domains such as psychology, NLP efficiency, or social sciences.

## Running the Full System

To execute the entire end-to-end pipeline:

```bash
# 1. Set up the environment
make bootstrap

# 2. Run the full research pipeline
make all CSV=data/sample.csv
```

This will:

1. Validate and clean your dataset
2. Run literature fetching and summarization
3. Build RAG indexes and perform factual QA
4. Train and evaluate LLM fine-tuning (PEFT)
5. Generate final Markdown + PDF reports
6. Store outputs under `results/` and `reports/`

## Running via Docker

```bash
make cpu-docker
docker run -it --rm -p 8501:8501 llm-research-assistant:cpu
```

## Output Summary

| Directory       | Description                                         |
| --------------- | --------------------------------------------------- |
| `results/`      | Intermediate outputs, JSON metrics, validation logs |
| `reports/`      | Final Markdown and PDF reports                      |
| `plots/`        | Visual charts and figures                           |
| `data/uploads/` | Cleaned survey data                                 |
| `experiments/`  | Experiment records and metadata                     |

## License

Developed for academic and applied research automation.
Open for extension, adaptation, and integration with other AI workflows.
