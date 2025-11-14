# ui/app.py
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# CSV loading that tolerates messy data and fixes dtypes
BOOL_TRUE = {"yes", "y", "true", "t", "1"}
BOOL_FALSE = {"no", "n", "false", "f", "0"}


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    # Normalize common truthy/falsey tokens -> pandas nullable boolean
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map(
            lambda x: True if x in BOOL_TRUE else (False if x in BOOL_FALSE else pd.NA)
        )
        .astype("boolean")
    )


def load_csv_arrow_safe(path: Path) -> pd.DataFrame:
    # Read leniently, keep empty strings as NaN
    df = pd.read_csv(
        path,
        na_values=["", "na", "n/a", "null", "None", "NaN"],
        keep_default_na=True,
        dtype=str,  # start as strings to avoid mixed object mess
        encoding_errors="ignore",
    )

    # Best-effort column-specific coercions by header name (if present)
    for col in df.columns:
        lc = col.lower()
        if lc in {"age", "satisfaction_score"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif lc in {"income_usd"}:
            df[col] = (
                df[col]
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif lc in {"completion_rate"}:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Float64")
        elif "date" in lc:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        elif lc in {"uses_ai_tools", "uses_ai", "ai_user"}:
            df[col] = _coerce_bool_series(df[col])
        elif lc.endswith("_id") or lc == "id":
            df[col] = df[col].astype("string")
        else:
            # Try numbers -> otherwise keep as pandas StringDtype
            coerced = pd.to_numeric(df[col], errors="ignore")
            if isinstance(coerced, pd.Series) and pd.api.types.is_numeric_dtype(
                coerced
            ):
                # make them nullable numeric so Arrow is happy
                if pd.api.types.is_integer_dtype(coerced):
                    df[col] = coerced.astype("Int64")
                elif pd.api.types.is_float_dtype(coerced):
                    df[col] = coerced.astype("Float64")
                else:
                    df[col] = coerced
            else:
                df[col] = df[col].astype("string")

    # keep NumPy-backed nullable dtypes (avoids PyArrow datetime reductions)
    df = df.convert_dtypes()

    return df


st.set_page_config(page_title="LLM Research Assistant", layout="wide")
st.title("LLM-Powered Research Assistant")

# Global Controls
st.sidebar.header("Controls")
model_name = st.sidebar.text_input("Summarization model", "google/flan-t5-small")
max_tokens = st.sidebar.number_input("Max new tokens", 32, 512, 128, step=8)
batch_size = st.sidebar.number_input("Batch size (summaries)", 1, 16, 4, step=1)
rag_topk = st.sidebar.number_input("RAG top-k", 1, 20, 5, step=1)

# env vars for downstream scripts that read config/model at runtime
os.environ["LLM_UI_MODEL_OVERRIDE"] = model_name
os.environ["LLM_UI_MAX_TOKENS"] = str(max_tokens)
os.environ["LLM_UI_BATCH"] = str(batch_size)
os.environ["LLM_UI_RAG_TOPK"] = str(rag_topk)


def run_cmd(args):
    st.write("```bash\n" + " ".join(args) + "\n```")
    try:
        cp = subprocess.run(args, check=True, capture_output=True, text=True)
        if cp.stdout.strip():
            st.code(cp.stdout, language="bash")
        if cp.stderr.strip():
            st.caption("stderr:")
            st.code(cp.stderr, language="bash")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"[command failed] exit {e.returncode}")
        if e.stdout:
            st.code(e.stdout, language="bash")
        if e.stderr:
            st.code(e.stderr, language="bash")
        return False


tabs = st.tabs(
    [
        "Metrics",
        "Summaries",
        "Clusters",
        "Data",
        "RAG",
        "PEFT",
        "Efficiency",
        "Survey Data",
    ]
)

# Metrics
with tabs[0]:
    st.subheader("Pipeline status")
    mp = Path("results/metrics.json")
    if mp.exists():
        try:
            obj = json.loads(mp.read_text())
            st.json(obj)
        except Exception:
            st.code(mp.read_text(), language="json")
    else:
        st.info("No metrics yet. Run literature pipeline.")

    mp = Path("results/summary_metrics.json")
    if mp.exists():
        try:
            obj = json.loads(mp.read_text())
            st.json(obj)  # Render JSON directly
        except Exception:
            st.code(mp.read_text(), language="json")  # Fallback if malformed
    else:
        st.info("No metrics yet. Run literature pipeline.")

    # Export Final PDF
    st.divider()
    st.subheader("Export / Download")
    pdf = Path("reports/final_report.pdf")
    if pdf.exists():
        st.download_button(
            "📄 Download Final PDF", pdf.read_bytes(), file_name="final_report.pdf"
        )
    else:
        if st.button("Build Final PDF"):
            run_cmd([sys.executable, "-m", "src.reporting.report_generator"])
            st.success(
                "Built reports/final_report.pdf (refresh to see download button)."
            )

# Summaries
with tabs[1]:
    st.subheader("Generated summaries")
    p = Path("results/summaries.json")
    if p.exists():
        data = json.loads(p.read_text())
        for d in data:
            with st.expander(d["title"][:120]):
                st.markdown(
                    f"**Abstract (truncated):** {d['summary'][:260]}{'…' if len(d['summary'])>260 else ''}"
                )
                st.markdown(f"**Generated:** {d['generated']}")
    else:
        st.warning(
            "results/summaries.json not found. Run the literature steps from Metrics tab."
        )

# Clusters
with tabs[2]:
    st.subheader("PCA Cluster Projection")
    pca = Path("plots/literature_pca.npy")
    labs = Path("results/literature_clusters.json")
    if pca.exists() and labs.exists():
        X = np.load(pca)
        labels = json.loads(labs.read_text())["labels"]
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in sorted(set(labels)):
            idx = [i for i, y in enumerate(labels) if y == k]
            ax.scatter(X[idx, 0], X[idx, 1], s=35, alpha=0.85, label=f"Cluster {k}")
        ax.set_xlabel("PCA-1")
        ax.set_ylabel("PCA-2")
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig, clear_figure=True)
        png = Path("results/literature_clusters.png")
        if png.exists():
            st.image(str(png), caption="Saved PNG", use_container_width=True)
    else:
        st.info(
            "Run embedding step to create plots/literature_pca.npy and results/literature_clusters.json."
        )

# Data
with tabs[3]:
    st.subheader("Upload a CSV and run analysis")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    target_hint = st.text_input(
        "Optional target column (numeric for regression; 2 classes for t-test)",
        value="",
    )
    run_btn = st.button("Run Data Analysis")
    if uploaded is not None:
        data_dir = Path("data/uploads")
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "uploaded.csv"
        csv_path.write_bytes(uploaded.getbuffer())
        df_preview = load_csv_arrow_safe(csv_path)
        st.caption("Preview (Arrow-safe dtypes):")
        st.dataframe(df_preview.head(20))
        st.success(f"Saved -> {csv_path}")
        if run_btn:
            args = [
                sys.executable,
                "-m",
                "src.data_analysis.analyze_data",
                "--csv",
                str(csv_path),
            ]
            if target_hint.strip():
                args += ["--target", target_hint.strip()]
            ok = run_cmd(args)
            if ok:
                ok = run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "src.data_analysis.visualize_results",
                        "--csv",
                        str(csv_path),
                    ]
                )
            if ok:
                run_cmd([sys.executable, "-m", "src.data_analysis.llm_commentary"])
            if ok:
                st.success("Data analysis done. See artifacts below.")

    st.divider()
    st.subheader("Artifacts")
    res = Path("results")
    if (res / "data_analysis_summary.json").exists():
        st.markdown("**Summary JSON:**")
        st.json(json.loads((res / "data_analysis_summary.json").read_text()))
    vis = res / "visualizations"
    if vis.exists():
        for p in sorted(vis.glob("*.png")):
            st.image(str(p), caption=p.name, use_container_width=True)
    if (res / "data_analysis_commentary.md").exists():
        st.markdown((res / "data_analysis_commentary.md").read_text())

# RAG
with tabs[4]:
    st.subheader("RAG Search & QA")
    q = st.text_input(
        "Question", "What are practical techniques to accelerate LLM decoding?"
    )
    k = st.number_input("Top-k", 1, 20, rag_topk, step=1)
    colA, colB = st.columns(2)
    if colA.button("Build/Refresh Index"):
        run_cmd([sys.executable, "-m", "src.rag.build_index"])
        st.success("Index rebuilt.")
    if colB.button("Answer via RAG"):
        run_cmd(
            [sys.executable, "-m", "src.rag.rag_qa", "--question", q, "--k", str(k)]
        )
        out = Path("results/rag_qa.json")
        if out.exists():
            st.json(json.loads(out.read_text()))
            # factuality check
            run_cmd(
                [sys.executable, "-m", "src.rag.factuality_check", "--qa", str(out)]
            )
            st.caption("Factuality check:")
            st.json(json.loads(Path("results/rag_factuality.json").read_text()))
    st.divider()
    st.subheader("Hybrid Reranker demo")
    if st.button("Run hybrid demo"):
        run_cmd([sys.executable, "-m", "src.rag.hybrid_reranker"])
        st.json(json.loads(Path("results/rag/hybrid_search_demo.json").read_text()))
    st.caption("Multi-query expansion:")
    if st.button("Run multi-query expansion"):
        run_cmd([sys.executable, "-m", "src.rag.multiquery_expansion"])
        out = Path("results/rag_multiquery.json")
        if out.exists():
            st.json(json.loads(out.read_text()))

# PEFT
with tabs[5]:
    st.subheader("PEFT: train/eval/export")
    c1, c2, c3 = st.columns(3)
    if c1.button("Train LoRA (quick)"):
        run_cmd([sys.executable, "-m", "src.llm_opt.train_peft_run"])
    if c2.button("Eval LoRA"):
        run_cmd([sys.executable, "-m", "src.llm_opt.peft_eval"])
        peft_eval_path = Path("results/peft_eval.json")
        if peft_eval_path.exists():
            try:
                st.json(json.loads(peft_eval_path.read_text()))
            except Exception:
                st.code(peft_eval_path.read_text(), language="json")
        else:
            st.info("No eval yet.")
    if c3.button("Export merged"):
        run_cmd([sys.executable, "-m", "src.llm_opt.peft_export"])
    # live view
    if Path("experiments/peft_flan_t5/checkpoint-best").exists():
        st.success("Best checkpoint present.")
    else:
        st.info("Train to create checkpoint-best.")

# Efficiency
with tabs[6]:
    st.subheader("LLM Efficiency Dashboard")
    if st.button("Run efficiency suite"):
        run_cmd([sys.executable, "-m", "src.llm_opt.quantize_models"])
        run_cmd([sys.executable, "-m", "src.llm_opt.evaluate_efficiency"])
        run_cmd([sys.executable, "-m", "src.llm_opt.compare_results"])
        run_cmd([sys.executable, "-m", "src.llm_opt.plot_tradeoffs"])
        run_cmd([sys.executable, "-m", "src.llm_opt.eval_tokens_per_sec"])
        run_cmd([sys.executable, "-m", "src.llm_opt.batch_ablation"])
        run_cmd([sys.executable, "-m", "src.llm_opt.profiler_trace"])
        run_cmd([sys.executable, "-m", "src.llm_opt.int8_vs_fp16_eval"])
        run_cmd([sys.executable, "-m", "src.llm_opt.system_resource_report"])
    if Path("results/benchmark_table.csv").exists():
        st.markdown("**Benchmark Table**")
        st.dataframe(pd.read_csv("results/benchmark_table.csv"))
    if Path("plots/tradeoff_curves.png").exists():
        st.image(
            "plots/tradeoff_curves.png",
            caption="Accuracy vs. Latency",
            use_container_width=True,
        )
    for p in [
        "results/efficiency_metrics.csv",
        "results/tokens_per_sec.json",
        "results/batch_ablation.json",
        "results/int8_vs_fp16.csv",
        "results/system_resources.json",
    ]:
        if Path(p).exists():
            st.caption(p)
            st.json(
                json.loads(Path(p).read_text())
                if p.endswith(".json")
                else pd.read_csv(p).to_dict(orient="records")
            )

# Survey Data
with tabs[7]:
    st.subheader("Survey Data")
    up = st.file_uploader(
        "Upload survey CSV (e.g., Likert responses)", type=["csv"], key="a"
    )
    if up is not None:
        p = Path("data/uploads/survey.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(up.getbuffer())
        df = load_csv_arrow_safe(p)
        st.dataframe(df.head(20))

        st.caption("Basic descriptives:")

        numeric = df.select_dtypes(include=["number"])
        categoricals = df.select_dtypes(exclude=["number", "datetime"])
        datetimes = df.select_dtypes(include=["datetime"])

        if not numeric.empty:
            st.write("**Numeric columns**")
            st.dataframe(numeric.describe())

        if not categoricals.empty:
            st.write("**Categorical columns**")
            st.dataframe(categoricals.describe(include="all"))

        if not datetimes.empty:
            st.write("**Datetime columns**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "count": datetimes.count(),
                        "min": datetimes.min(),
                        "max": datetimes.max(),
                    }
                )
            )

        num_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]
        if num_cols:
            col = st.selectbox("Numeric column to plot", num_cols)
            series = df[col].dropna()

            if pd.api.types.is_bool_dtype(series):
                # Plot boolean as counts
                counts = series.value_counts().reindex([True, False], fill_value=0)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                counts.plot(kind="bar", ax=ax)
                ax.set_title(f"Counts — {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("count")
                st.pyplot(fig, clear_figure=True)
            else:
                # Standard numeric histogram
                fig, ax = plt.subplots(figsize=(6, 3.5))
                series.plot(kind="hist", bins=30, ax=ax)
                ax.set_title(f"Histogram — {col}")
                st.pyplot(fig, clear_figure=True)
