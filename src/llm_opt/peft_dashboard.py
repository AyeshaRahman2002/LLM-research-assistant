# src/llm_opt/peft_dashboard.py
import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PEFT Dashboard", layout="wide")
st.title("PEFT Dashboard")

ckpt = Path("experiments/peft_flan_t5/checkpoint-best")
if ckpt.exists():
    st.success(f"Best checkpoint found: {ckpt}")
    if (Path("results/peft_eval.json")).exists():
        st.subheader("Evaluation")
        st.json(json.loads(Path("results/peft_eval.json").read_text()))
else:
    st.info("Train PEFT first (run train_peft_run.py)")
