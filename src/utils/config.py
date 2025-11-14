# src/utils/config.py
import os
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_override = os.getenv("LLM_UI_MODEL_OVERRIDE")
    if model_override:
        cfg["summarization"]["models"] = [model_override]
    max_tokens = os.getenv("LLM_UI_MAX_TOKENS")
    if max_tokens:
        cfg["summarization"]["max_tokens"] = int(max_tokens)
    return cfg


def ensure_dirs(cfg: dict):
    for key in ("data_dir", "results_dir", "reports_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
