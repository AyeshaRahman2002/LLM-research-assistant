# tests/test_ui_launch.py
import importlib
import os


def test_streamlit_app_imports(tmp_path, monkeypatch):
    """
    Ensure the Streamlit app module imports without raising.
    This does NOT start a server; it only verifies import-time safety.
    """
    os.chdir(tmp_path)
    # Headless / quiet
    monkeypatch.setenv("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    mod = importlib.import_module("ui.app")
    assert mod is not None
