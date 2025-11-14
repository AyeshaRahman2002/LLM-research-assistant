# scripts/run_ui.sh
#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
echo "==> Launching Streamlit UI on http://localhost:8501"
exec streamlit run ui/app.py --server.port 8501 --server.headless true
