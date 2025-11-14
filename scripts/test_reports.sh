# scripts/test_reports.sh
#!/usr/bin/env bash
set -euo pipefail

echo "==> Report build sanity"
python3 -m src.reporting.report_generator

PDF="reports/final_report.pdf"
MD="reports/final_report.md"

if [[ -s "$PDF" ]]; then
  echo "DONE PDF exists and is non-empty: $PDF"
else
  echo "[ERROR] PDF missing or empty: $PDF" >&2
  exit 1
fi

if [[ -s "$MD" ]]; then
  echo "DONE Markdown exists and is non-empty: $MD"
else
  echo "[WARN] Markdown missing or empty: $MD"
fi
